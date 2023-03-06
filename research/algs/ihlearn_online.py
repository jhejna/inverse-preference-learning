import itertools
from typing import Any, Dict, Optional, Tuple, Type

import gym
import imageio
import numpy as np
import torch

from research.datasets.feedback_buffer import PairwiseComparisonDataset
from research.datasets.replay_buffer import ReplayBuffer
from research.networks.base import ActorCriticPolicy
from research.utils.utils import to_device, to_tensor

from .off_policy_algorithm import OffPolicyAlgorithm


class IHLearnOnline(OffPolicyAlgorithm):
    def __init__(
        self,
        *args,
        tau: float = 0.005,
        init_temperature: float = 0.1,
        target_freq: int = 2,
        bc_coeff=0.0,
        learn_temperature: bool = True,
        target_entropy: Optional[float] = None,
        # all of the other kwargs for reward
        reward_freq: int = 5000,
        max_feedback: int = 1000,
        init_feedback_size: int = 64,
        feedback_sample_multiplier: float = 10,
        reward_batch_size: int = 256,
        segment_size: int = 25,
        subsample_size: Optional[int] = 15,
        chi2_coeff: float = 0.5,
        feedback_schedule: str = "constant",
        num_uniform_feedback: int = 0,
        use_min_target: bool = False,
        **kwargs,
    ):
        # Save values needed for optim setup.
        self.init_temperature = init_temperature
        self.learn_temperature = learn_temperature
        super().__init__(*args, **kwargs)
        assert isinstance(self.network, ActorCriticPolicy)

        # SAC parameters
        self.tau = tau
        self.target_freq = target_freq
        self.bc_coeff = bc_coeff
        self.target_entropy = (
            -np.prod(self.processor.action_space.low.shape) if target_entropy is None else -target_entropy
        )
        self.action_range = [
            float(self.processor.action_space.low.min()),
            float(self.processor.action_space.high.max()),
        ]

        # Timing parameters
        self.reward_freq = reward_freq

        # Feedback parameters
        self.segment_size = segment_size
        self.max_feedback = max_feedback
        self.init_feedback_size = init_feedback_size
        self.feedback_schedule = feedback_schedule
        self.feedback_sample_multiplier = feedback_sample_multiplier
        self.num_uniform_feedback = num_uniform_feedback

        # Reward parameters
        self.reward_batch_size = reward_batch_size
        self.reward_criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
        self.chi2_coeff = chi2_coeff
        self.subsample_size = subsample_size
        self.use_min_target = use_min_target

        # Initialize parameters
        self._total_feedback = 0
        self._saved_recent_visualizations = True
        self._last_feedback_step = None
        # Extra metrics for human labeling
        self._skipped_queries = 0
        self._correct_queries = 0

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def setup_network(self, network_class: Type[torch.nn.Module], network_kwargs: Dict) -> None:
        # Setup network and target network
        self.network = network_class(
            self.processor.observation_space, self.processor.action_space, **network_kwargs
        ).to(self.device)
        self.target_network = network_class(
            self.processor.observation_space, self.processor.action_space, **network_kwargs
        ).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        for param in self.target_network.parameters():
            param.requires_grad = False

        log_alpha = torch.tensor(np.log(self.init_temperature), dtype=torch.float).to(self.device)
        self.log_alpha = torch.nn.Parameter(log_alpha, requires_grad=self.learn_temperature)

    def setup_optimizers(self) -> None:
        # Default optimizer initialization
        keys = ("actor", "critic", "log_alpha")
        default_kwargs = {}
        for key, value in self.optim_kwargs.items():
            if key not in keys:
                default_kwargs[key] = value
            else:
                assert isinstance(value, dict), "Special keys must be kwarg dicts"

        actor_kwargs = default_kwargs.copy()
        actor_kwargs.update(self.optim_kwargs.get("actor", dict()))
        self.optim["actor"] = self.optim_class(self.network.actor.parameters(), **actor_kwargs)

        # Update the encoder with the critic.
        critic_params = itertools.chain(self.network.critic.parameters(), self.network.encoder.parameters())
        critic_kwargs = default_kwargs.copy()
        critic_kwargs.update(self.optim_kwargs.get("critic", dict()))
        self.optim["critic"] = self.optim_class(critic_params, **critic_kwargs)

        if self.learn_temperature:
            log_alpha_kwargs = default_kwargs.copy()
            log_alpha_kwargs.update(self.optim_kwargs.get("log_alpha", dict()))
            self.optim["log_alpha"] = self.optim_class([self.log_alpha], **log_alpha_kwargs)

    def setup_train_dataset(self) -> None:
        super().setup_train_dataset()
        assert isinstance(self.dataset, ReplayBuffer), "Must use replay buffer for PEBBLE"
        assert self.dataset.distributed == False, "Cannot use distributed replay buffer with PEBBLE"
        # Note that the dataloader for the reward model runs on a single thread!
        self.feedback_dataset = PairwiseComparisonDataset(
            self.env.observation_space,
            self.env.action_space,
            discount=self.dataset.discount,
            nstep=self.dataset.nstep,
            segment_size=self.segment_size,
            subsample_size=self.subsample_size,
            capacity=self.max_feedback + 1,
            batch_size=self.reward_batch_size,
        )
        self.feedback_dataloader = torch.utils.data.DataLoader(
            self.feedback_dataset, batch_size=None, num_workers=0, pin_memory=(self.device.type == "cuda")
        )
        self.feedback_iterator = iter(self.feedback_dataloader)

    def _get_reward(self, batch: Dict) -> torch.Tensor:
        obs = torch.cat([batch["obs_1"], batch["obs_2"]], dim=0)  # (B, S+1)
        action = torch.cat([batch["action_1"], batch["action_2"]], dim=0)  # (B, S+1)
        # Compute shapes and add everything to the batch dimension
        B, S = obs.shape[:2]
        S -= 1  # Subtract one for the next obs sequence.
        flat_obs_shape = (B * S,) + obs.shape[2:]
        flat_action_shape = (B * S,) + action.shape[2:]
        next_obs = obs[:, 1:].reshape(flat_obs_shape)
        obs = obs[:, :-1].reshape(flat_obs_shape)
        action = action[:, :-1].reshape(flat_action_shape)

        qs = self.network.critic(obs, action)
        with torch.no_grad():
            dist = self.network.actor(next_obs)
            next_action = dist.sample()
            next_vs = self.target_network.critic(next_obs, next_action)
            if self.use_min_target:
                next_vs = torch.min(next_vs, dim=0)[0].unsqueeze(0)
        reward = qs - self.dataset.discount * next_vs
        # view reward again in the correct shape
        E, B_times_S = reward.shape
        assert B_times_S == B * S, "Shapes were incorrect"
        reward = reward.view(E, B, S)
        r1, r2 = torch.chunk(reward, 2, dim=1)  # Should return two (E, B, S)
        return r1, r2

    def _oracle_label(self, batch: Dict) -> Tuple[np.ndarray, Dict]:
        label = 1.0 * (batch["reward_1"] < batch["reward_2"])
        return label, {}

    def _get_queries(self, batch_size):
        batch = self.dataset.sample(batch_size=2 * batch_size, stack=self.segment_size, pad=0)
        # Compute the discounted reward across each segment to be used for oracle labels
        returns = np.sum(batch["reward"] * np.power(self.dataset.discount, np.arange(batch["reward"].shape[1])), axis=1)
        segment_batch = dict(
            obs_1=batch["obs"][:batch_size],
            obs_2=batch["obs"][batch_size:],
            action_1=batch["action"][:batch_size],
            action_2=batch["action"][batch_size:],
            reward_1=returns[:batch_size],
            reward_2=returns[batch_size:],
        )
        if "state" in batch:
            segment_batch["state_1"] = batch["state"][:batch_size]
            segment_batch["state_2"] = batch["state"][batch_size:]
        del batch  # ensure memory is freed
        return segment_batch

    def _collect_feedback(self, step, total_steps) -> Dict:
        # Compute the amount of feedback to collect
        if self.feedback_schedule == "linear":
            batch_size = int(self.init_feedback_size * (total_steps - step) / total_steps)
        elif self.feedback_schedule == "constant":
            batch_size = self.init_feedback_size
        else:
            raise ValueError("Invalid Feedback Schedule Specified.")
        feedback_left = self.max_feedback - self._total_feedback
        batch_size = min(batch_size, feedback_left)
        assert batch_size > 0, "Called _collect_feedback when we have no more budget left."

        if self._total_feedback == 0 or self._total_feedback < self.num_uniform_feedback:
            # Collect segments for the initial part.
            queries = self._get_queries(batch_size)
        else:
            # Else, collect segments via disagreement
            queries = self._get_queries(batch_size=int(batch_size * self.feedback_sample_multiplier))
            # Compute disagreement via the ensemble
            with torch.no_grad():
                r1, r2 = self._get_reward(to_device(to_tensor(queries), self.device))
                logits = r2.sum(dim=2) - r1.sum(dim=2)  # Sum across sequence dim
                probs = torch.sigmoid(logits)
                probs = probs.cpu().numpy()  # Shape (E, B)
            disagreement = np.std(probs, axis=0)  # Compute along the ensemble axis
            top_k_index = (-disagreement).argsort()[:batch_size]
            # pare down the batch by the topk index
            queries = {k: v[top_k_index] for k, v in queries.items()}

        labels, metrics = self._oracle_label(queries)

        feedback_added = labels.shape[0]
        self._total_feedback += feedback_added
        metrics["feedback"] = self._total_feedback
        metrics["feedback_this_itr"] = feedback_added

        if feedback_added == 0:
            return metrics

        # Save the most recent queries for visualizations.
        self._recent_feedback = (queries, labels)
        self._saved_recent_visualizations = False

        # If we use human labels we can skip queries. We thus need to filter out any skipped queries.
        # This is done after updating metrics to insure that skipped queries are counted towards the total.
        valid_idxs = labels != -1
        if np.sum(valid_idxs) < labels.shape[0]:
            queries = {k: v[valid_idxs] for k, v in queries.items()}
            labels = labels[valid_idxs]

        self.feedback_dataset.add(queries, labels)
        return metrics

    def train_step(self, batch: Dict, step: int, total_steps: int) -> Dict:
        all_metrics = {}

        if "obs" not in batch or step < self.random_steps:
            return all_metrics

        if (
            (self._last_feedback_step is None or (step - self._last_feedback_step) % self.reward_freq == 0)
            and self._total_feedback < self.max_feedback
            and step >= self.random_steps
        ):
            self._last_feedback_step = step
            # First collect feedback
            metrics = self._collect_feedback(step, total_steps)
            all_metrics.update(metrics)
            # Delete and re-create the feedback iterator with new data.
            del self.feedback_iterator
            self.feedback_iterator = iter(self.feedback_dataloader)
        elif len(self.feedback_dataset) == 0:
            return all_metrics

        feedback_batch = next(self.feedback_iterator, None)
        if feedback_batch is None:
            self.feedback_iterator = iter(self.feedback_dataloader)
            feedback_batch = next(self.feedback_iterator, None)

        # Get the batch of feedback data.
        feedback_batch = to_device(feedback_batch, self.device)

        r1, r2 = self._get_reward(feedback_batch)
        logits = r2.sum(dim=2) - r1.sum(dim=2)  # Sum across sequence dim, (E, B)
        labels = feedback_batch["label"].float().unsqueeze(0).expand(logits.shape[0], -1)  # Shape (E, B)
        assert labels.shape == logits.shape
        q_loss = self.reward_criterion(logits, labels).mean()

        chi2_loss = (
            1 / (8 * self.chi2_coeff) * ((r1**2).mean() + (r2**2).mean())
        )  # Turn 1/4 to 1/8 because we sum over both.

        self.optim["critic"].zero_grad(set_to_none=True)
        (q_loss + chi2_loss).backward()
        self.optim["critic"].step()

        # Now update the actor.
        # Select a random sequence index to use for the feedback batches
        B, S = feedback_batch["obs_1"].shape[:2]
        idx = torch.randint(low=0, high=S, size=(B,), device=self.device, dtype=torch.long)
        arange = torch.arange(0, B, device=self.device, dtype=torch.long)
        obs = torch.cat(
            [batch["obs"], feedback_batch["obs_1"][arange, idx], feedback_batch["obs_2"][arange, idx]], dim=0
        )
        dist = self.network.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        qs = self.network.critic(obs, action)
        q = torch.mean(qs, dim=0)[0]  # Note: changed to mean from min.
        actor_loss = (self.alpha.detach() * log_prob - q).mean()
        if self.bc_coeff > 0.0:
            bc_loss = -dist.log_prob(batch["action"]).sum(dim=-1).mean()  # Simple NLL loss.
            actor_loss = actor_loss + self.bc_coeff * bc_loss

        self.optim["actor"].zero_grad(set_to_none=True)
        actor_loss.backward()
        self.optim["actor"].step()
        entropy = -log_prob.mean()

        # Update the learned temperature
        if self.learn_temperature:
            self.optim["log_alpha"].zero_grad(set_to_none=True)
            alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()
            alpha_loss.backward()
            self.optim["log_alpha"].step()

        # update the metrics
        all_metrics.update(
            dict(
                q_loss=q_loss.item(),
                chi2_loss=chi2_loss.item(),
                actor_loss=actor_loss.item(),
                entropy=entropy.item(),
                alpha=self.alpha.detach().item(),
                adv=r1.mean().item(),
            )
        )

        if step % self.target_freq == 0:
            # Only update the critic and encoder for speed. Ignore the actor.
            with torch.no_grad():
                for param, target_param in zip(self.network.parameters(), self.target_network.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return all_metrics

    def validation_extras(self, path: str, step: int) -> Dict:
        if self._saved_recent_visualizations:
            return {}
        self._saved_recent_visualizations = True
        return {}

    def _predict(self, batch: Any, sample: bool = False) -> torch.Tensor:
        with torch.no_grad():
            z = self.network.encoder(batch["obs"])
            dist = self.network.actor(z)
            if isinstance(dist, torch.distributions.Distribution):
                action = dist.sample() if sample else dist.loc
            elif torch.is_tensor(dist):
                action = dist
            else:
                raise ValueError("Invalid policy output")
            action = action.clamp(*self.action_range)
            return action

    def _get_train_action(self, step: int, total_steps: int) -> np.ndarray:
        batch = dict(obs=self._current_obs)
        with torch.no_grad():
            action = self.predict(batch, is_batched=False, sample=True)
        return action
