import collections
import itertools
import os
import tempfile
from typing import Any, Dict, Tuple, Type, Union

import gym
import imageio
import numpy as np
import torch

from research.datasets.feedback_buffer import PairwiseComparisonDataset
from research.datasets.replay_buffer import ReplayBuffer
from research.networks.base import ActorCriticRewardPolicy
from research.processors.normalization import RunningMeanStd
from research.utils.utils import to_device, to_tensor

from .off_policy_algorithm import OffPolicyAlgorithm


class PEBBLE(OffPolicyAlgorithm):
    def __init__(
        self,
        *args,
        tau: float = 0.005,
        init_temperature: float = 0.1,
        critic_freq: int = 1,
        actor_freq: int = 1,
        target_freq: int = 2,
        bc_coeff=0.0,
        # all of the other kwargs for reward
        reward_freq: int = 128,
        reward_epochs: Union[int, float] = 10,
        max_feedback: int = 1000,
        init_feedback_size: int = 64,
        feedback_sample_multiplier: float = 10,
        reward_batch_size: int = 256,
        segment_size: int = 25,
        feedback_schedule: str = "constant",
        reset_reward_net: bool = False,
        reward_shift: float = 0,
        reward_scale: float = 1,
        num_uniform_feedback: int = 0,
        # Unsupervised Parameters
        unsup_steps: int = 0,
        k_nearest_neighbors: int = 5,
        unsup_batch_size: int = 512,
        num_unsup_batches: int = 20,
        normalize_state_entropy: bool = True,
        **kwargs,
    ):
        # Save values needed for network setup.
        self.init_temperature = init_temperature
        super().__init__(*args, **kwargs)
        assert isinstance(self.network, ActorCriticRewardPolicy)

        # SAC parameters
        self.tau = tau
        self.critic_freq = critic_freq
        self.actor_freq = actor_freq
        self.target_freq = target_freq
        self.bc_coeff = bc_coeff
        self.target_entropy = -np.prod(self.processor.action_space.low.shape)
        self.action_range = [
            float(self.processor.action_space.low.min()),
            float(self.processor.action_space.high.max()),
        ]

        # Timing parameters
        self.reward_freq = reward_freq
        self.unsup_steps = unsup_steps

        # Feedback parameters
        self.segment_size = segment_size
        self.max_feedback = max_feedback
        self.init_feedback_size = init_feedback_size
        self.feedback_schedule = feedback_schedule
        self.feedback_sample_multiplier = feedback_sample_multiplier
        self.num_uniform_feedback = num_uniform_feedback

        # Reward parameters
        self.reward_shift = reward_shift
        self.reward_scale = reward_scale
        self.reward_epochs = reward_epochs
        self.reward_batch_size = reward_batch_size
        self.reward_criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
        self.reset_reward_net = reset_reward_net

        # Unsupervised parameters
        self.unsup_batch_size = unsup_batch_size
        self.num_unsup_batches = num_unsup_batches
        self.k_nearest_neighbors = k_nearest_neighbors
        self.normalize_state_entropy = normalize_state_entropy
        self.entropy_stats = RunningMeanStd(shape=()).to(self.device)

        # Initialize parameters
        self._total_feedback = 0
        self._saved_recent_visualizations = True
        self._last_feedback_step = None
        # Extra metrics for human labeling
        self._skipped_queries = 0
        self._correct_queries = 0

        # Checkpointing
        self.checkpoint_path = kwargs["checkpoint"] if "checkpoint" in kwargs else None
        if self.checkpoint_path is None and self.reset_reward_net:
            # Get a temporary file to store the initial reward model checkpoint
            tmp_dir = tempfile.mkdtemp()
            self.checkpoint_path = os.path.join(tmp_dir, "reward.pt")
            self.save(tmp_dir, "reward")

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

        # Setup the log alpha
        log_alpha = torch.tensor(np.log(self.init_temperature), dtype=torch.float).to(self.device)
        self.log_alpha = torch.nn.Parameter(log_alpha, requires_grad=True)

    def setup_optimizers(self) -> None:
        # Default optimizer initialization
        self.optim["actor"] = self.optim_class(self.network.actor.parameters(), **self.optim_kwargs)
        # Update the encoder with the critic.
        critic_params = itertools.chain(self.network.critic.parameters(), self.network.encoder.parameters())
        self.optim["critic"] = self.optim_class(critic_params, **self.optim_kwargs)
        self.optim["log_alpha"] = self.optim_class([self.log_alpha], **self.optim_kwargs)
        self.optim["reward"] = self.optim_class(self.network.reward.parameters(), **self.optim_kwargs)

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
            capacity=self.max_feedback + 1,
            batch_size=self.reward_batch_size,
        )
        self.feedback_dataloader = torch.utils.data.DataLoader(
            self.feedback_dataset, batch_size=None, num_workers=0, pin_memory=(self.device.type == "cuda")
        )

    def _update_critic(self, batch: Dict) -> Dict:
        with torch.no_grad():
            dist = self.network.actor(batch["next_obs"])
            next_action = dist.rsample()
            log_prob = dist.log_prob(next_action).sum(dim=-1)
            target_qs = self.target_network.critic(batch["next_obs"], next_action)
            target_v = torch.min(target_qs, dim=0)[0] - self.alpha.detach() * log_prob
            reward = self.network.reward(batch["obs"], batch["action"]).mean(dim=0)  # Should be shape (B,)
            reward = self.reward_scale * reward + self.reward_shift
            target_q = reward + batch["discount"] * target_v

        qs = self.network.critic(batch["obs"], batch["action"])
        q_loss = torch.nn.functional.mse_loss(qs, target_q.expand(qs.shape[0], -1), reduction="none").mean()

        self.optim["critic"].zero_grad(set_to_none=True)
        q_loss.backward()
        self.optim["critic"].step()

        return dict(q_loss=q_loss.item(), target_q=target_q.mean().item())

    def _update_actor_and_alpha(self, batch: Dict) -> Dict:
        obs = batch["obs"].detach()  # Detach the encoder so it isn't updated.
        dist = self.network.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        qs = self.network.critic(obs, action)
        q = torch.min(qs, dim=0)[0]

        actor_loss = (self.alpha.detach() * log_prob - q).mean()
        if self.bc_coeff > 0.0:
            bc_loss = -dist.log_prob(batch["action"]).sum(dim=-1).mean()  # Simple NLL loss.
            actor_loss = actor_loss + self.bc_coeff * bc_loss

        self.optim["actor"].zero_grad(set_to_none=True)
        actor_loss.backward()
        self.optim["actor"].step()
        entropy = -log_prob.mean()

        # Update the learned temperature
        self.optim["log_alpha"].zero_grad(set_to_none=True)
        alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()
        alpha_loss.backward()
        self.optim["log_alpha"].step()

        return dict(
            actor_loss=actor_loss.item(),
            entropy=entropy.item(),
            alpha_loss=alpha_loss.item(),
            alpha=self.alpha.detach().item(),
        )

    def _update_critic_unsup(self, batch: Dict) -> Dict:
        # Compute the state entropy
        assert not self.env.observation_space.dtype == np.uint8, "Image spaces not supported for unsup"
        with torch.no_grad():
            dist = self.network.actor(batch["next_obs"])
            next_action = dist.rsample()
            log_prob = dist.log_prob(next_action).sum(dim=-1)
            target_qs = self.target_network.critic(batch["next_obs"], next_action)
            target_v = torch.min(target_qs, dim=0)[0] - self.alpha.detach() * log_prob

            # Reward is my state entropy
            dists = []
            for _ in range(self.num_unsup_batches):
                full_obs = self.dataset.sample(batch_size=self.unsup_batch_size)["obs"]
                dist = torch.norm(batch["obs"][:, None, :] - full_obs[None, :, :], dim=-1, p=2)
                dists.append(dist)
            dists = torch.cat(dists, dim=1)
            state_entropy = torch.kthvalue(dists, k=self.k_nearest_neighbors + 1, dim=1).values
            self.entropy_stats.update(state_entropy)
            if self.normalize_state_entropy:
                state_entropy = state_entropy / self.entropy_stats.std
            target_q = state_entropy + batch["discount"] * target_v

        qs = self.network.critic(batch["obs"], batch["action"])
        q_loss = (
            torch.nn.functional.mse_loss(qs, target_q.expand(qs.shape[0], -1)).mean(dim=-1).sum()
        )  # averages over the ensemble. No for loop!

        self.optim["critic"].zero_grad(set_to_none=True)
        q_loss.backward()
        self.optim["critic"].step()

        return dict(unsup_q_loss=q_loss.item(), unsup_target_q=target_q.mean().item())

    def _get_reward_logits(self, batch: Dict) -> torch.Tensor:
        B, S = batch["obs_1"].shape[:2]  # Get the batch size and the segment length
        flat_obs_shape = (B * S,) + batch["obs_1"].shape[2:]
        flat_action_shape = (B * S,) + batch["action_1"].shape[2:]
        r_hat1 = self.network.reward(batch["obs_1"].view(*flat_obs_shape), batch["action_1"].view(flat_action_shape))
        r_hat2 = self.network.reward(batch["obs_2"].view(*flat_obs_shape), batch["action_2"].view(flat_action_shape))
        E, B_times_S = r_hat1.shape
        assert B_times_S == B * S, "Shapes were incorrect"
        r_hat1 = r_hat1.view(E, B, S).sum(dim=2)  # Now should be (E, B)
        r_hat2 = r_hat2.view(E, B, S).sum(dim=2)  # Now should be (E, B)
        logits = r_hat2 - r_hat1
        return logits

    def _oracle_label(self, batch: Dict) -> Tuple[np.ndarray, Dict]:
        label = 1.0 * (batch["reward_1"] < batch["reward_2"])
        return label, {}

    def _get_queries(self, batch_size):
        batch = self.dataset.sample(batch_size=2 * batch_size, stack=self.segment_size, pad=0)
        # Compute the discounted reward across each segment to be used for oracle labels
        returns = np.sum(batch["reward"], axis=1)  # Use non-discounted to match PEBBLE.
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
                logits = self._get_reward_logits(to_device(to_tensor(queries), self.device))
                probs = torch.sigmoid(logits)
                probs = probs.cpu().numpy()  # Shape (E, B)
            disagreement = np.std(probs, axis=0)  # Compute along the ensemble axis
            top_k_index = (-disagreement).argsort()[:batch_size]
            # pare down the batch by the topk index
            queries = {k: v[top_k_index] for k, v in queries.items()}

        # Label the queries
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

    def _update_reward_model(self) -> Dict:
        # Reset the weights and optim of the reward network if wanted.
        if self.reset_reward_net:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            reward_params = collections.OrderedDict(
                [(k[7:], v) for k, v in checkpoint["network"].items() if k.startswith("reward")]
            )
            self.network.reward.load_state_dict(reward_params)

        epochs = 0
        while True:
            losses, accuracies = [], []
            for batch in self.feedback_dataloader:
                batch = to_device(batch, self.device)
                self.optim["reward"].zero_grad(set_to_none=True)
                logits = self._get_reward_logits(batch)  # Shape (E, B)
                labels = batch["label"].float().unsqueeze(0).expand(logits.shape[0], -1)  # Shape (E, B)
                loss = self.reward_criterion(logits, labels).mean(dim=-1).sum(dim=0)  # Average on B, sum on E
                loss.backward()
                self.optim["reward"].step()

                losses.append(loss.item())
                # Compute the accuracy
                with torch.no_grad():
                    pred = (logits > 0).float()
                    accuracy = (pred == labels).float().mean()
                    accuracies.append(accuracy.item())
            epochs += 1
            mean_acc = np.mean(accuracies)
            if isinstance(self.reward_epochs, int) and epochs == self.reward_epochs:
                break
            elif isinstance(self.reward_epochs, float) and mean_acc >= self.reward_epochs:
                # Train until we reach a specific reward threshold
                break
            elif mean_acc > 0.97:
                break
            elif epochs > 25000:
                # We have run over 25k epochs, break anyways. For low feedback this is around 25k batches anyways
                break

        # Return the metrics, handling initial cases where the feedback buffer is empty.
        metrics = dict()
        if len(losses) > 0:
            metrics["reward_loss"] = np.mean(losses)
            metrics["reward_accuracy"] = np.mean(accuracies)
        return metrics

    def train_step(self, batch: Dict, step: int, total_steps: int) -> Dict:
        all_metrics = {}

        if "obs" not in batch or step < self.random_steps:
            return all_metrics

        # Determine how we update the critic based on whether or not we are doing unsupervised
        if step < self.random_steps + self.unsup_steps:
            critic_update_fn = self._update_critic_unsup
        elif step == self.random_steps + self.unsup_steps:
            self.network.reset_critic()
            self.target_network.critic.load_state_dict(self.network.critic.state_dict())
            critic_update_fn = self._update_critic
            print("[pebble] Reset the critic.")
        else:
            critic_update_fn = self._update_critic

        batch["obs"] = self.network.encoder(batch["obs"])
        with torch.no_grad():
            batch["next_obs"] = self.target_network.encoder(batch["next_obs"])

        if step % self.critic_freq == 0:
            metrics = critic_update_fn(batch)
            all_metrics.update(metrics)

        if step % self.actor_freq == 0:
            metrics = self._update_actor_and_alpha(batch)
            all_metrics.update(metrics)

        if step % self.target_freq == 0:
            # Only update the critic and encoder for speed. Ignore the actor.
            with torch.no_grad():
                for param, target_param in zip(
                    self.network.encoder.parameters(), self.target_network.encoder.parameters()
                ):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                for param, target_param in zip(
                    self.network.critic.parameters(), self.target_network.critic.parameters()
                ):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # Now check if we should udpate the feedback
        if (
            (self._last_feedback_step is None or (step - self._last_feedback_step) % self.reward_freq == 0)
            and self._total_feedback < self.max_feedback
            and step >= self.random_steps + self.unsup_steps
        ):
            self._last_feedback_step = step
            # First collect feedback
            metrics = self._collect_feedback(step, total_steps)
            all_metrics.update(metrics)
            # Next update the model
            metrics = self._update_reward_model()
            all_metrics.update(metrics)

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
            action = dist.sample() if sample else dist.loc
            action = action.clamp(*self.action_range)
            return action

    def _get_train_action(self, step: int, total_steps: int) -> np.ndarray:
        batch = dict(obs=self._current_obs)
        with torch.no_grad():
            action = self.predict(batch, is_batched=False, sample=True)
        return action
