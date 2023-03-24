import itertools
from typing import Any, Dict, Optional, Tuple, Type

import gym
import imageio
import numpy as np
import torch

from research.datasets.feedback_buffer import PairwiseComparisonDataset
from research.datasets.replay_buffer import ReplayBuffer
from research.networks.base import ActorCriticPolicy
from research.utils import utils

from .off_policy_algorithm import OffPolicyAlgorithm


class IPLearnBase(OffPolicyAlgorithm):
    """
    This is the general IP-Learn algorithm for online learning
    The main differences between this and the offline implementation are:
    """

    def __init__(
        self,
        *args,
        tau: float = 0.005,
        target_freq: int = 2,
        use_min_target: bool = True,
        chi2_coeff: float = 0.5,
        feedback_proportion: Optional[float] = None,
        reward_freq: int = 5000,
        max_feedback: int = 1000,
        init_feedback_size: int = 64,
        feedback_sample_multiplier: float = 10,
        feedback_schedule: str = "constant",
        num_uniform_feedback: int = 0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert isinstance(self.network, ActorCriticPolicy)
        # General IP-Learn parameters
        self.tau = tau
        self.target_freq = target_freq
        self.use_min_target = use_min_target
        self.chi2_coeff = chi2_coeff
        self.feedback_proportion = feedback_proportion
        self.action_range = [
            float(self.processor.action_space.low.min()),
            float(self.processor.action_space.high.max()),
        ]

        # Online Feedback parameters
        self.reward_freq = reward_freq
        self.max_feedback = max_feedback
        self.init_feedback_size = init_feedback_size
        self.feedback_schedule = feedback_schedule
        self.feedback_sample_multiplier = feedback_sample_multiplier
        self.num_uniform_feedback = num_uniform_feedback
        self.reward_criterion = torch.nn.BCEWithLogitsLoss(reduction="none")

        # Initialize feedback parameters
        self._total_feedback = 0
        self._last_feedback_step = None

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

    def _oracle_label(self, batch: Dict) -> Tuple[np.ndarray, Dict]:
        label = 1.0 * (batch["reward_1"] < batch["reward_2"])
        return label, {}

    def _get_queries(self, batch_size):
        segment_size = self.dataset.feedback_dataset.segment_size
        batch = self.dataset.replay_buffer.sample(batch_size=2 * batch_size, stack=segment_size, pad=0)
        returns = np.sum(batch["reward"], axis=1)  # Note: use undiscounted reward to make sub-sections equivalent.
        segment_batch = dict(
            obs_1=batch["obs"][:batch_size],
            obs_2=batch["obs"][batch_size:],
            action_1=batch["action"][:batch_size],
            action_2=batch["action"][batch_size:],
            reward_1=returns[:batch_size],
            reward_2=returns[batch_size:],
        )
        del batch  # ensure memory is freed
        return segment_batch

    def _collect_feedback(self, step, total_steps) -> Dict:
        all_metrics = {}
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
            queries = utils.to_device(utils.to_tensor(queries), self.device)
            # Compute disagreement via the ensemble
            with torch.no_grad():
                # We need to repeat the steps for reward computation with the network here
                # see train_step for more comments
                B_fb, S_fb = queries["obs_1"].shape[:2]
                S_fb -= 1  # Subtract one for the next_obs offset
                flat_obs_fb_shape = (B_fb * S_fb,) + queries["obs_1"].shape[2:]
                flat_action_fb_shape = (B_fb * S_fb,) + queries["action_1"].shape[2:]

                # Construct obs, action, next_obs batches
                obs = torch.cat(
                    [
                        queries["obs_1"][:, :-1].reshape(*flat_obs_fb_shape),
                        queries["obs_2"][:, :-1].reshape(*flat_obs_fb_shape),
                    ],
                    dim=0,
                )
                action = torch.cat(
                    [
                        queries["action_1"][:, :-1].reshape(*flat_action_fb_shape),
                        queries["action_2"][:, :-1].reshape(*flat_action_fb_shape),
                    ],
                    dim=0,
                )
                next_obs = torch.cat(
                    [
                        queries["obs_1"][:, 1:].reshape(*flat_obs_fb_shape),
                        queries["obs_2"][:, 1:].reshape(*flat_obs_fb_shape),
                    ],
                    dim=0,
                )

                # Apply Encoders
                obs = self.network.encoder(obs)
                next_obs = self.target_network.encoder(next_obs)

                # Compute reward
                qs = self.network.critic(obs, action)
                reward = qs - self.dataset.replay_buffer.discount * self._compute_next_V(next_obs)
                r1, r2 = reward.chunk(reward, 2, dim=1)  # Shape (E, B_fb * S_fb)
                E, B_times_S = r1.shape
                assert B_times_S == B_fb * S_fb

                r1 = r1.view(E, B_fb, S_fb)
                r2 = r2.view(E, B_fb, S_fb)
                logits = r2.sum(dim=2) - r1.sum(dim=2)  # Sum across sequence dim

                probs = torch.sigmoid(logits).cpu().numpy()  # Shape (E, B)

            disagreement = np.std(probs, axis=0)  # Compute along the ensemble axis
            top_k_index = (-disagreement).argsort()[:batch_size]
            # pare down the batch by the topk index
            queries = {k: v[top_k_index] for k, v in queries.items()}

        labels, metrics = self._oracle_label(queries)
        all_metrics.update(metrics)

        feedback_added = labels.shape[0]
        self._total_feedback += feedback_added
        all_metrics["feedback"] = self._total_feedback
        all_metrics["feedback_this_itr"] = feedback_added

        if feedback_added == 0:
            return all_metrics

        self.dataset.feedback_dataset.add(queries, labels)
        return all_metrics

    def _compute_next_V(self, next_obs):
        raise NotImplementedError

    def _update_actor(self, obs, action, qs):
        raise NotImplementedError

    def setup(self):
        super().setup()
        self._data_pts_seen = 0
        self._flops = 0

    def train_step(self, batch: Dict, step: int, total_steps: int) -> Dict:
        all_metrics = {}
        replay_batch, feedback_batch = batch

        if step < self.random_steps:
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

        if feedback_batch is None:  # We need a feedback batch to update IP-Learn
            return all_metrics

        using_replay_batch = replay_batch is not None and "obs" in replay_batch

        # We need to assemble all of the observation, next observations, and their splits to compute reward values.
        # First collect all the shapes
        B_fb, S_fb = feedback_batch["obs_1"].shape[:2]
        S_fb -= 1  # Subtract one for the next_obs offset
        flat_obs_fb_shape = (B_fb * S_fb,) + feedback_batch["obs_1"].shape[2:]
        flat_action_fb_shape = (B_fb * S_fb,) + feedback_batch["action_1"].shape[2:]
        B_r = replay_batch["obs"].shape[0] if using_replay_batch else 0

        # Compute the split over observations between feedback and replay batches
        split = [B_fb * S_fb, B_fb * S_fb, B_r]
        B_total = split[0] + split[1] + split[2]
        # Construct one large batch for each of obs, action, next obs, discount
        obs = torch.cat(
            [
                feedback_batch["obs_1"][:, :-1].reshape(*flat_obs_fb_shape),
                feedback_batch["obs_2"][:, :-1].reshape(*flat_obs_fb_shape),
                *((replay_batch["obs"],) if using_replay_batch else ()),
            ],
            dim=0,
        )
        action = torch.cat(
            [
                feedback_batch["action_1"][:, :-1].reshape(*flat_action_fb_shape),
                feedback_batch["action_2"][:, :-1].reshape(*flat_action_fb_shape),
                *((replay_batch["action"],) if using_replay_batch else ()),
            ],
            dim=0,
        )
        next_obs = torch.cat(
            [
                feedback_batch["obs_1"][:, 1:].reshape(*flat_obs_fb_shape),
                feedback_batch["obs_2"][:, 1:].reshape(*flat_obs_fb_shape),
                *((replay_batch["next_obs"],) if using_replay_batch else ()),
            ],
            dim=0,
        )
        # Make the fb discount exactly match the flat_shape.
        discount = feedback_batch["discount"].unsqueeze(1).repeat(1, S_fb).flatten().repeat(2)
        if using_replay_batch:
            discount = torch.cat((discount, replay_batch["discount"]), dim=0)

        # Apply Encoders
        obs = self.network.encoder(obs)
        with torch.no_grad():
            next_obs = self.target_network.encoder(next_obs)

        qs = self.network.critic(obs, action)
        reward = qs - discount * self._compute_next_V(next_obs)
        assert reward.shape[1] == B_total
        E = reward.shape[0]

        # Now re-chunk everything to get the logits
        r1, r2, rr = torch.split(reward, split, dim=1)  # Now slips over dim 1 because of ensemble.
        r1, r2 = r1.view(E, B_fb, S_fb), r2.view(E, B_fb, S_fb)
        logits = r2.sum(dim=2) - r1.sum(dim=2)  # Sum across sequence dim, (E, B_fb)

        # Compute the Q-loss over the imitation data
        labels = feedback_batch["label"].float().unsqueeze(0).expand(E, -1)  # Shape (E, B)
        assert labels.shape == logits.shape
        q_loss = self.reward_criterion(logits, labels).mean()

        # Compute the Chi2 Loss over EVERYTHING, including replay data
        if self.feedback_proportion is not None and B_r > 0:
            # This tries to balance the loss over data points.
            chi2_loss_fb = 1 / (8 * self.chi2_coeff) * (torch.square(r1).mean() + torch.square(r2).mean())
            chi2_loss_replay = 1 / (4 * self.chi2_coeff) * torch.square(rr).mean()
            chi2_loss = self.feedback_proportion * chi2_loss_fb + (1 - self.feedback_proportion) * chi2_loss_replay
        else:
            chi2_loss = 1 / (4 * self.chi2_coeff) * (reward**2).mean()

        self.optim["critic"].zero_grad(set_to_none=True)
        (q_loss + chi2_loss).backward()
        self.optim["critic"].step()

        # Next, call the actor function to update the actor
        # Before we do that, we rebalance the batches if feedback_proportion is not None.
        if self.feedback_proportion is not None and B_r > 0:
            num_fb_data_points = int(B_r * self.feedback_proportion / (1 - self.feedback_proportion))
            num_fb_data_points = min(num_fb_data_points, 2 * B_fb * S_fb)
            fb_idx = torch.randperm(2 * B_fb * S_fb, device=self.device)[:num_fb_data_points]
            replay_idx = torch.arange(2 * B_fb * S_fb, B_total, device=self.device)
            idx = torch.cat((fb_idx, replay_idx), dim=0)
            # Sub-sample to maintain balance when updating the actor.
            obs, action, qs = obs[idx], action[idx], qs[:, idx]  # Note the ensemble dim.

        metrics = self._update_actor(obs, action, qs)  # Cache the computation of q
        all_metrics.update(metrics)

        self._data_pts_seen += B_total
        self._flops += B_total + obs.shape[0]  # one grad pass for critic, one for actor

        # Add the IP-Learn metrics.
        all_metrics.update(
            dict(
                q_loss=q_loss.item(),
                q=qs.mean().item(),
                chi2_loss=chi2_loss.item(),
                reward=reward.mean().item(),
                data_pts_seen=self._data_pts_seen,
                flops=self._flops,
            )
        )

        # Finally, update the target network.
        if step % self.target_freq == 0:
            with torch.no_grad():
                for param, target_param in zip(self.network.parameters(), self.target_network.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return all_metrics

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


class IPLearnSAC(IPLearnBase):
    """
    This is the general IP-Learn algorithm for online learning
    The main differences between this and the offline implementation are that this implementation
    uses SAC for policy optimization
    """

    def __init__(
        self,
        *args,
        target_freq: int = 2,  # NOTE: Diff default target freq for online and offline.
        init_temperature: float = 0.1,
        bc_coeff=0.0,
        target_entropy: Optional[float] = None,
        # all of the other kwargs for reward
        use_soft_target: bool = True,
        **kwargs,
    ):
        # Save values needed for optim setup.
        self.init_temperature = init_temperature
        super().__init__(*args, target_freq=target_freq, **kwargs)

        # SAC parameters
        self.bc_coeff = bc_coeff
        self.target_entropy = (
            -np.prod(self.processor.action_space.low.shape) if target_entropy is None else -target_entropy
        )
        self.use_soft_target = use_soft_target

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def setup_network(self, network_class: Type[torch.nn.Module], network_kwargs: Dict) -> None:
        super().setup_network(network_class, network_kwargs)
        log_alpha = torch.tensor(np.log(self.init_temperature), dtype=torch.float).to(self.device)
        self.log_alpha = torch.nn.Parameter(log_alpha, requires_grad=True)

    def setup_optimizers(self) -> None:
        super().setup_optimizers()
        # Default optimizer initialization
        keys = ("actor", "critic", "log_alpha")
        default_kwargs = {}
        for key, value in self.optim_kwargs.items():
            if key not in keys:
                default_kwargs[key] = value
            else:
                assert isinstance(value, dict), "Special keys must be kwarg dicts"

        log_alpha_kwargs = default_kwargs.copy()
        log_alpha_kwargs.update(self.optim_kwargs.get("log_alpha", dict()))
        self.optim["log_alpha"] = self.optim_class([self.log_alpha], **log_alpha_kwargs)

    def _compute_next_V(self, next_obs):
        with torch.no_grad():
            dist = self.network.actor(next_obs)
            next_action = dist.sample()
            next_vs = self.target_network.critic(next_obs, next_action)

            if self.use_min_target:
                next_vs = next_vs.min(dim=0, keepdim=True)[0]
            else:
                next_vs = next_vs.mean(dim=0, keepdim=True)

            if self.use_soft_target:
                log_prob = dist.log_prob(next_action).sum(dim=-1)
                next_vs = next_vs - self.alpha.detach() * log_prob

        return next_vs  # Shape (1, B)

    def _update_actor(self, obs, action, qs):
        dist = self.network.actor(obs)
        action_pi = dist.rsample()
        log_prob = dist.log_prob(action_pi).sum(dim=-1)
        qs_pi = self.network.critic(obs, action_pi)
        q_pi = torch.mean(qs_pi, dim=0)[0]  # Note: changed to mean from min.
        actor_loss = (self.alpha.detach() * log_prob - q_pi).mean()
        if self.bc_coeff > 0.0:
            bc_loss = -dist.log_prob(action).sum(dim=-1).mean()  # Simple NLL loss.
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
            alpha_loss=alpha_loss.item(),
            alpha=self.alpha.item(),
            entropy=entropy.item(),
        )


class IPLearnAWAC(IPLearnBase):
    def __init__(
        self,
        *args,
        target_freq: int = 1,  # NOTE: different init target freq for offline and online.
        beta: float = 0.33,
        clip_score: float = 100.0,
        max_feedback: int = 0,  # Default to fully offline
        **kwargs,
    ):
        super().__init__(*args, target_freq=target_freq, max_feedback=max_feedback, **kwargs)
        self.beta = beta
        self.clip_score = clip_score

    def _compute_next_V(self, next_obs):
        with torch.no_grad():
            dist = self.network.actor(next_obs)
            next_action = dist.sample() if isinstance(dist, torch.distributions.Distribution) else dist
            next_vs = self.target_network.critic(next_obs, next_action)
            if self.use_min_target:
                next_vs = next_vs.min(dim=0, keepdim=True)[0]
            else:
                next_vs = next_vs.mean(dim=0, keepdim=True)

        return next_vs  # Shape (1, B)

    def _update_actor(self, obs, action, qs):
        dist = self.network.actor(obs)

        # We need to compute the advantage, which is equal to Q(s,a) - V(s) = Q(s,a) - Q(s,pi(s))
        # Sometimes this is also computed using the target network as target_q - Q(s, pi(s))
        # but in our case that doesn't make sense since we don't have the reward!
        with torch.no_grad():
            action_pi = dist.sample() if isinstance(dist, torch.distributions.Distribution) else dist
            adv = qs.mean(dim=0) - self.network.critic(obs, action_pi).mean(dim=0)
            exp_adv = torch.exp(adv / self.beta)
            if self.clip_score is not None:
                exp_adv = torch.clamp(exp_adv, max=self.clip_score)

        if isinstance(dist, torch.distributions.Distribution):
            bc_loss = -dist.log_prob(action).sum(dim=-1)
        elif torch.is_tensor(dist):
            assert dist.shape == action.shape
            bc_loss = torch.nn.functional.mse_loss(dist, action, reduction="none").sum(dim=-1)
        else:
            raise ValueError("Invalid policy output provided")
        actor_loss = (exp_adv * bc_loss).mean()

        self.optim["actor"].zero_grad(set_to_none=True)
        actor_loss.backward()
        self.optim["actor"].step()

        return dict(
            actor_loss=actor_loss.item(),
            adv=adv.mean().item(),
        )
