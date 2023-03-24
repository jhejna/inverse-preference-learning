import itertools
from typing import Dict, Optional, Type

import numpy as np
import torch

from research.networks.base import ActorCriticValueRewardPolicy

from .off_policy_algorithm import OffPolicyAlgorithm


def iql_loss(pred, target, expectile=0.5):
    err = target - pred
    weight = torch.abs(expectile - (err < 0).float())
    return weight * torch.square(err)


class PreferenceIQL(OffPolicyAlgorithm):
    def __init__(
        self,
        *args,
        tau: float = 0.005,
        target_freq: int = 1,
        expectile: Optional[float] = None,
        beta: float = 1,
        clip_score: float = 100.0,
        reward_steps: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        assert isinstance(self.network, ActorCriticValueRewardPolicy)
        self.tau = tau
        self.target_freq = target_freq
        self.expectile = expectile
        self.beta = beta
        self.clip_score = clip_score
        self.reward_steps = reward_steps
        self.action_range = [
            float(self.processor.action_space.low.min()),
            float(self.processor.action_space.high.max()),
        ]
        self.reward_criterion = torch.nn.BCEWithLogitsLoss(reduction="none")

    def setup_network(self, network_class: Type[torch.nn.Module], network_kwargs: Dict) -> None:
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
        keys = ("actor", "critic", "value", "reward")
        default_kwargs = {}
        for key, value in self.optim_kwargs.items():
            if key not in keys:
                default_kwargs[key] = value
            else:
                assert isinstance(value, dict), "Special keys must be kwarg dicts"

        actor_kwargs = default_kwargs.copy()
        actor_kwargs.update(self.optim_kwargs.get("actor", dict()))
        actor_params = itertools.chain(self.network.actor.parameters(), self.network.encoder.parameters())
        self.optim["actor"] = self.optim_class(actor_params, **actor_kwargs)

        # Update the encoder with the critic.
        critic_kwargs = default_kwargs.copy()
        critic_kwargs.update(self.optim_kwargs.get("critic", dict()))
        self.optim["critic"] = self.optim_class(self.network.critic.parameters(), **critic_kwargs)

        value_kwargs = default_kwargs.copy()
        value_kwargs.update(self.optim_kwargs.get("value", dict()))
        self.optim["value"] = self.optim_class(self.network.value.parameters(), **value_kwargs)

        reward_kwargs = default_kwargs.copy()
        reward_kwargs.update(self.optim_kwargs.get("reward", dict()))
        self.optim["reward"] = self.optim_class(self.network.reward.parameters(), **reward_kwargs)

    def _update_reward(self, batch):
        obs = torch.cat([batch["obs_1"], batch["obs_2"]], dim=0)  # (B, S+1)
        action = torch.cat([batch["action_1"], batch["action_2"]], dim=0)  # (B, S+1)
        obs = self.network.encoder(obs)  # Encode all the observations.

        # Compute shapes and add everything to the batch dimension
        B, S = obs.shape[:2]
        flat_obs_shape = (B * S,) + obs.shape[2:]
        flat_action_shape = (B * S,) + action.shape[2:]
        obs = obs.view(flat_obs_shape)
        action = action.view(flat_action_shape)

        reward = self.network.reward(obs, action)

        # First update the reward net.
        E, B_times_S = reward.shape
        assert B_times_S == B * S
        r1, r2 = torch.chunk(reward.view(E, B, S), 2, dim=1)  # Should return two (E, B, S)
        logits = r2.sum(dim=2) - r1.sum(dim=2)  # Sum across sequence dim, (E, B)
        labels = batch["label"].float().unsqueeze(0).expand(E, -1)  # Shape (E, B)
        assert labels.shape == logits.shape
        reward_loss = self.reward_criterion(logits, labels).mean()

        self.optim["reward"].zero_grad(set_to_none=True)
        reward_loss.backward()
        self.optim["reward"].step()

        return dict(reward_loss=reward_loss.item(), reward=reward.mean().item())

    def _update_iql(self, batch):
        # Run the encoders
        obs = self.network.encoder(batch["obs"])
        with torch.no_grad():
            next_obs = self.network.encoder(batch["next_obs"])
        action = batch["action"]

        # compute the value loss
        with torch.no_grad():
            target_q = self.target_network.critic(obs, action)
            target_q = torch.min(target_q, dim=0)[0]
        vs = self.network.value(obs)
        v_loss = iql_loss(vs, target_q.expand(vs.shape[0], -1), self.expectile).mean()

        self.optim["value"].zero_grad(set_to_none=True)
        v_loss.backward()
        self.optim["value"].step()

        # Next, update the actor. We detach and use the old value, v for computational efficiency
        # and use the target_q value though the JAX IQL recomputes both
        # Pytorch IQL versions have not.
        with torch.no_grad():
            adv = target_q - torch.mean(vs, dim=0)[0]  # min trick is not used on value.
            exp_adv = torch.exp(adv / self.beta)
            if self.clip_score is not None:
                exp_adv = torch.clamp(exp_adv, max=self.clip_score)

        dist = self.network.actor(obs)  # Use encoder gradients for the actor.
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

        # Next, Finally update the critic
        with torch.no_grad():
            next_vs = self.network.value(next_obs)
            next_v = torch.mean(next_vs, dim=0, keepdim=True)
            reward = self.network.reward(obs, action)
            target = reward + batch["discount"] * next_v  # use the predicted reward.
        qs = self.network.critic(obs, action)
        q_loss = torch.nn.functional.mse_loss(qs, target.expand(qs.shape[0], -1), reduction="none").mean()

        self.optim["critic"].zero_grad(set_to_none=True)
        q_loss.backward()
        self.optim["critic"].step()

        return dict(
            q_loss=q_loss.item(),
            v_loss=v_loss.item(),
            actor_loss=actor_loss.item(),
            v=vs.mean().item(),
            q=qs.mean().item(),
            adv=adv.mean().item(),
            reward=reward.mean().item(),
        )

    def setup(self):
        super().setup()
        self._data_pts_seen = 0
        self._flops = 0

    def train_step(self, batch: Dict, step: int, total_steps: int) -> Dict:
        all_metrics = {}

        replay_batch, feedback_batch = batch

        if step < self.reward_steps:
            # Update the reward network
            metrics = self._update_reward(feedback_batch)
            all_metrics.update(metrics)
            self._data_pts_seen += 2 * feedback_batch["obs_1"].shape[0]
            self._flops += 2 * feedback_batch["obs_1"].shape[0]

        # Determine which data to use for training
        if replay_batch is None or "obs" not in replay_batch:
            # Construct an IQL training batch from the replay batch
            obs = torch.cat([feedback_batch["obs_1"], feedback_batch["obs_2"]], dim=0)  # (B, S+1)
            action = torch.cat([feedback_batch["action_1"], feedback_batch["action_2"]], dim=0)  # (B, S+1)
            discount = feedback_batch["discount"].repeat(2)

            # Compute shapes and add everything to the batch dimension
            B, S = obs.shape[:2]
            S -= 1  # Subtract one for the next obs sequence.
            flat_obs_shape = (B * S,) + obs.shape[2:]
            flat_action_shape = (B * S,) + action.shape[2:]
            next_obs = obs[:, 1:].reshape(flat_obs_shape)
            obs = obs[:, :-1].reshape(flat_obs_shape)
            action = action[:, :-1].reshape(flat_action_shape)
            discount = discount.unsqueeze(1).repeat(1, S).flatten()
            replay_batch = dict(obs=obs, action=action, next_obs=next_obs, discount=discount)
        else:
            self._data_pts_seen += replay_batch["obs"].shape[0]

        metrics = self._update_iql(replay_batch)
        all_metrics.update(metrics)
        self._flops += 3 * replay_batch["obs"].shape[0]

        all_metrics["data_pts_seen"] = self._data_pts_seen
        all_metrics["flops"] = self._flops

        # Update the networks. These are done in a stack to support different grad options for the encoder.
        if step % self.target_freq == 0:
            with torch.no_grad():
                # Only run on the critic and encoder, those are the only weights we update.
                for param, target_param in zip(
                    self.network.critic.parameters(), self.target_network.critic.parameters()
                ):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                for param, target_param in zip(
                    self.network.encoder.parameters(), self.target_network.encoder.parameters()
                ):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return all_metrics

    def _predict(self, batch: Dict, sample: bool = False) -> torch.Tensor:
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
