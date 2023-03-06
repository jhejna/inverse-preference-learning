import itertools
from typing import Any, Dict, Optional, Type

import gym
import imageio
import numpy as np
import torch

from research.datasets.feedback_buffer import PairwiseComparisonDataset
from research.datasets.replay_buffer import ReplayBuffer
from research.networks.base import ActorCriticPolicy

from .base import Algorithm


class IHLearnSACOffline(Algorithm):
    def __init__(
        self,
        *args,
        tau: float = 0.005,
        init_temperature: float = 0.1,
        target_freq: int = 2,
        bc_coeff=0.0,
        learn_temperature: bool = True,
        target_entropy: Optional[float] = None,
        chi2_coeff: float = 0.5,
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
        self.reward_criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
        self.chi2_coeff = chi2_coeff

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

    def train_step(self, batch: Dict, step: int, total_steps: int) -> Dict:
        obs = torch.cat([batch["obs_1"], batch["obs_2"]], dim=0)  # (B, S+1)
        action = torch.cat([batch["action_1"], batch["action_2"]], dim=0)  # (B, S+1)
        obs = self.network.encoder(obs)  # Encode all the observations.

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
            dist = self.network.actor(next_obs.detach())
            next_action = dist.sample()
            next_vs = self.target_network.critic(next_obs, next_action).min(dim=0)[0]
        reward = qs - batch["discount"] * next_vs
        # view reward again in the correct shape
        E, B_times_S = reward.shape
        assert B_times_S == B * S, "Shapes were incorrect"
        reward = reward.view(E, B, S)
        r1, r2 = torch.chunk(reward, 2, dim=1)  # Should return two (E, B, S)
        logits = r2.sum(dim=2) - r1.sum(dim=2)  # Sum across sequence dim, (E, B)

        labels = batch["label"].float().unsqueeze(0).expand(logits.shape[0], -1)  # Shape (E, B)
        assert labels.shape == logits.shape
        q_loss = self.reward_criterion(logits, labels).mean()

        chi2_loss = (
            1 / (8 * self.chi2_coeff) * ((r1**2).mean() + (r2**2).mean())
        )  # Turn 1/4 to 1/8 because we sum over both.

        self.optim["critic"].zero_grad(set_to_none=True)
        (q_loss + chi2_loss).backward()
        self.optim["critic"].step()

        # Update Actor SAC-style.
        dist = self.network.actor(obs.detach())  # Encoder is updated with critic.
        policy_action = dist.rsample()
        log_prob = dist.log_prob(policy_action).sum(dim=-1)
        qs = self.network.critic(obs.detach(), policy_action)
        q = torch.min(qs, dim=0)[0]
        actor_loss = (self.alpha.detach() * log_prob - q).mean()
        if self.bc_coeff > 0.0:
            bc_loss = -dist.log_prob(action).sum(dim=-1).mean()  # Simple NLL loss.
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

        if step % self.target_freq == 0:
            # Only update the critic and encoder for speed. Ignore the actor.
            with torch.no_grad():
                for param, target_param in zip(self.network.parameters(), self.target_network.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return dict(
            q_loss=q_loss.item(),
            chi2_loss=chi2_loss.item(),
            actor_loss=actor_loss.item(),
            entropy=entropy.item(),
            alpha=self.alpha.detach().item(),
            adv=r1.mean().item(),
        )

    def _predict(self, batch: Any, sample: bool = False) -> torch.Tensor:
        with torch.no_grad():
            z = self.network.encoder(batch["obs"])
            dist = self.network.actor(z)
            action = dist.sample() if sample else dist.loc
            action = action.clamp(*self.action_range)
            return action


class IHLearnAWAC(Algorithm):
    def __init__(
        self,
        *args,
        tau: float = 0.005,
        target_freq: int = 1,
        beta: float = 1,
        clip_score: float = 100.0,
        chi2_coeff: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        assert isinstance(self.network, ActorCriticPolicy)
        self.tau = tau
        self.target_freq = target_freq
        self.beta = beta
        self.clip_score = clip_score
        self.action_range = [
            float(self.processor.action_space.low.min()),
            float(self.processor.action_space.high.max()),
        ]
        self.chi2_coeff = chi2_coeff
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

    def train_step(self, batch: Dict, step: int, total_steps: int) -> Dict:
        obs = torch.cat([batch["obs_1"], batch["obs_2"]], dim=0)  # (B, S+1)
        action = torch.cat([batch["action_1"], batch["action_2"]], dim=0)  # (B, S+1)
        obs = self.network.encoder(obs)  # Encode all the observations.

        # Compute shapes and add everything to the batch dimension
        B, S = obs.shape[:2]
        S -= 1  # Subtract one for the next obs sequence.
        flat_obs_shape = (B * S,) + obs.shape[2:]
        flat_action_shape = (B * S,) + action.shape[2:]
        next_obs = obs[:, 1:].reshape(flat_obs_shape)
        obs = obs[:, :-1].reshape(flat_obs_shape)
        action = action[:, :-1].reshape(flat_action_shape)

        qs = self.network.critic(obs.detach(), action)
        with torch.no_grad():
            dist = self.network.actor(next_obs.detach())
            next_action = dist.sample() if isinstance(dist, torch.distributions.Distribution) else dist
            next_vs = self.target_network.critic(next_obs, next_action).min(dim=0)[0]
        reward = qs - batch["discount"] * next_vs
        # view reward again in the correct shape
        E, B_times_S = reward.shape
        assert B_times_S == B * S, "Shapes were incorrect"
        reward = reward.view(E, B, S)
        r1, r2 = torch.chunk(reward, 2, dim=1)  # Should return two (E, B, S)
        logits = r2.sum(dim=2) - r1.sum(dim=2)  # Sum across sequence dim, (E, B)

        labels = batch["label"].float().unsqueeze(0).expand(logits.shape[0], -1)  # Shape (E, B)
        assert labels.shape == logits.shape
        q_loss = self.reward_criterion(logits, labels).mean()

        chi2_loss = (
            1 / (8 * self.chi2_coeff) * ((r1**2).mean() + (r2**2).mean())
        )  # Turn 1/4 to 1/8 because we sum over both.

        self.optim["critic"].zero_grad(set_to_none=True)
        (q_loss + chi2_loss).backward()
        self.optim["critic"].step()

        dist = self.network.actor(obs)  # Use encoder gradients for the actor.

        # We need to compute the advantage, which is equal to Q(s,a) - V(s) = Q(s,a) - Q(s,pi(s))
        with torch.no_grad():
            policy_action = dist.sample() if isinstance(dist, torch.distributions.Distribution) else dist
            adv = qs.mean(dim=0) - self.network.critic(obs, policy_action).mean(dim=0)
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

        return dict(
            q_loss=q_loss.item(),
            chi2_loss=chi2_loss.item(),
            actor_loss=actor_loss.item(),
            q=qs.mean().item(),
            adv=adv.mean().item(),
            reward=r1.mean().item(),
        )

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
