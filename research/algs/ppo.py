from re import L
import torch
import numpy as np
import itertools
import gym
import collections

from .base import Algorithm
from research.utils import utils
from research.processors.normalization import RunningMeanStd, RunningObservationNormalizer
from research.networks.base import ActorCriticPolicy
from research.datasets import RolloutBuffer

class PPO(Algorithm):
    
    def __init__(self, *args, 
            num_epochs=10,
            clip_range=0.2,
            clip_range_vf=None,
            ent_coeff=0.0,
            vf_coeff=0.5,
            max_grad_norm=0.5,
            normalize_advantage=True,
            normalize_returns=False,
            reward_clip=None,            
            **kwargs
            ):
        super().__init__(*args, **kwargs)
        # Perform initial type checks
        assert isinstance(self.network, ActorCriticPolicy)

        # Store algorithm values
        self.num_epochs = num_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.ent_coeff = ent_coeff
        self.vf_coeff = vf_coeff
        self.max_grad_norm = max_grad_norm
        self.normalize_advantage = normalize_advantage

        # Return normalization
        self.normalize_returns = normalize_returns
        self.reward_clip = reward_clip
        if self.normalize_returns:
            self.return_rms = RunningMeanStd(shape=(), dtype=np.float64)

        # Losses
        self.value_criterion = torch.nn.MSELoss()

    def _collect_rollouts(self):
        # Setup the dataset and network
        self.dataset.setup()
        self.eval_mode()

        # Setup metrics
        metrics = dict(reward=[], length=[], success=[])
        ep_reward, ep_length, ep_return, ep_success = 0, 0, 0, False

        obs = self.env.reset()
        if isinstance(self.processor, RunningObservationNormalizer):
            self.processor.update(obs)
        self.dataset.add(obs=obs) # Add the first observation
        while not self.dataset.is_full:
            
            with torch.no_grad():
                batch = self._format_batch(utils.unsqueeze(obs, 0)) # Preprocess obs
                latent = self.network.encoder(batch)
                dist = self.network.actor(latent)
                # Collect relevant information
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)
                value = self.network.critic(latent)
                # Unprocess back to numpy
                action = utils.to_np(utils.get_from_batch(action, 0))
                log_prob = utils.to_np(utils.get_from_batch(log_prob, 0))
                value = utils.to_np(utils.get_from_batch(value, 0))
                extras = self._compute_extras(dist)
            
            if isinstance(self.env.action_space, gym.spaces.Box): # Clip the actions
                clipped_action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
            
            obs, reward, done, info = self.env.step(clipped_action)
            if isinstance(self.processor, RunningObservationNormalizer):
                self.processor.update(obs)
            
            ep_reward += reward
            ep_length += 1
            ep_return = self.dataset.discount*ep_return + reward
            if self.normalize_returns:
                self.return_rms.update(ep_return)
                reward = reward / self.return_rms.std
                if self.reward_clip is not None:
                    reward = np.clip(reward, -self.reward_clip, self.reward_clip)

            if ('success' in info and info['success']) or ('is_success' in info and info['is_success']):
                ep_success = True
            
            if done:
                # Update metrics
                metrics['reward'].append(ep_reward)
                metrics['length'].append(ep_length)
                metrics['success'].append(ep_success)
                ep_reward, ep_length, ep_return, ep_success = 0, 0, 0, False
                # If its done, we need to update the observation as well as the terminal reward
                with torch.no_grad():
                    batch = self._format_batch(utils.unsqueeze(obs, 0)) # Preprocess obs
                    terminal_value = self.network.critic(self.network.encoder(batch))
                    terminal_value = utils.to_np(utils.get_from_batch(terminal_value, 0))
                reward += self.dataset.discount * terminal_value
                obs = self.env.reset()

            # Note: Everything is from the last observation except for the observation, which is really next_obs
            self.dataset.add(obs=obs, action=action, reward=reward, done=done, value=value, log_prob=log_prob, **extras)
            
            self._env_steps += 1

        self.train_mode()
        metrics['env_steps'] = self._env_steps # Log environment steps because it isn't proportional to the number of batches.
        metrics['reward_std'] = np.std(metrics['reward'])
        return metrics

    def _compute_extras(self, dist):
        # Used for computing extras values for different versions of PPO
        return {}

    def _setup_train(self):
        # Checks
        assert isinstance(self.dataset, RolloutBuffer)
        # Logging metrics
        self._env_steps = 0
        self._collect_rollouts()
    
    def _train_step(self, batch):
        metrics = dict(env_steps=self._env_steps)
        if self.dataset.last_batch and self.epochs % self.num_epochs == 0:
            # On the last batch of the epoch recollect data.
            metrics.update(self._collect_rollouts())
        
        # Run the policy to predict the values, log probs, and entropies
        latent = self.network.encoder(batch["obs"])
        dist = self.network.actor(latent)
        log_prob = dist.log_prob(batch["action"]).sum(dim=-1)
        value = self.network.critic(latent)

        advantage = batch["advantage"]
        if self.normalize_advantage:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        
        ratio = torch.exp(log_prob - batch["log_prob"])
        policy_loss_1 = advantage * ratio
        policy_loss_2 = advantage * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
        policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

        if self.clip_range_vf is not None:
            value = batch["value"] + torch.clamp(value - batch["value"], -self.clip_range_vf, self.clip_range_vf)
        value_loss = self.value_criterion(batch["returns"], value)

        entropy_loss = -torch.mean(dist.entropy().sum(dim=-1))

        total_loss = policy_loss + self.vf_coeff * value_loss  + self.ent_coeff * entropy_loss

        self.optim["network"].zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optim["network"].step()

        # Update all of the metrics at the end to not break computation flow
        metrics["policy_loss"] = policy_loss.item()
        metrics["value_loss"] = value_loss.item()
        metrics["entropy_loss"] = entropy_loss.item()
        metrics["loss"] = total_loss.item()
        return metrics

    def _validation_step(self, batch):
        raise NotImplementedError("RL Algorithm does not have a validation dataset.")

class AdaptiveKLPPO(Algorithm):
    
    def __init__(self, *args, 
            target_kl=0.025,
            kl_window=None,
            **kwargs
            ):
        super().__init__(*args, **kwargs)
        assert self.clip_range is None, "Clip range is not used in Adaptive KL based PPO"
        self.target_kl = target_kl
        self.kl_window = kl_window
        self.beta = 1

    def _compute_extras(self, dist):
        mu = utils.to_np(utils.get_from_batch(dist.loc, 0))
        sigma = utils.to_np(utils.get_from_batch(dist.scale, 0))
        return dict(mu=mu, sigma=sigma)

    def _setup_train(self):
        # Logging metrics
        self._env_steps = 0
        self._collect_rollouts()
        self._kl_divs = collections.deque(maxlen=self.kl_window)
    
    def _train_step(self, batch):
        metrics = dict(env_steps=self._env_steps)
        if self.dataset.last_batch and self.epochs % self.num_epochs == 0:
            # On the last batch of the epoch recollect data.
            metrics.update(self._collect_rollouts())
            # set flag for updating KL divergence
            update_kl_beta = True
        else:
            update_kl_beta = False
        
        # Run the policy to predict the values, log probs, and entropies
        latent = self.network.encoder(batch["obs"])
        dist = self.network.actor(latent)
        log_prob = dist.log_prob(batch["action"]).sum(dim=-1)
        value = self.network.critic(latent)

        advantage = batch["advantage"]
        if self.normalize_advantage:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        ratio = torch.exp(log_prob - batch["log_prob"])
        policy_loss = -(advantage * ratio).mean()

        # Compute the KL divergence here.
        # Note that this could be done more numerically stable by using log_sigma instead of sigma
        old_dist = torch.distributions.Normal(batch["mu"], batch["sigma"])
        kl_div = torch.distributions.kl.kl_divergence(old_dist, dist).sum(dim=-1).mean()
        
        if self.clip_range_vf is not None:
            value = batch["value"] + torch.clamp(value - batch["value"], -self.clip_range_vf, self.clip_range_vf)
        value_loss = self.value_criterion(batch["returns"], value)

        entropy_loss = -torch.mean(dist.entropy().sum(dim=-1))

        total_loss = policy_loss + self.beta * kl_div + self.vf_coeff * value_loss  + self.ent_coeff * entropy_loss

        self.optim["network"].zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optim["network"].step()

        # Update KL Divergences
        self._kl_divs.append(kl_div.detach().cpu().numpy())
        if update_kl_beta:
            avg_kl = np.mean(self._kl_divs)
            if avg_kl < self.target_kl / 1.5:
                self.beta = self.beta / 2
            elif avg_kl > self.target_kl * 1.5:
                self.beta = self.beta * 2
            # Empty the KL buffer
            self._kl_divs = collections.deque(maxlen=self.kl_window)

        # Update all of the metrics at the end to not break computation flow
        metrics["policy_loss"] = policy_loss.item()
        metrics["kl_divergence"] = kl_div.item()
        metrics["value_loss"] = value_loss.item()
        metrics["entropy_loss"] = entropy_loss.item()
        metrics["loss"] = total_loss.item()
        metrics["beta"] = self.beta
        return metrics

    def _validation_step(self, batch):
        raise NotImplementedError("RL Algorithm does not have a validation dataset.")
