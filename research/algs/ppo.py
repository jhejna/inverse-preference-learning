import torch
import numpy as np
import itertools
import gym

from .base import Algorithm
from research.utils import utils
from research.networks.base import ActorCriticPolicy

class PPO(Algorithm):
    
    def __init__(self, *args, 
            num_epochs=10,
            clip_range=0.2,
            clip_range_vf=None,
            ent_coeff=0.0,
            vf_coeff=0.5,
            max_grad_norm=0.5,
            normalize_advantage=True,
            **kwargs
            ):
        super().__init__(*args, **kwargs)
        self.num_epochs = num_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range
        self.ent_coeff = ent_coeff
        self.vf_coeff = vf_coeff
        self.max_grad_norm = max_grad_norm
        self.normalize_advantage = normalize_advantage

        # Losses
        self.value_criterion = torch.nn.MSELoss()

    def _collect_rollouts(self):
        # Setup the dataset and network
        self.dataset.setup()
        self.eval_mode()

        # Setup metrics
        metrics = dict(reward=[], length=[], success=[])
        ep_reward, ep_length, ep_success = 0, 0, False

        obs = self.env.reset()
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
            
            if isinstance(self.env.action_space, gym.spaces.Box): # Clip the actions
                clipped_action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
            obs, reward, done, info = self.env.step(clipped_action)
            
            ep_reward += reward
            ep_length += 1
            if ('success' in info and info['success']) or ('is_success' in info and info['is_success']):
                ep_success = True
            
            if done:
                # Update metrics
                metrics['reward'].append(ep_reward)
                metrics['length'].append(ep_length)
                metrics['success'].append(ep_success)
                ep_reward, ep_length, ep_success = 0, 0, False
                # If its done, we need to update the observation as well as the terminal reward
                with torch.no_grad():
                    batch = self._format_batch(utils.unsqueeze(obs, 0)) # Preprocess obs
                    terminal_value = self.network.critic(self.network.encoder(batch))
                    terminal_value = utils.to_np(utils.get_from_batch(terminal_value, 0))
                reward += self.dataset.discount * terminal_value
                obs = self.env.reset()

            # Note: Everything is from the last observation except for the observation, which is really next_obs
            self.dataset.add(obs=obs, action=action, reward=reward, done=done, value=value, log_prob=log_prob)
            
            self._env_steps += 1

        self.train_mode()
        metrics['env_steps'] = self._env_steps # Log environment steps because it isn't proportional to the number of batches.
        return metrics

    def _setup_train(self):
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
