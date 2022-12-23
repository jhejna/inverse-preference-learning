import itertools
import os
from typing import Any, Dict, Tuple, Type, Union

import gym
import imageio
import numpy as np
import torch

from research.datasets.feedback_buffer import FeedbackLabelDataset
from research.datasets.replay_buffer import ReplayBuffer
from research.networks.base import ActorCriticPolicy
from research.utils.utils import to_device, to_tensor

from .base import Algorithm


class IHLearn(Algorithm):
    def __init__(
        self,
        *args,
        tau: float = 0.005,
        init_temperature: float = 0.1,
        env_freq: int = 1,
        target_freq: int = 2,
        init_steps: int = 1000,
        # all of the other kwargs for reward
        reward_freq: int = 128,
        reward_epochs: Union[int, float] = 10,
        max_feedback: int = 1000,
        init_feedback_size: int = 64,
        feedback_sample_multiplier: float = 10,
        reward_batch_size: int = 256,
        segment_size: int = 25,
        feedback_schedule: str = "constant",
        human_feedback: bool = False,
        num_uniform_feedback: int = 0,
        **kwargs,
    ):
        # Save values needed for optim setup.
        self.init_temperature = init_temperature
        super().__init__(*args, **kwargs)
        assert isinstance(self.network, ActorCriticPolicy)

        # Save extra parameters
        self.tau = tau
        self.target_freq = target_freq
        self.env_freq = env_freq
        self.reward_freq = reward_freq
        self.init_steps = init_steps
        self.action_range = [float(self.action_space.low.min()), float(self.action_space.high.max())]
        self.reward_init_steps = self.init_steps

        # Segment parameters
        self.segment_size = segment_size

        # Feedback Parameters
        self.max_feedback = max_feedback
        self.total_feedback = 0
        self.init_feedback_size = init_feedback_size
        self.feedback_schedule = feedback_schedule
        self.feedback_sample_multiplier = feedback_sample_multiplier
        self._saved_recent_visualizations = True
        self.human_feedback = human_feedback
        self.num_uniform_feedback = num_uniform_feedback

        # Reward Learning Parameters
        self.reward_epochs = reward_epochs
        self.reward_batch_size = reward_batch_size
        self.reward_criterion = torch.nn.BCEWithLogitsLoss(reduction="none")

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def setup_network(self, network_class: Type[torch.nn.Module], network_kwargs: Dict) -> None:
        self.network = network_class(self.env.observation_space, self.env.action_space, **network_kwargs).to(
            self.device
        )
        self.target_network = network_class(self.env.observation_space, self.env.action_space, **network_kwargs).to(
            self.device
        )
        self.target_network.load_state_dict(self.network.state_dict())
        for param in self.target_network.parameters():
            param.requires_grad = False

        # Code currently does not support encoder networks.
        # this can be updated later
        assert isinstance(self.network.encoder, torch.nn.Identity)

    def setup_optimizers(self, optim_class: Type[torch.optim.Optimizer], optim_kwargs: Dict) -> None:
        # save the optim_kwargs for resetting the critic
        self.optim_kwargs = optim_kwargs
        # Default optimizer initialization
        self.optim["actor"] = optim_class(self.network.actor.parameters(), **optim_kwargs)
        # Update the encoder with the critic.
        critic_params = itertools.chain(self.network.critic.parameters(), self.network.encoder.parameters())
        self.optim["critic"] = optim_class(critic_params, **optim_kwargs)

        # Setup the learned entropy coefficients. This has to be done first so its present in the setup_optim call.
        self.log_alpha = torch.tensor(np.log(self.init_temperature), dtype=torch.float).to(self.device)
        self.log_alpha.requires_grad = True
        self.target_entropy = -np.prod(self.env.action_space.low.shape)
        self.optim["log_alpha"] = optim_class([self.log_alpha], **optim_kwargs)

    def setup_datasets(self) -> None:
        super().setup_datasets()
        assert isinstance(self.dataset, ReplayBuffer), "Must use replay buffer for PEBBLE"
        assert self.dataset.distributed == False, "Cannot use distributed replay buffer with PEBBLE"
        # Note that the dataloader for the reward model runs on a single thread!
        self.feedback_dataset = FeedbackLabelDataset(
            self.observation_space,
            self.action_space,
            discount=self.dataset.discount,
            nstep=self.dataset.nstep,
            segment_size=self.segment_size,
            capacity=self.max_feedback + 1,
        )
        self.feedback_dataloader = torch.utils.data.DataLoader(
            self.feedback_dataset, batch_size=None, num_workers=0, pin_memory=(self.device.type == "cuda")
        )
        self.feedback_iterator = iter(self.feedback_dataloader)

    def _get_reward_logits(self, batch: Dict) -> torch.Tensor:
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
            next_action = dist.rsample()
            next_vs = self.target_network.critic(next_obs, next_action)
        reward = qs - self.dataset.discount * next_vs
        # view reward again in the correct shape
        E, B_times_S = reward.shape
        assert B_times_S == B * S, "Shapes were incorrect"
        reward = reward.view(E, B, S).sum(dim=2)
        r1, r2 = torch.chunk(reward, 2, dim=1)
        logits = r2 - r1
        return logits  # Shape (E, B)

    def _oracle_label(self, batch: Dict) -> Tuple[np.ndarray, Dict]:
        label = 1.0 * (batch["reward_1"] < batch["reward_2"])
        return label, {}

    def _human_label(self, batch: Dict) -> Tuple[np.ndarray, Dict]:
        from matplotlib import pyplot as plt

        # Get GT labels for metric computation later
        gt_labels, _ = self._oracle_label(batch)
        labels = []
        batch_size = batch["reward_1"].shape[0]
        print("Rendering images.")
        for i in range(batch_size):
            state_1, state_2 = batch["state_1"][i], batch["state_2"][i]  # Shape (S, D)
            segment_1, segment_2 = self._render_segment(state_1), self._render_segment(state_2)
            # Display the overall plot
            fig, ax = plt.subplots(2, 1, figsize=(12, 4))
            ax[0].imshow(segment_1)
            ax[0].set_ylabel("Segment 1")
            ax[1].imshow(segment_2)
            ax[1].set_ylabel("Segment 2")
            plt.tight_layout()
            plt.show()
            inp = None
            while inp not in ("1", "2", "s", "d"):
                inp = input("Segment 1, 2, skip (s), or done (d): ")
            if inp == "1":
                labels.append(0)
            elif inp == "2":
                labels.append(1)
            elif inp == "s":
                labels.append(-1)
            else:
                if len(labels) == 0:
                    labels.append(-1)  # we skip the query so we have a non-empyt list
                # We are done! A bit hacky but set max_feedback to be zero so we never ask for feedback again
                self.max_feedback = -1
                break
            print("Ground truth was", 2 if batch["reward_1"][i] < batch["reward_2"][i] else 1)
        labels = np.array(labels)
        gt_labels = gt_labels[: labels.shape[0]]
        self._correct_queries += np.sum(gt_labels == labels)
        self._skipped_queries += np.sum(labels == -1)
        return labels, dict(correct_queries=self._correct_queries, skipped_queries=self._skipped_queries)

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

    def _collect_feedback(self) -> Dict:
        # Compute the amount of feedback to collect
        if self.feedback_schedule == "linear":
            batch_size = int(self.init_feedback_size * (self.total_steps - self.steps) / self.total_steps)
        elif self.feedback_schedule == "constant":
            batch_size = self.init_feedback_size
        else:
            raise ValueError("Invalid Feedback Schedule Specified.")
        feedback_left = self.max_feedback - self.total_feedback
        batch_size = min(batch_size, feedback_left)
        assert batch_size > 0, "Called _collect_feedback when we have no more budget left."

        if self.total_feedback == 0 or self.total_feedback < self.num_uniform_feedback:
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
        if self.human_feedback:
            labels, metrics = self._human_label(queries)
        else:
            labels, metrics = self._oracle_label(queries)

        feedback_added = labels.shape[0]
        self.total_feedback += feedback_added
        metrics["feedback"] = self.total_feedback
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

    def _step_env(self) -> Dict:
        # Step the environment and store the transition data.
        metrics = dict()
        if self._env_steps < self.init_steps:
            action = self.env.action_space.sample()
        else:
            self.eval_mode()
            with torch.no_grad():
                action = self.predict(dict(obs=self._current_obs), sample=True)
            self.train_mode()
        action = np.clip(action, self.env.action_space.low, self.env.action_space.high)

        next_obs, reward, done, info = self.env.step(action)
        self._episode_length += 1
        self._episode_reward += reward

        if "discount" in info:
            discount = info["discount"]
        elif hasattr(self.env, "_max_episode_steps") and self._episode_length == self.env._max_episode_steps:
            discount = 1.0
        else:
            discount = 1 - float(done)

        # Store the consequences, adding state if the environment has it.
        add_kwargs = {"state": info["state"]} if "state" in info else {}
        self.dataset.add(next_obs, action, reward, done, discount, **add_kwargs)

        if done:
            self._num_ep += 1
            # update metrics
            metrics["reward"] = self._episode_reward
            metrics["length"] = self._episode_length
            metrics["num_ep"] = self._num_ep

            # Reset the environment
            self._current_obs = self.env.reset()
            self.dataset.add(self._current_obs)  # Add the first timestep
            self._episode_length = 0
            self._episode_reward = 0
        else:
            self._current_obs = next_obs

        self._env_steps += 1
        metrics["env_steps"] = self._env_steps
        return metrics

    def _setup_train(self) -> None:
        self._current_obs = self.env.reset()
        self._episode_reward = 0
        self._episode_length = 0
        self._num_ep = 0
        self._env_steps = 0
        self.dataset.add(self._current_obs)  # Store the initial reset observation!
        self.last_feedback_step = None
        # Extra metrics for human labeling
        self._skipped_queries = 0
        self._correct_queries = 0

    def _train_step(self, batch: Dict) -> Dict:
        all_metrics = {}

        if self.steps % self.env_freq == 0 or self._env_steps < self.init_steps:
            # step the environment with freq env_freq or if we are before learning starts
            metrics = self._step_env()
            all_metrics.update(metrics)
            if self._env_steps <= self.init_steps:
                return all_metrics  # return here.

        if "obs" not in batch:
            return all_metrics

        if (
            (self.last_feedback_step is None or (self.steps - self.last_feedback_step) % self.reward_freq == 0)
            and self.total_feedback < self.max_feedback
            and self.steps >= self.reward_init_steps
        ):
            self.last_feedback_step = self.steps
            # First collect feedback
            metrics = self._collect_feedback()
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
        feedback_batch = to_device(feedback_batch, self.device)

        logits = self._get_reward_logits(feedback_batch)
        labels = feedback_batch["label"].float().unsqueeze(0).expand(logits.shape[0], -1)  # Shape (E, B)
        assert labels.shape == logits.shape
        q_loss = self.reward_criterion(logits, labels).mean(dim=-1).sum(dim=0)  # Average on B, sum on E

        self.optim["critic"].zero_grad(set_to_none=True)
        q_loss.backward()
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
        q = torch.min(qs, dim=0)[0]
        actor_loss = (self.alpha.detach() * log_prob - q).mean()

        self.optim["actor"].zero_grad(set_to_none=True)
        actor_loss.backward()
        self.optim["actor"].step()
        entropy = -log_prob.mean()

        # Update the learned temperature
        self.optim["log_alpha"].zero_grad(set_to_none=True)
        alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()
        alpha_loss.backward()
        self.optim["log_alpha"].step()

        # update the metrics
        all_metrics.update(
            dict(
                q_loss=q_loss.item(),
                actor_loss=actor_loss.item(),
                entropy=entropy.item(),
                alpha_loss=alpha_loss.item(),
                alpha=self.alpha.detach().item(),
            )
        )

        if self.steps % self.target_freq == 0:
            # Only update the critic and encoder for speed. Ignore the actor.
            with torch.no_grad():
                for param, target_param in zip(self.network.parameters(), self.target_network.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return all_metrics

    def _validation_step(self, batch: Dict):
        raise NotImplementedError("RL Algorithm does not have a validation dataset.")

    def _validation_extras(self, path: str, step: int, validation_dataloader) -> Dict:
        assert validation_dataloader is None
        if self._saved_recent_visualizations:
            return {}
        self._saved_recent_visualizations = True

        if self.eval_env is None or not hasattr(self.eval_env, "set_state"):
            return {}  # Return if we don't have a visualization method

        # try to set the state of the environment
        # Save a batch of visualizations from the dataset
        num_visualizations = 8
        queries, labels = self._recent_feedback
        num_visualizations = min(num_visualizations, labels.shape[0])
        # Render the first observation from everything
        for i in range(num_visualizations):
            pos_segment = self._render_segment(queries["state_1"][i])
            neg_segment = self._render_segment(queries["state_2"][i])
            if labels[i] == 1:
                # This means that the second is prefered, so swap them
                pos_segment, neg_segment = neg_segment, pos_segment
            # Save the rendering
            grid = np.concatenate((pos_segment, neg_segment), axis=0)
            out_path = os.path.join(path, "feedback_%d_query_%d.png" % (self.total_feedback, i))
            imageio.imwrite(out_path, grid)
        return {}

    def _render_segment(self, states: np.ndarray, height: int = 128, width: int = 128) -> np.ndarray:
        assert self.eval_env is not None and hasattr(self.eval_env, "set_state")
        imgs = []
        max_imgs = 12
        if len(states) > max_imgs:
            factor = int(round(len(states) / max_imgs))
            states = states[::factor][:max_imgs]
        for state in states:
            self.eval_env.set_state(state)
            img = self.eval_env.render(mode="rgb_array", height=height, width=width)
            imgs.append(img)
        # Concatenate the images on the last axis
        imgs = np.concatenate(imgs, axis=1)
        return imgs

    def _predict(self, batch: Any, sample: bool = False) -> torch.Tensor:
        with torch.no_grad():
            z = self.network.encoder(batch["obs"])
            dist = self.network.actor(z)
            if sample:
                action = dist.sample()
            else:
                action = dist.loc
            action = action.clamp(*self.action_range)
            return action

    def _save_extras(self) -> Dict:
        return {"log_alpha": self.log_alpha}

    def _load_extras(self, checkpoint, strict=True) -> None:
        if "log_alpha" in checkpoint:
            self.log_alpha.data = checkpoint["log_alpha"].data
