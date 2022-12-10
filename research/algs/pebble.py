import collections
import itertools
import os
import tempfile
from typing import Any, Dict, Optional, Tuple, Type, Union

import gym
import imageio
import numpy as np
import torch

from research.datasets.feedback_buffer import FeedbackLabelDataset
from research.datasets.replay_buffer import ReplayBuffer
from research.networks.base import ActorCriticRewardPolicy
from research.utils.utils import to_device, to_tensor

from .base import Algorithm


class RunningStats(object):
    def __init__(
        self, epsilon: float = 1e-5, shape: Tuple = (), device: Optional[Union[str, torch.device]] = None
    ) -> None:
        self._mean = torch.zeros(shape, device=device)
        self._var = torch.ones(shape, device=device)
        self._count = epsilon

    def update(self, x: torch.Tensor) -> None:
        assert x.shape[1:] == self._mean.shape, "Incorrect shape provided"
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0)
        batch_size = x.shape[0]
        total_count = self._count + batch_size
        delta = batch_mean - self._mean

        # Update the mean
        self._mean = self._mean + delta + batch_size / total_count
        # Update the variance
        self._var = (
            self._var * self._count
            + batch_var * batch_size
            + torch.pow(delta, 2) * self._count * batch_size / total_count
        ) / total_count
        # Update the count
        self._count = total_count

    @property
    def mean(self) -> torch.Tensor:
        return self._mean

    @property
    def std(self) -> torch.Tensor:
        return torch.sqrt(self._var)


class PEBBLE(Algorithm):
    def __init__(
        self,
        env: gym.Env,
        network_class: Type[torch.nn.Module],
        dataset_class: Union[Type[torch.utils.data.IterableDataset], Type[torch.utils.data.Dataset]],
        tau: float = 0.005,
        init_temperature: float = 0.1,
        env_freq: int = 1,
        critic_freq: int = 1,
        actor_freq: int = 1,
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
        reward_optim: Type[torch.optim.Optimizer] = torch.optim.Adam,
        reward_optim_kwargs: Dict = {"lr": 0.0003},
        reset_reward_net: bool = False,
        reward_shift: float = 0,
        reward_scale: float = 1,
        human_feedback: bool = False,
        num_uniform_feedback: int = 0,
        # Unsupervised Parameters
        unsup_steps: int = 0,
        k_nearest_neighbors: int = 5,
        unsup_batch_size: int = 512,
        num_unsup_batches: int = 20,
        normalize_state_entropy: bool = True,
        **kwargs,
    ):
        # Save values needed for optim setup.
        self.init_temperature = init_temperature
        self.reward_optim = reward_optim
        self.reward_optim_kwargs = reward_optim_kwargs
        super().__init__(env, network_class, dataset_class, **kwargs)
        assert isinstance(self.network, ActorCriticRewardPolicy)

        # Save extra parameters
        self.tau = tau
        self.critic_freq = critic_freq
        self.actor_freq = actor_freq
        self.target_freq = target_freq
        self.env_freq = env_freq
        self.reward_freq = reward_freq
        self.init_steps = init_steps
        self.unsup_steps = unsup_steps
        self.reward_init_steps = self.init_steps + self.unsup_steps
        self.action_range = [float(self.action_space.low.min()), float(self.action_space.high.max())]

        # Feedback Parameters
        self.segment_size = segment_size
        self.max_feedback = max_feedback
        self.total_feedback = 0
        self.init_feedback_size = init_feedback_size
        self.feedback_schedule = feedback_schedule
        self.feedback_sample_multiplier = feedback_sample_multiplier
        self._saved_recent_visualizations = True
        self.reward_shift = reward_shift
        self.reward_scale = reward_scale
        self.human_feedback = human_feedback
        self.num_uniform_feedback = num_uniform_feedback

        # Reward Learning Parameters
        self.reward_epochs = reward_epochs
        self.reward_batch_size = reward_batch_size
        self.reward_criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
        self.reset_reward_net = reset_reward_net

        # Checkpointing
        self.checkpoint_path = kwargs["checkpoint"] if "checkpoint" in kwargs else None
        if self.checkpoint_path is None and self.reset_reward_net:
            # Get a temporary file to store the initial reward model checkpoint
            tmp_dir = tempfile.mkdtemp()
            self.checkpoint_path = os.path.join(tmp_dir, "reward.pt")
            self.save(tmp_dir, "reward")

        # Unsupervised Parameters
        self.unsup_batch_size = unsup_batch_size
        self.num_unsup_batches = num_unsup_batches
        self.k_nearest_neighbors = k_nearest_neighbors
        self.normalize_state_entropy = normalize_state_entropy
        self.entropy_stats = RunningStats(shape=(), device=self.device)

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

        self.optim["reward"] = self.reward_optim(self.network.reward.parameters(), **self.reward_optim_kwargs)

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

    def _reset_critic(self) -> None:
        self.network.reset_critic(device=self.device)  # Reset the critic weights
        optim_class = type(self.optim["critic"])
        del self.optim["critic"]  # explicitly remove this from optimization
        critic_params = itertools.chain(self.network.critic.parameters(), self.network.encoder.parameters())
        self.optim["critic"] = optim_class(critic_params, **self.optim_kwargs)
        # Sync the target network
        self.target_network.critic.load_state_dict(self.network.critic.state_dict())
        self.target_network.encoder.load_state_dict(self.network.encoder.state_dict())

    def _update_critic(self, batch: Dict) -> Dict:
        with torch.no_grad():
            dist = self.network.actor(batch["next_obs"])
            next_action = dist.rsample()
            log_prob = dist.log_prob(next_action).sum(dim=-1)
            target_qs = self.target_network.critic(batch["next_obs"], next_action)
            target_v = torch.min(target_qs, dim=0)[0] - self.alpha.detach() * log_prob
            reward = self.network.reward(batch["obs"], batch["action"]).mean(dim=0)  # Should be shape (B, 0)
            reward = self.reward_scale * reward + self.reward_shift
            target_q = reward + batch["discount"] * target_v

        qs = self.network.critic(batch["obs"], batch["action"])
        q_loss = (
            torch.nn.functional.mse_loss(qs, target_q.expand(qs.shape[0], -1)).mean(dim=-1).sum()
        )  # averages over the ensemble. No for loop!

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
        assert not self.dataset.is_parallel, "Unsupervised does not support parallel dataset for now."
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

        # Determine how we update the critic based on whether or not we are doing unsupervised
        if self.steps < self.init_steps + self.unsup_steps:
            critic_update_fn = self._update_critic_unsup
        elif self.steps == self.init_steps + self.unsup_steps:
            self._reset_critic()
            critic_update_fn = self._update_critic
            print("[pebble] Reset the critic.")
        else:
            critic_update_fn = self._update_critic

        updating_critic = self.steps % self.critic_freq == 0
        updating_actor = self.steps % self.actor_freq == 0

        if updating_actor or updating_critic:
            batch["obs"] = self.network.encoder(batch["obs"])
            with torch.no_grad():
                batch["next_obs"] = self.target_network.encoder(batch["next_obs"])

        if updating_critic:
            metrics = critic_update_fn(batch)
            all_metrics.update(metrics)

        if updating_actor:
            metrics = self._update_actor_and_alpha(batch)
            all_metrics.update(metrics)

        if self.steps % self.target_freq == 0:
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

        if (
            (self.last_feedback_step is None or (self.steps - self.last_feedback_step) % self.reward_freq == 0)
            and self.total_feedback < self.max_feedback
            and self.steps >= self.reward_init_steps
        ):
            self.last_feedback_step = self.steps
            # First collect feedback
            metrics = self._collect_feedback()
            all_metrics.update(metrics)
            # Next update the model
            metrics = self._update_reward_model()
            all_metrics.update(metrics)

        return all_metrics

    def _validation_step(self, batch: Dict):
        raise NotImplementedError("RL Algorithm does not have a validation dataset.")

    def _validation_extras(self, path: str, step: int, validation_dataloader) -> Dict:
        return {}  # temporary override on saving visualizations. Need to debug mujoco renderer.
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


class FewShotPEBBLE(PEBBLE):
    """
    Overrides the way we adapt to directly use MAML
    """

    def __init__(self, *args, adapt_lr_mult=1.0, **kwargs):
        self.adapt_lr_mult = adapt_lr_mult
        super().__init__(*args, **kwargs)

    def setup_optimizers(self, optim_class, optim_kwargs):
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

        if self.reward_optim is None:
            self.reward_optim = optim_class
        if self.reward_optim_kwargs is None:
            self.reward_optim_kwargs = optim_kwargs

        self._inner_lrs = torch.nn.ParameterDict(
            {
                k: torch.nn.Parameter(torch.tensor(self.reward_optim_kwargs.get("lr", 0.0003)), requires_grad=False)
                for k, v in self.network.reward.params.items()
            }
        )
        self.optim["reward"] = optim_class(
            itertools.chain(self.network.reward.params.values(), self._inner_lrs.values()), **self.reward_optim_kwargs
        )

    def _save_extras(self):
        return {"log_alpha": self.log_alpha, "lrs": self._inner_lrs.state_dict()}

    def _load_extras(self, checkpoint, strict=True):
        if "log_alpha" in checkpoint:
            self.log_alpha.data = checkpoint["log_alpha"].data
        self._inner_lrs.load_state_dict(checkpoint["lrs"], strict=strict)

    def _update_reward_model(self):
        assert self.reset_reward_net, "Must reset network for PEBBLE with explicit MAML"
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        reward_params = collections.OrderedDict(
            [(k[7:], v) for k, v in checkpoint["network"].items() if k.startswith("reward")]
        )
        self.network.reward.load_state_dict(reward_params)
        self._inner_lrs.load_state_dict(checkpoint["lrs"])

        epochs = 0
        reached_max_epochs = False
        while True:
            losses, accuracies = [], []
            for batch in self.feedback_dataloader:
                batch = to_device(batch, self.device)
                logits = self._get_reward_logits(batch)  # Shape (E, B)
                labels = batch["label"].float().unsqueeze(0).expand(logits.shape[0], -1)  # Shape (E, B)
                loss = self.reward_criterion(logits, labels).mean(dim=-1).sum(dim=0)  # Average on B, sum on E
                # This runs the MAML style adaptation at each iteration.
                grads = torch.autograd.grad(loss, self.network.reward.params.values(), create_graph=False)
                for j, (k, v) in enumerate(self.network.reward.params.items()):
                    self.network.reward.params[k].data = v - self._inner_lrs[k] * grads[j]
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
            elif mean_acc > 0.95:
                break
            elif epochs > 40:
                reached_max_epochs = True
                break

        if not reached_max_epochs:
            return dict(reward_loss=np.mean(losses), reward_accuracy=np.mean(accuracies))

        # if accuracy is still below a certain threshold, then finetune again with adam
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
            elif mean_acc > 0.95:
                break
            elif epochs > 1000:
                break

        return dict(reward_loss=np.mean(losses), reward_accuracy=np.mean(accuracies))
