from typing import Any, Optional

import d4rl
import gym
import numpy as np

from .replay_buffer import HindsightReplayBuffer, ReplayBuffer


class D4RLDataset(ReplayBuffer):
    """
    This class is designed to be able to produce the same dataset configs used in the IQL paper.
    See https://github.com/ikostrikov/implicit_q_learning
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        name: str,
        d4rl_path: Optional[str] = None,  # where to save D4RL files.
        use_rtg: bool = False,
        use_timesteps: bool = False,
        normalize_reward: bool = False,
        reward_scale: float = 1.0,
        reward_shift: float = 0.0,
        action_eps: float = 0.00001,
        **kwargs,
    ) -> None:
        self.env_name = name
        self.reward_scale = reward_scale
        self.reward_shift = reward_shift
        self.normalize_reward = normalize_reward
        self.action_eps = action_eps
        self.use_rtg = use_rtg
        self.use_timesteps = use_timesteps
        if d4rl_path is not None:
            d4rl.set_dataset_path(d4rl_path)
        super().__init__(observation_space, action_space, **kwargs)

    def _data_generator(self):
        env = gym.make(self.env_name)
        dataset = env.get_dataset()

        def get_done(i, ep_step=None):
            nonlocal dataset, env
            done = False
            if "ant" not in self.env_name:
                # use terminals
                done = done or dataset["terminals"][i]
            if "timeouts" in dataset:
                done = done or dataset["timeouts"][i]
            elif ep_step is not None:
                done = done or (episode_step == env._max_episode_steps - 1)
            return done

        # Compute dataset normalization as in https://github.com/ikostrikov/implicit_q_learning
        if self.normalize_reward:
            ep_rewards = []
            ep_reward, ep_length = 0, 0
            for i in range(len(dataset["observations"])):
                ep_reward += dataset["rewards"][i]
                done = get_done(i, ep_length)
                ep_length += 1
                if done:
                    ep_rewards.append(ep_reward)
                    ep_reward, ep_length = 0, 0
            min_reward, max_reward = min(ep_rewards), max(ep_rewards)
            print("[research] Normalized D4RL range:", min_reward, max_reward)
            self.reward_scale *= env._max_episode_steps / (max_reward - min_reward)

        # Lots of this code was borrowed from https://github.com/rail-berkeley/d4rl/blob/master/d4rl/__init__.py
        obs_ = []
        action_ = [self.dummy_action]
        reward_ = [0.0]
        done_ = [False]
        discount_ = [1.0]

        episode_step = 0
        for i in range(dataset["rewards"].shape[0]):
            obs = dataset["observations"][i].astype(np.float32)
            action = dataset["actions"][i].astype(np.float32)
            reward = dataset["rewards"][i].astype(np.float32)
            terminal = bool(dataset["terminals"][i])
            done = get_done(i, episode_step)

            obs_.append(obs)
            action_.append(action)
            reward_.append(reward)
            discount_.append(1 - float(terminal))
            done_.append(done)

            episode_step += 1

            if done:
                if "next_observations" in dataset:
                    obs_.append(dataset["next_observations"][i].astype(np.float32))
                else:
                    # We need to do somethign to pad to the full length.
                    # The default solution is to get rid of this transtion
                    # but we need a transition with the terminal flag for our replay buffer
                    # implementation to work.
                    # Since we always end up masking this out anyways, it shouldn't matter and we can just repeat
                    obs_.append(dataset["observations"][i].astype(np.float32))

                obs_ = np.array(obs_)
                action_ = np.array(action_)
                if self.action_eps > 0.0:
                    action_ = np.clip(action_, -1.0 + self.action_eps, 1.0 - self.action_eps)
                reward_ = np.array(reward_).astype(np.float32) * self.reward_scale + self.reward_shift
                discount_ = np.array(discount_).astype(np.float32)
                done_ = np.array(done_, dtype=np.bool_)
                kwargs = {}

                # Support Decision Transformer.
                if self.use_rtg:
                    # Compute reward to go
                    rtg = np.zeros_like(reward_, dtype=np.float32)
                    rtg[-1] = reward_[-1]
                    for t in reversed(range(reward_.shape[0] - 1)):
                        rtg[t] = reward_[t] + self.discount * rtg[t + 1]
                    kwargs["rtg"] = rtg

                if self.use_timesteps:
                    kwargs["timestep"] = np.arange(len(reward_), dtype=np.int64)

                yield (obs_, action_, reward_, done_, discount_, kwargs)

                # reset the episode trackers
                episode_step = 0
                obs_ = []
                action_ = [self.dummy_action]
                reward_ = [0.0]
                done_ = [False]
                discount_ = [1.0]

        # Finally clean up the environment
        del dataset
        del env

    def add(
        self,
        obs: Any,
        action: Optional[Any] = None,
        reward: Optional[Any] = None,
        done: Optional[Any] = None,
        discount: Optional[Any] = None,
        **kwargs,
    ) -> None:
        # Make sure to consistently process the environment reward.
        if reward is not None:
            reward = reward * self.reward_scale + self.reward_shift
        return super().add(obs, action, reward, done, discount, **kwargs)
