import argparse
import os
from typing import List

import gym
import numpy as np

import research
from research.algs.base import Algorithm
from research.datasets import ReplayBuffer
from research.utils.config import Config
from research.utils.trainer import load, load_from_path


def collect_random_episode(env: gym.Env, dataset: ReplayBuffer) -> None:
    obs = env.reset()
    dataset.add(obs)
    episode_length = 0
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        episode_length += 1
        if "discount" in info:
            discount = info["discount"]
        elif hasattr(env, "_max_episode_steps") and episode_length == env._max_episode_steps:
            discount = 1.0
        else:
            discount = 1 - float(done)
        dataset.add(obs, action, reward, done, discount)


def collect_policy_episode(env: gym.Env, model: Algorithm, dataset: ReplayBuffer, noise: float = 0.1):
    obs = env.reset()
    dataset.add(obs)
    episode_length = 0
    done = False
    success_count = 0
    while not done and success_count < 15:
        action = model.predict(obs)
        action = action + noise * np.random.randn(*action.shape)
        obs, reward, done, info = env.step(action)
        episode_length += 1
        success_count += int(info["success"])
        if "discount" in info:
            discount = info["discount"]
        elif hasattr(env, "_max_episode_steps") and episode_length == env._max_episode_steps:
            discount = 1.0
        else:
            discount = 1 - float(done)
        dataset.add(obs, action, reward, done, discount)


def collect_dataset(
    task_path: str,
    policy_paths: List[str],
    save_path: str,
    random_ep: int = 1,
    expert_ep: int = 1,
    cross_ep: int = 1,
    init_noise: float = 0.0,
    policy_noise: float = 0.0,
) -> None:
    config = Config.load(task_path)
    config["env_kwargs"]["initialization_noise"] = init_noise
    expert_model = load(config, os.path.join(task_path, "best_model.pt"), device="auto", strict=False)
    del expert_model.eval_env
    env = expert_model.env
    dataset = ReplayBuffer(
        env.observation_space, env.action_space, capacity=1000000, distributed=False
    )  # hardcode to 1 mil max transitions
    dataset.setup()
    for _ in range(random_ep):
        collect_random_episode(env, dataset)

    used_expert = False
    for policy_path in policy_paths:
        if policy_path == task_path:
            current_model = expert_model
            num_ep = expert_ep
            used_expert = True
        else:
            current_model = load_from_path(os.path.join(policy_path, "best_model.pt"), device="auto", strict=False)
            del current_model.env
            del current_model.eval_env
            num_ep = cross_ep
        for _ in range(num_ep):
            collect_policy_episode(env, current_model, dataset, noise=policy_noise)
    assert used_expert, "Must have used expert policy"

    # save the dataset
    goal = config["env_kwargs"]["goal"]
    dataset.save_flat(os.path.join(save_path, "x{}_y{}".format(goal[0], goal[1])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--random-ep", type=int, default=2)
    parser.add_argument("--expert-ep", type=int, default=1)
    parser.add_argument("--cross-ep", type=int, default=1)
    parser.add_argument("--policies", type=str, default=None)
    parser.add_argument("--init-noise", type=float, default=0.3)
    parser.add_argument("--policy-noise", type=float, default=0.1)
    parser.add_argument("--path", "-p", type=str, required=True, help="output path")
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()

    policy_paths = sorted([os.path.join(args.policies, p) for p in os.listdir(args.policies)])
    if args.seed is None:
        task_paths = policy_paths.copy()
    else:
        task_paths = [policy_paths[args.seed]]

    for task in task_paths:
        collect_dataset(task, policy_paths, args.path)
