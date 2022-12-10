import argparse
import os
import random

import gym
import numpy as np

import research
from research.datasets import ReplayBuffer

EE_POS_LOWER_LIMIT = np.array([0.1, -0.4, 0.12])
EE_POS_UPPER_LIMIT = np.array([1.0, 0.4, 1.0])

parser = argparse.ArgumentParser()
parser.add_argument("--num-steps", "-n", type=int, default=2000)
parser.add_argument("--ep-length", "-l", type=int, default=200)
parser.add_argument("--num-tasks", type=int, default=50)
parser.add_argument("--path", type=str, required=True)
parser.add_argument("--seed", type=int, default=0, required=False)
parser.add_argument("--noise-magnitude", type=float, default=0.5, required=False)
args = parser.parse_args()

# Deterministically seed everything
np.random.seed(args.seed)
random.seed(args.seed)

# Construct validation first to keep it detemrinistc
TARGET_POSITIONS_VALID = [np.random.uniform(low=EE_POS_LOWER_LIMIT, high=EE_POS_UPPER_LIMIT) for _ in range(10)]
TARGET_POSITIONS_TRAIN = [
    np.random.uniform(low=EE_POS_LOWER_LIMIT, high=EE_POS_UPPER_LIMIT) for _ in range(args.num_tasks)
]


def create_random_dataset(target_pos: np.ndarray, path: str) -> None:
    initialization_noise = args.noise_magnitude
    env = gym.make("PyBulletPandaReach-v0", goal=target_pos, initialization_noise=initialization_noise)
    env._max_episode_steps = args.ep_length
    dataset = ReplayBuffer(env.observation_space, env.action_space, capacity=args.num_steps, distributed=False)
    dataset.setup()
    # Collect data
    num_steps = 0
    done = True
    episode_length = 0
    while num_steps < args.num_steps:
        if done:
            obs = env.reset()
            episode_length = 0
            dataset.add(obs)
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        # Determine the discount factor.
        if "discount" in info:
            discount = info["discount"]
        elif hasattr(env, "_max_episode_steps") and episode_length == env._max_episode_steps:
            discount = 1.0
        else:
            discount = 1 - float(done)

        num_steps += 1
        dataset.add(obs, action, reward, done, discount)

    # save the dataset
    save_path = os.path.join(path, "x{}_y{}_z{}".format(target_pos[0], target_pos[1], target_pos[2]))
    dataset.save_flat(save_path)


# Create the train and validation datasets
for target_pos in TARGET_POSITIONS_TRAIN:
    create_random_dataset(target_pos, args.path)
    print("Finished", target_pos)
print("Finished train.")
for target_pos in TARGET_POSITIONS_VALID:
    create_random_dataset(target_pos, args.path + "_valid")
print("Finished validation.")
