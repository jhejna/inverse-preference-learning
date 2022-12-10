import argparse
import os
from typing import Tuple

import gym
import numpy as np

import research  # to register the environments
from research.datasets import ReplayBuffer

TARGET_POSITIONS_TRAIN = {
    "point_mass": [
        (1, 0),
        (1, 1),
        (0, 1),
        (0, -1),
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (1, -1),
        (1, 0.5),
        (0.5, 1),
        (-1, 0.5),
        (-0.5, 1),
        (1, -0.5),
        (0.5, -1),
        (-1, -0.5),
        (-0.5, -1),
    ],
    "reacher": [
        (0 * (np.pi / 2), 1),
        (1 * (np.pi / 2), 1),
        (2 * (np.pi / 2), 1),
        (3 * (np.pi / 2), 1),
        (1 * (np.pi / 4), 0.66),
        (3 * (np.pi / 4), 0.66),
        (5 * (np.pi / 4), 0.66),
        (7 * (np.pi / 4), 0.66),
        (1 * (np.pi / 8), 0.33),
        (5 * (np.pi / 8), 0.33),
        (9 * (np.pi / 8), 0.33),
        (13 * (np.pi / 8), 0.33),
    ],
}
TARGET_POSITIONS_TRAIN["point_mass_random"] = TARGET_POSITIONS_TRAIN["point_mass"]

TARGET_POSITIONS_VALID = {
    "point_mass": [
        (1, 0.75),
        (-0.75, 0.8),
    ],
    "reacher": [
        (3 * (np.pi / 8), 0.5),
    ],
}
TARGET_POSITIONS_VALID["point_mass_random"] = TARGET_POSITIONS_VALID["point_mass"]

parser = argparse.ArgumentParser()
parser.add_argument("--num-steps", "-n", type=int, default=10000)
parser.add_argument("--path", type=str, required=True)
parser.add_argument("--vision", action="store_true", default=False)
parser.add_argument("--env", type=str, choices=["point_mass", "point_mass_random", "reacher"], required=True)
args = parser.parse_args()

env_id = {
    "point_mass": "Goal_point_massReach_l2",
    "point_mass_random": "Goal_point_massReach_l2_random",
    "reacher": "Goal_reacherReach_l2",
}[args.env]
if args.vision:
    env_id += "-vision"
env_id += "-v0"


def create_random_dataset(target_pos: Tuple[float, float], path: str) -> None:
    env = gym.make(env_id, task_kwargs=dict(target_pos=target_pos))
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
        elif hasattr(env, "_max_episode_stes") and episode_length == env._max_episode_steps:
            discount = 1.0
        else:
            discount = 1 - float(done)

        num_steps += 1

        dataset.add(obs, action, reward, done, discount)

    # save the dataset
    save_path = os.path.join(path, "x{}_y{}".format(target_pos[0], target_pos[1]))
    dataset.save_flat(save_path)


# Create the train and validation datasets
for target_pos in TARGET_POSITIONS_TRAIN[args.env]:
    create_random_dataset(target_pos, args.path)
print("Finished train.")
for target_pos in TARGET_POSITIONS_VALID[args.env]:
    create_random_dataset(target_pos, args.path + "_valid")
print("Finished validation.")
