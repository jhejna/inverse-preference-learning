import argparse
import os

import gym
import numpy as np

import research
from research.datasets.feedback_buffer import PairwiseComparisonDataset
from research.datasets.replay_buffer import ReplayBuffer

if __name__ == "__main__":
    # This is a short script that generates a pairwise preference dataset from a ReplayBuffer

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True, help="Env associated with the dataset.")
    parser.add_argument("--path", type=str, required=True, help="Path to the ReplayBuffer")
    parser.add_argument("--output", type=str, required=True, help="Output path for the dataset")
    parser.add_argument("--capacity", type=int, default=20000, help="How big to make the dataset")
    parser.add_argument("--segment-size", type=int, default=25, help="How large to make segments")
    parser.add_argument("--discount", type=float, default=0.99)
    args = parser.parse_args()

    assert os.path.exists(args.path)

    env = gym.make(args.env)
    capacity = args.capacity  # Rename for ease

    # Set to not distributed so we immediately load all of the data
    replay_buffer = ReplayBuffer(
        env.observation_space, env.action_space, distributed=False, path=args.path, discount=args.discount
    )
    feedback_dataset = PairwiseComparisonDataset(
        env.observation_space,
        env.action_space,
        discount=args.discount,
        segment_size=args.segment_size,
        capacity=capacity,
    )

    batch = replay_buffer.sample(batch_size=2 * capacity, stack=args.segment_size, pad=0)

    returns = np.sum(batch["reward"] * np.power(replay_buffer.discount, np.arange(batch["reward"].shape[1])), axis=1)
    queries = dict(
        obs_1=batch["obs"][:capacity],
        obs_2=batch["obs"][capacity:],
        action_1=batch["action"][:capacity],
        action_2=batch["action"][capacity:],
    )
    labels = 1.0 * (returns[:capacity] < returns[capacity:])
    feedback_dataset.add(queries, labels)  # Write the data into the buffer.
    print("Feedback Dataset size", len(feedback_dataset))
    feedback_dataset.save(args.output)
    # Finished.
