import research
from research.datasets import ReplayBuffer
import gym
import argparse
import os
import numpy as np
import metaworld
import random

# Seed everything for deterministic choice order.
SEED = 0
random.seed(SEED)
np.random.seed(SEED)

parser = argparse.ArgumentParser()
parser.add_argument('--benchmark', '-b', type=str, default='ml1_pick-place-v2')
parser.add_argument('--tasks-per-env', type=int, default=None)
parser.add_argument('--num-steps', type=int, default=2500)
parser.add_argument('--path', '-p', type=str, required=True, help="output path")

args = parser.parse_args()

def create_random_dataset(env, max_steps):
    dataset = ReplayBuffer(env.observation_space, env.action_space, capacity=args.num_steps, cleanup=False)
    # Collect data
    num_steps = 0
    done = True
    while num_steps < args.num_steps:
        if done:
            obs = env.reset()
            dataset.add(obs)
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        num_steps += 1
        # Determine the discount factor.
        if 'discount' in info:
            discount = info['discount']
        elif hasattr(env, "_max_episode_steps") and episode_length == env._max_episode_steps:
            discount = 1.0
        else:
            discount = 1 - float(done)
        if getattr(env, 'curr_path_length', 0) == env.max_path_length:
            done = True # set done to true manually for meta world.
        dataset.add(obs, action, reward, done, discount)
    return dataset # Return the dataset object

if '_' in args.benchmark:
    benchmark, task = args.benchmark.split('_')
    task = (task,)
else:
    benchmark, task = args.benchmark, ()

benchmark = vars(metaworld)[benchmark.upper()](*task)

# First create the train benchmark
for name, env_cls in benchmark.train_classes.items():
    env = env_cls()
    tasks = [task for task in benchmark.train_tasks if task.env_name == name]
    if args.tasks_per_env is not None:
        tasks = tasks[:args.tasks_per_env]
    for i, task in enumerate(tasks):
        env.set_task(task)
        # Construct the task
        dataset = create_random_dataset(env, args.num_steps)
        path = os.path.join(args.path, "cls_{}_task_{}".format(name, i))
        dataset.save(path)
print("Finished train.")
for name, env_cls in benchmark.test_classes.items():
    env = env_cls()
    tasks = [task for task in benchmark.test_tasks if task.env_name == name]
    if args.tasks_per_env is not None:
        tasks = tasks[:args.tasks_per_env]
    for i, task in enumerate(tasks):
        env.set_task(task)
        # Construct the task
        dataset = create_random_dataset(env, args.num_steps)
        path = os.path.join(args.path + "_valid", "cls_{}_task_{}".format(name, i))
        dataset.save(path)
print("Finished validation.")
