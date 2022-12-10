import argparse
import math
import os
import pickle
import random
import subprocess
from typing import List, Tuple

import gym
import metaworld
import numpy as np
from metaworld import Task, policies
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE

from research.datasets import ReplayBuffer


def collect_episode(
    src_env: gym.Env,
    dest_env: gym.Env,
    policy,
    dataset: ReplayBuffer,
    epsilon: float = 0.0,
    noise_type: str = "gaussian",
):
    obs = dest_env.reset()
    src_obs = src_env.reset()
    dataset.add(obs)
    episode_length = 0
    done = False
    while not done:
        action = policy.get_action(src_obs)
        if noise_type == "gaussian":
            action = action + epsilon * np.random.randn(*action.shape)
        elif noise_type == "uniform":
            if np.random.random() < epsilon:
                action = src_env.action_space.sample()

        obs, reward, done, info = dest_env.step(action)
        src_obs, _, _, src_info = src_env.step(action)
        episode_length += 1
        # TODO: Set done to true if the other env is done.
        if "discount" in info:
            discount = info["discount"]
        elif hasattr(dest_env, "_max_episode_steps") and episode_length == dest_env._max_episode_steps:
            discount = 1.0
        else:
            discount = 1 - float(done)

        if getattr(src_env, "curr_path_length", 0) == src_env.max_path_length:
            done = True  # set done to true manually for meta world.
        if info["success"] or src_info["success"]:
            done = True

        dataset.add(obs, action, reward, done, discount)


def collect_random_episode(env: gym.Env, dataset: ReplayBuffer):
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
        # Check to see if we hit the max episode length
        if getattr(env, "curr_path_length", 0) == env.max_path_length:
            done = True  # set done to true manually for meta world.


def sample_policy(benchmark, train: bool = True, env_name: str = None) -> Tuple:
    env_dict = benchmark.train_classes if train else benchmark.test_classes
    task_list = benchmark.train_tasks if train else benchmark.test_tasks
    if env_name is None:
        env_name = random.choice(list(env_dict.keys()))
    task = random.choice([task for task in task_list if task.env_name == env_name])
    env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[task.env_name + "-goal-observable"]()
    data = pickle.loads(task.data)
    data["partially_observable"] = False
    env.set_task(Task(env_name=env_name, data=pickle.dumps(data)))
    # Get the policy
    policy_name = "".join([s.capitalize() for s in task.env_name.split("-")])
    policy_name = policy_name.replace("PegInsert", "PegInsertion")
    policy_name = "Sawyer" + policy_name + "Policy"
    policy = vars(policies)[policy_name]()
    return env, policy


def collect_dataset(
    benchmark,
    names: List[str],
    path: str,
    train: bool = True,
    tasks_per_env: int = 10,
    expert_ep: int = 5,
    within_env_ep: int = 5,
    cross_env_ep: int = 10,
    random_ep: int = 2,
    epsilon: float = 0.1,
    noise_type: str = "gaussian",
) -> None:
    total_ep_per_env = expert_ep + cross_env_ep + within_env_ep + random_ep
    env_dict = benchmark.train_classes if train else benchmark.test_classes
    task_list = benchmark.train_tasks if train else benchmark.test_tasks
    for name in names:
        dest_env = env_dict[name]()
        tasks = [task for task in task_list if task.env_name == name]
        tasks = tasks[:tasks_per_env]
        for i, task in enumerate(tasks):
            dest_env.set_task(task)
            dataset = ReplayBuffer(
                dest_env.observation_space, dest_env.action_space, capacity=total_ep_per_env * 502, distributed=False
            )
            dataset.setup()

            # Setup the expert policy
            src_env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[task.env_name + "-goal-observable"]()
            data = pickle.loads(task.data)
            data["partially_observable"] = False
            src_env.set_task(Task(env_name=task.env_name, data=pickle.dumps(data)))  # POTENTIAL ERROR
            policy_name = "".join([s.capitalize() for s in task.env_name.split("-")])
            policy_name = policy_name.replace("PegInsert", "PegInsertion")
            policy_name = "Sawyer" + policy_name + "Policy"
            policy = vars(policies)[policy_name]()
            for _ in range(expert_ep):
                collect_episode(src_env, dest_env, policy, dataset, epsilon=epsilon, noise_type=noise_type)

            # Now collect the other episodes
            for _ in range(within_env_ep):
                src_env, policy = sample_policy(benchmark, train=train, env_name=name)
                collect_episode(src_env, dest_env, policy, dataset, epsilon=epsilon)
            for _ in range(cross_env_ep):
                src_env, policy = sample_policy(benchmark, train=train, env_name=None)
                collect_episode(src_env, dest_env, policy, dataset, epsilon=epsilon)
            for _ in range(random_ep):
                collect_random_episode(dest_env, dataset)
            save_path = os.path.join(path, "cls_{}_task_{}".format(name, i))
            dataset.save_flat(save_path)


def seed(seed):
    random.seed(seed)
    np.random.seed(seed)


if __name__ == "__main__":
    # Only execute this code if this script is called
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", "-b", type=str, default="ml1_pick-place-v2")
    parser.add_argument("--tasks-per-env", type=int, default=None)
    parser.add_argument("--cross-env-ep", type=int, default=10)
    parser.add_argument("--within-env-ep", type=int, default=5)
    parser.add_argument("--expert-ep", type=int, default=2)
    parser.add_argument("--random-ep", type=int, default=1)
    parser.add_argument("--epsilon", type=float, default=0.1, help="probabiliyt of taking a random action")
    parser.add_argument("--noise-type", type=str, default="gaussain")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--valid", action="store_true", default=False)
    parser.add_argument("--path", "-p", type=str, required=True, help="output path")

    args = parser.parse_args()

    # Seed everything for deterministic creation of benchmarks
    seed(0)
    if "_" in args.benchmark:
        benchmark, task = args.benchmark.split("_")
        task = (task,)
    else:
        benchmark, task = args.benchmark, ()
    benchmark = vars(metaworld)[benchmark.upper()](*task)

    # Compute the chunk size of each worker
    env_dict = benchmark.test_classes if args.valid else benchmark.train_classes
    chunk_size = math.ceil(len(env_dict) / args.num_workers)

    if args.rank == 0:
        assert not args.valid
        # We are the coordinating thread, launch the jobs for train
        processes = []
        for i in range(args.num_workers):
            command_list = ["python", __file__]
            command_list.extend(["--benchmark", args.benchmark])
            command_list.extend(["--tasks-per-env", args.tasks_per_env])
            command_list.extend(["--cross-env-ep", args.cross_env_ep])
            command_list.extend(["--within-env-ep", args.within_env_ep])
            command_list.extend(["--expert-ep", args.expert_ep])
            command_list.extend(["--random-ep", args.random_ep])
            command_list.extend(["--epsilon", args.epsilon])
            command_list.extend(["--noise-type", args.noise_type])
            command_list.extend(["--num-workers", args.num_workers])
            command_list.extend(["--path", args.path])
            command_list.extend(["--rank", i + 1])
            command_list = list(map(str, command_list))
            proc = subprocess.Popen(command_list)
            processes.append(proc)

        print("[Dataset Collection] Waiting for completion of train.")
        try:
            exit_codes = [p.wait() for p in processes]
        except KeyboardInterrupt:
            for p in processes:
                try:
                    p.terminate()
                except OSError:
                    pass
                p.wait()

        # Launch the jobs for validation
        processes = []
        for i in range(args.num_workers):
            command_list = ["python", __file__]
            command_list.extend(["--benchmark", args.benchmark])
            command_list.extend(["--tasks-per-env", args.tasks_per_env])
            command_list.extend(["--cross-env-ep", args.cross_env_ep])
            command_list.extend(["--within-env-ep", args.within_env_ep])
            command_list.extend(["--expert-ep", args.expert_ep])
            command_list.extend(["--random-ep", args.random_ep])
            command_list.extend(["--epsilon", args.epsilon])
            command_list.extend(["--noise-type", args.noise_type])
            command_list.extend(["--num-workers", args.num_workers])
            command_list.extend(["--path", args.path + "_valid"])
            command_list.extend(["--rank", i + 1])
            command_list.append("--valid")
            command_list = list(map(str, command_list))
            proc = subprocess.Popen(command_list)
            processes.append(proc)

        print("[Dataset Collection] Waiting for completion of validation.")
        try:
            exit_codes = [p.wait() for p in processes]
        except KeyboardInterrupt:
            for p in processes:
                try:
                    p.terminate()
                except OSError:
                    pass
                p.wait()

    else:
        # We are a worker thread
        # If we're a worker thread, our rank is the index into the env_list
        env_dict = benchmark.test_classes if args.valid else benchmark.train_classes
        names = list(env_dict.keys())[(args.rank - 1) * chunk_size : args.rank * chunk_size]
        seed(args.rank)
        collect_dataset(
            benchmark,
            names,
            args.path,
            train=not args.valid,
            tasks_per_env=args.tasks_per_env,
            expert_ep=args.expert_ep,
            within_env_ep=args.within_env_ep,
            cross_env_ep=args.cross_env_ep,
            random_ep=args.random_ep,
            epsilon=args.epsilon,
            noise_type=args.noise_type,
        )
        # Done.
