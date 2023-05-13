import argparse
import os
import pickle
import random
from typing import Optional

import gym
import metaworld
import numpy as np
from metaworld import Task, policies
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE

from research.datasets import ReplayBuffer
from research.utils.evaluate import EvalMetricTracker


def collect_episode(
    env: gym.Env,
    policy_env: gym.Env,
    policy,
    dataset: ReplayBuffer,
    metric_tracker: EvalMetricTracker,
    epsilon: float = 0.0,
    noise_type: str = "gaussian",
    init_obs: Optional[np.ndarray] = None,
):
    if init_obs is None:
        obs = env.reset()
    else:
        obs = init_obs
    policy_obs = policy_env.reset()

    dataset.add(obs)
    episode_length = 0
    success_steps = 25
    done = False

    metric_tracker.reset()

    while not done:
        action = policy.get_action(policy_obs)
        if noise_type == "gaussian":
            action = action + epsilon * np.random.randn(*action.shape)
        elif noise_type == "uniform":
            action = action + epsilon * policy_env.action_space.sample()
        elif noise_type == "random":
            action = policy_env.action_space.sample()
        else:
            raise ValueError("Invalid noise type provided.")

        action = np.clip(action, -1 + 1e-5, 1 - 1e-5)  # Clip the action to the valid range after noise.
        obs, reward, done, info = env.step(action)
        policy_obs, _, _, src_info = policy_env.step(action)
        metric_tracker.step(reward, info)

        episode_length += 1
        # TODO: Set done to true if the other env is done.
        if "discount" in info:
            discount = info["discount"]
        elif hasattr(env.unwrapped, "_max_episode_steps") and episode_length == env.unwrapped._max_episode_steps:
            discount = 1.0
        else:
            discount = 1 - float(done)

        # If the other env finishes we have to terminate
        if getattr(policy_env, "curr_path_length", 0) == policy_env.max_path_length:
            done = True  # set done to true manually for meta world.

        if info["success"]:
            success_steps -= 1

        if success_steps == 0:
            done = True  # If we have been successful for a while, break.

        dataset.add(obs, action, reward, done, discount)


if __name__ == "__main__":
    # Only execute this code if this script is called
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True, help="A Metaworld task name, like drawer-open-v2")
    parser.add_argument("--cross-env-ep", type=int, default=100)
    parser.add_argument("--within-env-ep", type=int, default=100)
    parser.add_argument("--expert-ep", type=int, default=100)
    parser.add_argument("--random-ep", type=int, default=100)
    parser.add_argument("--epsilon", type=float, default=1.0, help="magnitude of gaussian noise.")
    parser.add_argument("--path", "-p", type=str, required=True, help="output path")

    args = parser.parse_args()

    # Make the path, do not double write datasets
    os.makedirs(args.path, exist_ok=False)

    env = gym.make("mw_" + args.env)

    ep_length = env.unwrapped._max_episode_steps + 2
    total_ep = args.cross_env_ep + args.within_env_ep + args.expert_ep + args.random_ep
    dataset = ReplayBuffer(env.observation_space, env.action_space, capacity=total_ep * ep_length, distributed=False)
    metric_tracker = EvalMetricTracker()

    # Construct the same environment
    observable_env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[args.env + "-goal-observable"]()

    # Create the expert policy
    policy_name = "".join([s.capitalize() for s in args.env.split("-")])
    policy_name = policy_name.replace("PegInsert", "PegInsertion")
    policy_name = "Sawyer" + policy_name + "Policy"
    policy = vars(policies)[policy_name]()

    for i in range(args.expert_ep):
        obs = env.reset()

        # Get the the random vector
        _last_rand_vec = env.unwrapped._last_rand_vec
        data = dict(rand_vec=_last_rand_vec)
        data["partially_observable"] = False
        data["env_cls"] = type(env.unwrapped._env)
        task = Task(env_name=args.env, data=pickle.dumps(data))  # POTENTIAL ERROR
        observable_env.set_task(task)

        collect_episode(
            env,
            observable_env,
            policy,
            dataset,
            metric_tracker,
            epsilon=args.epsilon,
            noise_type="gaussian",
            init_obs=obs,
        )
        if (i + 1) % 20 == 0:
            print("Finished", i + 1, "expert ep.")

    # Now collect the other episodes
    observable_env._freeze_rand_vec = False  # Unfreeze the random vector.
    for i in range(args.within_env_ep):
        collect_episode(
            env,
            observable_env,
            policy,
            dataset,
            metric_tracker,
            epsilon=args.epsilon,
            noise_type="gaussian",
            init_obs=None,
        )
        if (i + 1) % 20 == 0:
            print("Finished", i + 1, "within env ep.")

    for i in range(args.random_ep):
        collect_episode(
            env,
            observable_env,
            policy,
            dataset,
            metric_tracker,
            epsilon=args.epsilon,
            noise_type="random",
            init_obs=None,
        )
        if (i + 1) % 20 == 0:
            print("Finished", i + 1, "random ep.")

    env_names = [name[: -len("-goal-observable")] for name in ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE.keys()]
    for i in range(args.cross_env_ep):
        env_name = random.choice(env_names)
        observable_env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name + "-goal-observable"]()

        # Create the expert policy for this env.
        policy_name = "".join([s.capitalize() for s in env_name.split("-")])
        policy_name = policy_name.replace("PegInsert", "PegInsertion")
        policy_name = "Sawyer" + policy_name + "Policy"
        policy = vars(policies)[policy_name]()

        collect_episode(
            env,
            observable_env,
            policy,
            dataset,
            metric_tracker,
            epsilon=args.epsilon,
            noise_type="gaussian",
            init_obs=None,
        )
        if (i + 1) % 20 == 0:
            print("Finished", i + 1, "cross env ep.")

    fname = dataset.save_flat(args.path)
    fname = os.path.basename(fname)

    metrics = metric_tracker.export()
    print("Metrics:")
    print(metrics)
    with open(os.path.join(args.path, "metrics.txt"), "a") as f:
        f.write("Collected data: " + str(fname) + "\n")
        for k, v in metrics.items():
            f.write(k + ": " + str(v) + "\n")
