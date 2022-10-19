import collections
from typing import Any, Dict

import gym
import numpy as np
import torch

from . import utils

MAX_METRICS = {"success", "is_success"}
LAST_METRICS = {"goal_distance"}
MEAN_METRICS = {}


class EvalMetricTracker(object):
    """
    A simple class to make keeping track of eval metrics easy.
    Usage:
        Call reset before each episode starts
        Call step after each environment step
        call export to get the final metrics
    """

    def __init__(self):
        self.metrics = collections.defaultdict(list)
        self.ep_length = 0
        self.ep_reward = 0
        self.ep_metrics = collections.defaultdict(list)

    def reset(self) -> None:
        if self.ep_length > 0:
            # Add the episode to overall metrics
            self.metrics["reward"].append(self.ep_reward)
            self.metrics["length"].append(self.ep_length)
            for k, v in self.ep_metrics.items():
                if k in MAX_METRICS:
                    self.metrics[k].append(np.max(v))
                elif k in LAST_METRICS:  # Append the last value
                    self.metrics[k].append(v[-1])
                elif k in MEAN_METRICS:
                    self.metrics[k].append(np.mean(v))
                else:
                    self.metrics[k].append(np.sum(v))

            self.ep_length = 0
            self.ep_reward = 0
            self.ep_metrics = collections.defaultdict(list)

    def step(self, reward: float, info: Dict) -> None:
        self.ep_length += 1
        self.ep_reward += reward
        for k, v in info.items():
            if isinstance(v, float) or np.isscalar(v):
                self.ep_metrics[k].append(v)

    def add(self, k: str, v: Any):
        self.metrics[k].append(v)

    def export(self) -> Dict:
        if self.ep_length > 0:
            # We have one remaining episode to log, make sure to get it.
            self.reset()
        metrics = {k: np.mean(v) for k, v in self.metrics.items()}
        metrics["reward_std"] = np.std(self.metrics["reward"])
        return metrics


def eval_policy(env: gym.Env, model, num_ep: int = 10) -> Dict:
    metric_tracker = EvalMetricTracker()

    for _ in range(num_ep):
        # Reset Metrics
        done = False
        ep_length, ep_reward = 0, 0
        obs = env.reset()
        metric_tracker.reset()
        while not done:
            batch = dict(obs=obs)
            with torch.no_grad():
                action = model.predict(batch)
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            metric_tracker.step(reward, info)
            ep_length += 1

        if hasattr(env, "get_normalized_score"):
            metric_tracker.add("score", env.get_normalized_score(ep_reward))

    return metric_tracker.export()
