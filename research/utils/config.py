import copy
import gc
import importlib
import os
import pprint
import random
from typing import Any, Dict, Optional, Union

import gym
import numpy as np
import torch
import yaml

import research

from . import schedules
from .trainer import Trainer
from .utils import flatten_dict

DEFAULT_NETWORK_KEY = "network"


def get_env(env: gym.Env, env_kwargs: Dict, wrapper: Optional[gym.Env], wrapper_kwargs: Dict) -> gym.Env:
    # Try to get the environment
    try:
        env = vars(research.envs)[env](**env_kwargs)
    except KeyError:
        env = gym.make(env, **env_kwargs)
    if wrapper is not None:
        env = vars(research.envs)[wrapper](env, **wrapper_kwargs)
    return env


class BareConfig(object):
    """
    This is a bare copy of the config that does not require importing any of the research packages.
    This file has been copied out for use in the tools/trainer etc. to avoid loading heavy packages
    when the goal is just to create configs. It defines no structure.
    """

    def __init__(self):
        # Define the necesary structure for a complete training configuration
        self._parsed = False
        self.config = dict()

    def save(self, path: str) -> None:
        if os.path.isdir(path):
            path = os.path.join(path, "config.yaml")
        with open(path, "w") as f:
            yaml.dump(self.config, f)

    def update(self, d: Dict) -> None:
        self.config.update(d)

    @classmethod
    def load(cls, path: str) -> "Config":
        if os.path.isdir(path):
            path = os.path.join(path, "config.yaml")
        with open(path, "r") as f:
            data = yaml.load(f, Loader=yaml.Loader)
        config = cls()
        config.update(data)
        return config

    def get(self, key: str, default: Optional[Any] = None):
        return self.config.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self.config[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.config[key] = value

    def __contains__(self, key: str):
        return self.config.__contains__(key)

    def __str__(self) -> str:
        return pprint.pformat(self.config, indent=4)

    def copy(self) -> "Config":
        assert not self.parsed, "Cannot copy a parsed config"
        config = type(self)()
        config.config = copy.deepcopy(self.config)
        return config


class Config(BareConfig):
    def __init__(self):
        super().__init__()
        # Define necesary fields

        # Manage seeding.
        self._seeded = False
        self.config["seed"] = None

        # Env Args
        self.config["env"] = None
        self.config["env_kwargs"] = {}

        self.config["eval_env"] = None
        self.config["eval_env_kwargs"] = {}

        self.config["wrapper"] = None
        self.config["wrapper_kwargs"] = {}

        # Algorithm Args
        self.config["alg"] = None
        self.config["alg_kwargs"] = {}

        # Dataset Args
        self.config["dataset"] = None
        self.config["dataset_kwargs"] = {}

        self.config["validation_dataset"] = None
        self.config["validation_dataset_kwargs"] = None

        # Processor arguments
        self.config["processor"] = None
        self.config["processor_kwargs"] = {}

        # Optimizer Args
        self.config["optim"] = None
        self.config["optim_kwargs"] = {}

        # Network Args
        self.config["network"] = None
        self.config["network_kwargs"] = {}

        # Checkpoint
        self.config["checkpoint"] = None

        # Schedule args
        self.config["schedule"] = None
        self.config["schedule_kwargs"] = {}

        self.config["trainer_kwargs"] = {}

    @property
    def parsed(self):
        return self._parsed

    @staticmethod
    def _parse_helper(d: Dict) -> None:
        for k, v in d.items():
            if isinstance(v, list) and len(v) > 1 and v[0] == "import":
                # parse the value to an import
                d[k] = getattr(importlib.import_module(v[1]), v[2])
            elif isinstance(v, dict):
                Config._parse_helper(v)

    def parse(self) -> "Config":
        config = self.copy()
        Config._parse_helper(config.config)
        config._parsed = True
        # Before we make any objects, make sure we set the seeds.
        if self.config["seed"] is not None:
            torch.manual_seed(self.config["seed"])
            np.random.seed(self.config["seed"])
            random.seed(self.config["seed"])
        return config

    def flatten(self) -> Dict:
        """Returns a flattened version of the config where '.' separates nested values"""
        return flatten_dict(self.config)

    def __setitem__(self, key: str, value: Any):
        if key not in self.config:
            raise ValueError(
                "Attempting to set an out of structure key: " + key + ". Configs must follow the format in config.py"
            )
        super().__setitem__(key, value)

    def get_train_env(self):
        assert self.parsed
        # We need to return an environment with the correct spaces
        # If we don't have a train env, use the eval env to create one.
        if self["env"] is None:
            # Construct an empty env with the same state and action space as the eval env.
            assert self["eval_env"] is not None, "If no train env, must have eval_env"
            # Note that we can't call self.get_eval_env here because that returns None under certain conditions.
            eval_env = self.get_eval_env()
            env = research.envs.base.Empty(
                observation_space=eval_env.observation_space, action_space=eval_env.action_space
            )
            del eval_env
            gc.collect()  # manually run the garbage collector.
            return env
        else:
            return get_env(self["env"], self["env_kwargs"], self["wrapper"], self["wrapper_kwargs"])

    def get_eval_env(self):
        assert self.parsed
        # Return the evalutaion environment.
        if self["eval_env"] is None:
            return get_env(self["env"], self["env_kwargs"], self["wrapper"], self["wrapper_kwargs"])
        else:
            return get_env(self["eval_env"], self["eval_env_kwargs"], self["wrapper"], self["wrapper_kwargs"])

    def get_schedules(self):
        assert self.parsed

        # Fetch the schedulers. If we don't have an optim dict, change it to one.
        if not isinstance(self["schedule"], dict):
            schedulers = {DEFAULT_NETWORK_KEY: self["schedule"]}
            schedulers_kwargs = {DEFAULT_NETWORK_KEY: self["schedule_kwargs"]}
        else:
            schedulers = self["schedule"]
            schedulers_kwargs = self["schedule_kwargs"]

        # Make sure we fetch the schedule if its provided as a string
        for k in schedulers.keys():
            if isinstance(schedulers[k], str):
                schedulers[k] = torch.optim.lr_scheduler.LambdaLR
                # Create the lambda function, and pass it in as a keyword arg
                schedulers_kwargs[k] = dict(lr_lambda=vars(schedules)[self["schedule"]](**schedulers_kwargs[k]))

        return schedulers, schedulers_kwargs

    def get_model(self, device: Union[str, torch.device] = "auto"):
        assert self.parsed
        # Return the model
        alg_class = vars(research.algs)[self["alg"]]
        dataset_class = None if self["dataset"] is None else vars(research.datasets)[self["dataset"]]
        validation_dataset_class = (
            None if self["validation_dataset"] is None else vars(research.datasets)[self["validation_dataset"]]
        )
        network_class = None if self["network"] is None else vars(research.networks)[self["network"]]
        optim_class = None if self["optim"] is None else vars(torch.optim)[self["optim"]]
        processor_class = None if self["processor"] is None else vars(research.processors)[self["processor"]]
        schedulers_class, schedulers_kwargs = self.get_schedules()
        env = self.get_train_env()
        algo = alg_class(
            env,
            network_class,
            dataset_class,
            network_kwargs=self["network_kwargs"],
            dataset_kwargs=self["dataset_kwargs"],
            validation_dataset_class=validation_dataset_class,
            validation_dataset_kwargs=self["validation_dataset_kwargs"],
            processor_class=processor_class,
            processor_kwargs=self["processor_kwargs"],
            optim_class=optim_class,
            optim_kwargs=self["optim_kwargs"],
            schedulers_class=schedulers_class,
            schedulers_kwargs=schedulers_kwargs,
            checkpoint=self["checkpoint"],
            device=device,
            **self["alg_kwargs"],
        )
        return algo

    def get_trainer(self):
        assert self.parsed
        # Returns a Trainer Object that can be used to train a model
        if (
            self["trainer_kwargs"].get("eval_fn", None) is None
            or self["trainer_kwargs"].get("subproc_eval", False) == True
        ):
            eval_env = None
        else:
            eval_env = self.get_eval_env()
        # Return the trainer...
        return Trainer(eval_env, **self["trainer_kwargs"])
