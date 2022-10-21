import copy
import importlib
import os
import pprint
from typing import Any, Dict

import yaml

from .utils import flatten_dict


class Config(object):
    def __init__(self):
        # Define the necesary structure for a complete training configuration
        self.parsed = False
        self.config = dict()

        # Env Args
        self.config["env"] = None
        self.config["env_kwargs"] = {}
        # self.config['parallel_envs'] = False # TODO: for future parallel env support

        self.config["eval_env"] = None
        self.config["eval_env_kwargs"] = {}

        self.config["wrapper"] = None
        self.config["wrapper_kwargs"] = {}

        # Vector args. TODO: for future parallel env support
        # self.config['vec_kwargs'] = {}

        # Algorithm Args
        self.config["alg"] = None
        self.config["alg_kwargs"] = {}

        # Dataset args
        self.config["dataset"] = None
        self.config["dataset_kwargs"] = {}
        self.config["validation_dataset_kwargs"] = None

        # Dataloader arguments
        self.config["collate_fn"] = None
        self.config["batch_size"] = None

        # Processor arguments
        self.config["processor"] = None
        self.config["processor_kwargs"] = {}

        # Optimizer Args
        self.config["optim"] = None
        self.config["optim_kwargs"] = {}

        # network Args
        self.config["network"] = None
        self.config["network_kwargs"] = {}

        # Schedule args
        self.config["schedule"] = None
        self.config["schedule_kwargs"] = {}

        # General arguments
        self.config["checkpoint"] = None
        self.config["seed"] = None  # Does nothing right now.
        self.config["train_kwargs"] = {}

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
        return config

    def update(self, d: Dict) -> None:
        self.config.update(d)

    def save(self, path: str) -> None:
        if os.path.isdir(path):
            path = os.path.join(path, "config.yaml")
        with open(path, "w") as f:
            yaml.dump(self.config, f)

    @classmethod
    def load(cls, path: str) -> "Config":
        if os.path.isdir(path):
            path = os.path.join(path, "config.yaml")
        with open(path, "r") as f:
            data = yaml.load(f, Loader=yaml.Loader)
        config = cls()
        config.update(data)
        return config

    def flatten(self) -> Dict:
        """Returns a flattened version of the config where '.' separates nested values"""
        return flatten_dict(self.config)

    def __getitem__(self, key: str) -> Any:
        return self.config[key]

    def __setitem__(self, key: str, value: Any):
        if key not in self.config:
            raise ValueError("Attempting to set an out of structure key. Configs must follow the format in config.py")
        self.config[key] = value

    def __contains__(self, key: str) -> bool:
        return self.config.__contains__(key)

    def __str__(self) -> str:
        return pprint.pformat(self.config, indent=4)

    def copy(self) -> "Config":
        config = type(self)()
        config.config = copy.deepcopy(self.config)
        return config
