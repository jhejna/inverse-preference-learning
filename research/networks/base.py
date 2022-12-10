from functools import partial
from typing import Optional, Union

import gym
import torch

import research


class NetworkContainer(torch.nn.Module):
    CONTAINERS = []

    def __init__(self, observation_space: gym.Space, action_space: gym.Space, **kwargs) -> None:
        super().__init__()
        # save the classes and containers
        base_kwargs = {k: v for k, v in kwargs.items() if not k.endswith("_class") and not k.endswith("_kwargs")}
        self._kwargs = dict()
        self._classes = dict()
        for container in self.CONTAINERS:
            container_class = kwargs.get(container + "_class", torch.nn.Identity)
            self._classes[container] = container_class
            if container_class is torch.nn.Identity:
                container_kwargs = dict()
            else:
                container_kwargs = base_kwargs.copy()
                container_kwargs.update(kwargs.get(container + "_kwargs", dict()))
            self._kwargs[container] = container_kwargs

        self._spaces = dict()

        self.action_space = action_space
        self.observation_space = observation_space

        # Now create all of the networks, save spaces along the way.
        output_space = self.observation_space
        for container in self.CONTAINERS:
            self._spaces[container] = output_space
            reset_fn = partial(self._reset, container=container)
            setattr(self, "reset_" + container, reset_fn)
            reset_fn()  # Call reset to instantiate it.
            if hasattr(getattr(self, container), "output_space"):
                # update the output space
                output_space = getattr(self, container).output_space

    def _reset(self, container: str, device: Optional[Union[str, torch.device]] = None) -> None:
        # Fetch the cachced parameters
        network_class = self._classes[container]
        network_kwargs = self._kwargs[container]
        observation_space = self._spaces[container]
        # Refresh
        network_class = vars(research.networks)[network_class] if isinstance(network_class, str) else network_class
        network = network_class(observation_space, self.action_space, **network_kwargs)
        if device is not None:
            network = network.to(device)
        setattr(self, container, network)


class ActorCriticPolicy(NetworkContainer):
    CONTAINERS = ["encoder", "actor", "critic"]


class ActorCriticRewardPolicy(NetworkContainer):
    CONTAINERS = ["encoder", "actor", "critic", "reward"]
