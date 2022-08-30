from functools import partial
from typing import Optional, Union

import gym
import torch

import research


class NetworkContainer(torch.nn.Module):
    CONTAINERS = []

    def __init__(self, observation_space: gym.Space, action_space: gym.Space, **kwargs) -> None:
        super().__init__()
        # Check to make sure that we have the required classes
        assert all([container + "_class" in kwargs for container in self.CONTAINERS])
        # save the classes and containers
        self._classes = {container: kwargs[container + "_class"] for container in self.CONTAINERS}
        self._kwargs = {container: kwargs.get(container + "_kwargs", dict()) for container in self.CONTAINERS}
        self._spaces = {}

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
