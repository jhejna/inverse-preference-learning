import copy
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Type, Union

import gym
import numpy as np
import torch

from research.processors.base import IdentityProcessor, Processor
from research.utils import utils


class Algorithm(ABC):
    _save_keys: Set[str]
    _compiled: bool

    def __init__(
        self,
        env: gym.Env,
        network_class: Type[torch.nn.Module],
        dataset_class: Union[Type[torch.utils.data.IterableDataset], Type[torch.utils.data.Dataset]],
        network_kwargs: Dict = {},
        dataset_kwargs: Dict = {},
        validation_dataset_kwargs: Optional[Dict] = None,
        optim_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optim_kwargs: Dict = {"lr": 0.0001},
        schedulers_class: Dict = {},
        schedulers_kwargs: Dict[str, Dict] = {},
        processor_class: Optional[Type[Processor]] = None,
        processor_kwargs: Dict = {},
        checkpoint: Optional[str] = None,
        device: Union[str, torch.device] = "auto",
    ):
        # Initialize the _save_keys attribute using the superclass.
        # These are used for automatically identifying keys for saving/loading.
        super().__setattr__("_save_keys", set())
        super().__setattr__("_compiled", False)

        # Save relevant values
        self.env = env
        self.optim = {}

        # setup devices
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Setup the data preprocessor first. Thus, if we need to reference it in network setup we can.
        # Everything here is saved in self.processor
        self.setup_processor(processor_class, processor_kwargs)

        # Create the network.
        self.setup_network(network_class, network_kwargs)

        # Save values for optimizers, which will be lazily initialized later
        self.optim = {}
        self.optim_class = optim_class
        self.optim_kwargs = optim_kwargs

        # Save values for schedulers, which will be lazily initialized later
        self.schedulers = {}
        self.schedulers_class = schedulers_class
        self.schedulers_kwargs = schedulers_kwargs

        # Save values for datasets, which will be lazily initialized later
        self.dataset_class = dataset_class
        self.dataset_kwargs = dataset_kwargs
        self.validation_dataset_kwargs = validation_dataset_kwargs

        self._training = False

        # Load a check point if we have one -- using non-strict enforcement.
        if checkpoint is not None:
            self.load(checkpoint, strict=False)

    @property
    def training(self) -> bool:
        return self._training

    def __setattr__(self, name: str, value: Any) -> None:
        # Check to see if the value is a module etc.
        if hasattr(self, "_save_keys") and name in self._save_keys:
            pass
        elif isinstance(value, torch.nn.Parameter):
            self._save_keys.add(name)
        elif isinstance(value, torch.nn.Module) and sum(p.numel() for p in value.parameters()) > 0:
            self._save_keys.add(name)  # store if we have a module with more than zero parameters.
        return super().__setattr__(name, value)

    @property
    def save_keys(self) -> List[str]:
        return self._save_keys

    @property
    def compiled(self) -> bool:
        return self._compiled

    def to(self, device) -> "Algorithm":
        for k in self.save_keys:
            if k == "processor" and not self.processor.supports_gpu:
                continue
            else:
                setattr(self, k, getattr(self, k).to(device))
        return self

    def compile(self, **kwargs):
        for k in self.save_keys:
            attr = getattr(self, k)
            if isinstance(attr, torch.nn.Module):
                if type(attr).forward == torch.nn.Module.forward:
                    # In this case, the forward method hasn't been overriden.
                    # Thus we assume there is a compile argument.
                    assert hasattr(attr, "compile"), (
                        "save key " + k + " is nn.Module without forward() but didn't define `compile`."
                    )
                    attr.compile(**kwargs)
                else:
                    setattr(self, k, torch.compile(attr, **kwargs))
        # indicate that we have compiled the models.
        self._compiled = True

    def train(self) -> None:
        for k in self._save_keys:
            v = getattr(self, k)
            if isinstance(v, torch.nn.Module):
                v.train()
        self._training = True

    def eval(self) -> None:
        for k in self._save_keys:
            v = getattr(self, k)
            if isinstance(v, torch.nn.Module):
                v.eval()
        self._training = False

    @property
    def num_params(self):
        _num_params = 0
        for k in self.save_keys:
            attr = getattr(self, k)
            if hasattr(attr, "parameters"):
                _num_params += sum(p.numel() for p in attr.parameters() if p.requires_grad)
            else:
                assert isinstance(attr, torch.nn.Parameter), "Can only save Modules or Parameters."
                if attr.requires_grad:
                    _num_params += attr.numel()
        return _num_params

    @property
    def nbytes(self):
        # Returns the size of all the parameters in bytes
        _bytes = 0
        for k in self.save_keys:
            attr = getattr(self, k)
            if hasattr(attr, "parameters"):
                for p in attr.parameters():
                    _bytes += p.nelement() * p.element_size()
            if hasattr(attr, "buffers"):
                for b in attr.buffers():
                    _bytes += b.nelement() * b.element_size()
        return _bytes

    def setup_processor(self, processor_class: Optional[Type[Processor]], processor_kwargs: Dict) -> None:
        if processor_class is None:
            self.processor = IdentityProcessor(self.env.observation_space, self.env.action_space)
        else:
            self.processor = processor_class(self.env.observation_space, self.env.action_space, **processor_kwargs)

        if self.processor.supports_gpu:  # move it to device if it supports GPU computation.
            self.processor = self.processor.to(self.device)

    def setup_network(self, network_class: Type[torch.nn.Module], network_kwargs: Dict) -> None:
        self.network = network_class(
            self.processor.observation_space, self.processor.action_space, **network_kwargs
        ).to(self.device)

    def setup_optimizers(self) -> None:
        # Setup Optimizers
        assert len(self.optim) == 0, "setup_optimizers called twice!"
        for k in self.save_keys:
            attr = getattr(self, k)
            if hasattr(attr, "parameters"):
                parameters = attr.parameters()
            else:
                assert isinstance(attr, torch.nn.Parameter), "Can only save Modules or Parameters."
                parameters = [attr]
            # Constrcut the optimizer
            self.optim[k] = self.optim_class(parameters, **self.optim_kwargs)

    def setup_schedulers(self):
        assert len(self.schedulers) == 0, "setup_optimizers called twice!"
        for k in self.schedulers_class.keys():
            if self.schedulers_class[k] is not None:
                assert k in self.optim, "Did not find schedule key in optimizers dict."
                self.schedulers[k] = self.schedulers_class[k](self.optim[k], **self.schedulers_kwargs.get(k, dict()))

    def setup(self):
        """
        Called after everything else has been setup, right before training starts
        """
        pass

    def setup_train_dataset(self) -> None:
        """
        Setup the datasets. Note that this is called only during the learn method and thus doesn't take any arguments.
        Everything must be saved apriori. This is done to ensure that we don't need to load all of the data to load
        the model.
        """
        assert not hasattr(self, "dataset"), "Setup dataset called twice!"
        self.dataset = self.dataset_class(self.env.observation_space, self.env.action_space, **self.dataset_kwargs)

    def setup_validation_dataset(self) -> None:
        if self.validation_dataset_kwargs is not None:
            validation_dataset_kwargs = copy.deepcopy(self.dataset_kwargs)
            validation_dataset_kwargs.update(self.validation_dataset_kwargs)
            self.validation_dataset = self.dataset_class(
                self.env.observation_space, self.env.action_space, **validation_dataset_kwargs
            )
        else:
            self.validation_dataset = None

    def save(self, path: str, extension: str, metadata: Dict = {}) -> None:
        """
        Saves a checkpoint of the model and the optimizers
        """
        save_dict = {}
        if len(self.optim) > 0:
            save_dict["optim"] = {k: v.state_dict() for k, v in self.optim.items()}
        if len(self.schedulers) > 0:
            save_dict["schedulers"] = {k: v.state_dict() for k, v in self.schedulers.items()}
        for k in self._save_keys:
            attr = getattr(self, k)
            if hasattr(attr, "state_dict"):
                save_dict[k] = attr.state_dict()
            else:
                assert isinstance(attr, torch.nn.Parameter), "Can only save Modules or Parameters."
                save_dict[k] = attr

        # Add the metadata
        save_dict["metadata"] = metadata
        save_path = os.path.join(path, extension)
        if not save_path.endswith(".pt"):
            save_path += ".pt"
        torch.save(save_dict, save_path)

    def load(self, checkpoint: str, strict: bool = True) -> Dict:
        """
        Loads the model and its associated checkpoints.
        If we haven't created the optimizers and schedulers, do not load those.
        """
        print("[research] loading checkpoint:", checkpoint)
        checkpoint = torch.load(checkpoint, map_location=self.device)
        remaining_checkpoint_keys = set(checkpoint.keys())

        # First load everything except for the optim
        for k in self.save_keys:  # Loop through keys in the Algorithm.
            if k not in checkpoint:
                if strict:
                    raise ValueError("Checkpoint did not have key " + str(k))
                else:
                    print("[research] Warning: Checkpoint did not have key", k)
                    continue

            if isinstance(getattr(self, k), torch.nn.Parameter):
                # directly set the data, this is for nn.Parameters
                getattr(self, k).data = checkpoint[k].data
            else:
                # Otherwise, load via state dict
                getattr(self, k).load_state_dict(checkpoint[k], strict=strict)
            remaining_checkpoint_keys.remove(k)

        # Now load the optimizer and its associated keys
        for k in self.optim.keys():
            if strict and k not in checkpoint["optim"]:
                raise ValueError("Strict mode was enabled, but couldn't find optimizer key")
            elif k not in checkpoint["optim"]:
                print("[research] Warning: Checkpoint did not have optimizer key", k)
                continue
            self.optim[k].load_state_dict(checkpoint["optim"][k])
        if "optim" in checkpoint:
            remaining_checkpoint_keys.remove("optim")

        # Now load the schedulers
        for k in self.schedulers.keys():
            if strict and k not in checkpoint["schedulers"]:
                raise ValueError("Strict mode was enabled, but couldn't find scheduler key")
            elif k not in checkpoint["schedulers"]:
                print("[research] Warning: Checkpoint did not have scheduler key", k)
                continue
            self.schedulers[k].load_state_dict(checkpoint["schedulers"][k])
        if "schedulers" in checkpoint:
            remaining_checkpoint_keys.remove("schedulers")

        remaining_checkpoint_keys.remove("metadata")  # Do not count metadata key, which is always addded.
        if strict and len(remaining_checkpoint_keys) > 0:
            raise ValueError("Algorithm did not have keys ", +str(remaining_checkpoint_keys))
        elif len(remaining_checkpoint_keys) > 0:
            print("[research] Warning: Checkpoint keys", remaining_checkpoint_keys, "were not loaded.")

        return checkpoint["metadata"]

    def format_batch(self, batch: Any) -> Any:
        # Convert items to tensor if they are not.
        # Checking first makes sure we do not distrub memory pinning
        if not utils.contains_tensors(batch):
            batch = utils.to_tensor(batch)
        if self.processor.supports_gpu:
            # Move to CUDA first.
            batch = utils.to_device(batch, self.device)
            batch = self.processor(batch)
        else:
            batch = self.processor(batch)
            batch = utils.to_device(batch, self.device)
        return batch

    @abstractmethod
    def train_step(self, batch: Any, step: int, total_steps: int) -> Dict:
        """
        Train the model. Should return a dict of loggable values
        """
        return {}

    def validation_step(self, batch: Any) -> Dict:
        """
        perform a validation step. Should return a dict of loggable values.
        """
        raise NotImplementedError

    def train_extras(self, step: int, total_steps: int) -> Dict:
        """
        Perform any extra training operations. This is done before the train step is called.
        A common use case for this would be stepping the environment etc.
        """
        return {}

    def validation_extras(self, path: str, step: int) -> Dict:
        """
        Perform any extra validation operations.
        A common usecase for this is saving visualizations etc.
        """
        return {}

    def _predict(self, batch: Any, **kwargs) -> Any:
        """
        Internal prediction function, can be overridden
        By default, we call torch.no_grad(). If this behavior isn't desired,
        override the _predict funciton in your algorithm.
        """
        with torch.no_grad():
            if len(kwargs) > 0:
                raise ValueError("Default predict method does not accept key word args, but they were provided.")
            pred = self.network(batch)
        return pred

    def predict(self, batch: Any, is_batched: bool = False, **kwargs) -> Any:
        is_np = not utils.contains_tensors(batch)
        if not is_batched:
            # Unsqeeuze everything
            batch = utils.unsqueeze(batch, 0)
        batch = self.format_batch(batch)
        pred = self._predict(batch, **kwargs)
        if not is_batched:
            pred = utils.get_from_batch(pred, 0)
        if is_np:
            pred = utils.to_np(pred)
        return pred
