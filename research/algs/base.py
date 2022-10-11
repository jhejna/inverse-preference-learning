import copy
import os
import random
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, Optional, Type, Union

import gym
import numpy as np
import torch

from research.processors.base import IdentityProcessor, Processor
from research.utils import evaluate, utils
from research.utils.logger import Logger

MAX_VALID_METRICS = {"reward", "accuracy", "success", "is_success"}


def log_from_dict(logger: Logger, metric_lists: Dict, prefix: str) -> None:
    keys_to_remove = []
    for metric_name, metric_value in metric_lists.items():
        if isinstance(metric_value, list) and len(metric_value) > 0:
            logger.record(prefix + "/" + metric_name, np.mean(metric_value))
            keys_to_remove.append(metric_name)
        else:
            logger.record(prefix + "/" + metric_name, metric_value)
            keys_to_remove.append(metric_name)
    for key in keys_to_remove:
        del metric_lists[key]


def _worker_init_fn(worker_id: int) -> None:
    seed = torch.utils.data.get_worker_info().seed
    seed = seed % (2**32 - 1)  # Reduce to valid 32bit unsigned range
    np.random.seed(seed)
    random.seed(seed)


class Algorithm(ABC):
    def __init__(
        self,
        env: gym.Env,
        network_class: Type[torch.nn.Module],
        dataset_class: Union[Type[torch.utils.data.IterableDataset], Type[torch.utils.data.Dataset]],
        network_kwargs: Dict = {},
        dataset_kwargs: Dict = {},
        device: Union[str, torch.device] = "auto",
        optim_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optim_kwargs: Dict = {"lr": 0.0001},
        processor_class: Optional[Type[Processor]] = None,
        processor_kwargs: Dict = {},
        checkpoint: Optional[str] = None,
        validation_dataset_kwargs: Optional[Dict] = None,
        collate_fn: Optional[Any] = None,
        batch_size: Optional[int] = 64,
        eval_env: Optional[gym.Env] = None,
    ):
        # Save relevant values
        self.env = env
        self.eval_env = eval_env

        self.dataset_class = dataset_class
        self.dataset_kwargs = dataset_kwargs
        self.validation_dataset_kwargs = validation_dataset_kwargs
        self.collate_fn = collate_fn
        self.batch_size = batch_size

        # setup devices
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Setup the data preprocessor first. Thus, if we need to reference it in
        # network setup we can.
        self.setup_processor(processor_class, processor_kwargs)

        # Create the network
        self.setup_network(network_class, network_kwargs)

        # Create the optimizers
        self.optim = {}
        self.setup_optimizers(optim_class, optim_kwargs)

        # Load a check point if we have one
        if checkpoint:
            try:
                self.load(checkpoint, strict=True)
            except:
                print("[research] Warning: Loading in non-strict mode, thus optimizer may be skipped.")
                self.load(checkpoint, strict=False)

    @property
    def action_space(self) -> gym.Space:
        if self.env is not None:
            return self.env.action_space
        elif self.eval_env is not None:
            return self.eval_env.action_space
        else:
            raise ValueError("Neither env or eval_env was provided.")

    @property
    def observation_space(self) -> gym.Space:
        if self.env is not None:
            return self.env.observation_space
        elif self.eval_env is not None:
            return self.eval_env.observation_space
        else:
            raise ValueError("Neither env or eval_env was provided.")

    def setup_processor(self, processor_class: Optional[Type[Processor]], processor_kwargs: Dict) -> None:
        if processor_class is None:
            self.processor = IdentityProcessor(self.observation_space, self.action_space)
        else:
            self.processor = processor_class(self.observation_space, self.action_space, **processor_kwargs)

        if self.processor.supports_gpu:  # move it to device if it supports GPU computation.
            self.processor = self.processor.to(self.device)

    def setup_network(self, network_class: Type[torch.nn.Module], network_kwargs: Dict) -> None:
        self.network = network_class(
            self.processor.observation_space, self.processor.action_space, **network_kwargs
        ).to(self.device)

    def setup_optimizers(self, optim_class: Type[torch.optim.Optimizer], optim_kwargs: Dict) -> None:
        # Default optimizer initialization
        self.optim["network"] = optim_class(self.network.parameters(), **optim_kwargs)

    def setup_datasets(self) -> None:
        """
        Setup the datasets. Note that this is called only during the learn method and thus doesn't take any arguments.
        Everything must be saved apriori. This is done to ensure that we don't need to load all of the data to load
        the model.
        """
        self.dataset = self.dataset_class(self.observation_space, self.action_space, **self.dataset_kwargs)
        if self.validation_dataset_kwargs is not None:
            validation_dataset_kwargs = copy.deepcopy(self.dataset_kwargs)
            validation_dataset_kwargs.update(self.validation_dataset_kwargs)
            self.validation_dataset = self.dataset_class(
                self.observation_space, self.action_space, **validation_dataset_kwargs
            )
        else:
            self.validation_dataset = None

    def save(self, path: str, extension: str) -> None:
        """
        Saves a checkpoint of the model and the optimizers
        """
        optim = {k: v.state_dict() for k, v in self.optim.items()}
        save_dict = {"network": self.network.state_dict(), "optim": optim, "processor": self.processor.state_dict()}
        save_dict.update(self._save_extras())
        torch.save(save_dict, os.path.join(path, extension + ".pt"))

    def _save_extras(self) -> None:
        """
        override this method to return any extra values or tensors that should be saved
        """
        return {}

    def load(self, checkpoint: str, initial_lr: Optional[float] = None, strict: bool = True) -> None:
        """
        Loads the model and its associated checkpoints.
        """
        print("[research] loading checkpoint:", checkpoint)
        checkpoint = torch.load(checkpoint, map_location=self.device)
        self.network.load_state_dict(checkpoint["network"], strict=strict)
        self.processor.load_state_dict(checkpoint["processor"], strict=strict)

        for k, v in self.optim.items():
            if strict and k not in checkpoint["optim"]:
                raise ValueError("Strict mode was enabled, but couldn't find optimizer key")
            elif k not in checkpoint["optim"]:
                continue

            try:
                self.optim[k].load_state_dict(checkpoint["optim"][k])
            except ValueError as e:
                if strict:
                    raise e
                else:
                    continue

        # make sure that we reset the learning rate in case we decide to not use scheduling for finetuning.
        if initial_lr is not None:
            for param_group in self.optim.param_groups:
                param_group["lr"] = initial_lr

        self._load_extras(checkpoint, strict=strict)

    def _load_extras(self, checkpoint: Any, strict: bool = True) -> None:
        """
        override this method to load any extra values or tensors that were saved
        """
        return

    def seed(self, seed: int) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if self.env is not None:
            self.env.seed(seed)
        if self.eval_env is not None:
            self.eval_env.seed(seed)

    @property
    def steps(self) -> int:
        return self._steps

    @property
    def epochs(self) -> int:
        return self._epochs

    @property
    def total_steps(self) -> int:
        if hasattr(self, "_total_steps"):
            return self._total_steps
        else:
            raise ValueError("alg.train has not been called, no total step count available.")

    def _format_batch(self, batch: Any) -> Any:
        # Convert items to tensor if they are not.
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

    def train(
        self,
        path: str,
        total_steps: int,
        schedule: Optional[Any] = None,
        schedule_kwargs: Dict = {},
        log_freq: int = 100,
        eval_freq: int = 1000,
        max_eval_steps: int = -1,
        workers: Optional[int] = 4,
        loss_metric: Optional[str] = "loss",
        profile_freq: int = -1,
        use_wandb: bool = False,
        x_axis: str = "steps",
        eval_fn: Optional[Any] = None,
        eval_kwargs: Dict = {},
    ) -> None:
        writers = ["tb", "csv"]
        if use_wandb:
            writers.append("wandb")
        logger = Logger(path=path, writers=writers)

        # Construct the dataloaders.
        self.setup_datasets()
        shuffle = not issubclass(self.dataset_class, torch.utils.data.IterableDataset)
        pin_memory = self.device.type == "cuda"
        worker_init_fn = _worker_init_fn if workers > 0 else None
        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=workers,
            worker_init_fn=worker_init_fn,
            pin_memory=pin_memory,
            collate_fn=self.collate_fn,
        )
        if self.validation_dataset is not None:
            validation_dataloader = torch.utils.data.DataLoader(
                self.validation_dataset,
                batch_size=self.batch_size,
                shuffle=shuffle,
                num_workers=0,
                pin_memory=False,  # Not for the validation dataset
                collate_fn=self.collate_fn,
            )
        else:
            validation_dataloader = None

        # Create schedulers for the optimizers
        schedulers = {}
        for k in schedule.keys():
            if schedule[k] is not None:
                assert k in self.optim, "Did not find schedule key in optimizers dict."
                schedulers[k] = schedule[k](self.optim[k], **schedule_kwargs.get(k, dict()))

        # Grab the correct evaluation function
        eval_fn = None if eval_fn is None else vars(evaluate)[eval_fn]

        # Setup model metrics.
        self._steps = 0
        self._epochs = 0
        self._total_steps = total_steps
        current_step = 0
        train_metric_lists = defaultdict(list)
        best_validation_metric = -1 * float("inf") if loss_metric in MAX_VALID_METRICS else float("inf")
        last_train_log = -log_freq  # Ensure that we log on the first step
        last_validation_log = -eval_freq  # Ensure that we log on the first step

        # Setup training
        self._setup_train()
        self.network.train()

        # Setup profiling immediately before we start the loop.
        start_time = current_time = time.time()
        profiling_metric_lists = defaultdict(list)

        while current_step <= total_steps:
            for batch in dataloader:
                # Profiling
                if profile_freq > 0 and self._steps % profile_freq == 0:
                    stop_time = time.time()
                    profiling_metric_lists["dataset"].append(stop_time - current_time)
                    current_time = stop_time

                batch = self._format_batch(batch)

                if profile_freq > 0 and self._steps % profile_freq == 0:
                    stop_time = time.time()
                    profiling_metric_lists["preprocess"].append(stop_time - current_time)
                    current_time = stop_time

                # Train the network
                assert self.network.training, "Network was not in training mode and trainstep was called."
                train_metrics = self._train_step(batch)
                for metric_name, metric_value in train_metrics.items():
                    train_metric_lists[metric_name].append(metric_value)

                if profile_freq > 0 and self._steps % profile_freq == 0:
                    stop_time = time.time()
                    profiling_metric_lists["train_step"].append(stop_time - current_time)

                # Update the schedulers
                for scheduler in schedulers.values():
                    scheduler.step()

                # Compute the current step. This is so we can use other metrics
                if x_axis in train_metrics:
                    current_step = train_metrics[x_axis]
                elif x_axis == "epoch":
                    current_step = self.epochs
                else:
                    current_step = self._steps

                if (current_step - last_train_log) >= log_freq:
                    # Timing metrics
                    current_time = time.time()
                    logger.record("time/steps", self._steps)
                    logger.record("time/epochs", self._epochs)
                    logger.record(
                        "time/steps_per_second", (current_step - last_train_log) / (current_time - start_time)
                    )
                    start_time = current_time
                    # Record Other metrics
                    for name, scheduler in schedulers.items():
                        logger.record("lr/" + name, scheduler.get_last_lr()[0])
                    log_from_dict(logger, profiling_metric_lists, "time")
                    log_from_dict(logger, train_metric_lists, "train")
                    logger.dump(step=current_step)
                    last_train_log = current_step

                if (current_step - last_validation_log) >= eval_freq:
                    self.eval_mode()
                    current_validation_metric = None
                    if validation_dataloader is not None:
                        eval_steps = 0
                        validation_metric_lists = defaultdict(list)
                        for batch in validation_dataloader:
                            batch = self._format_batch(batch)
                            losses = self._validation_step(batch)
                            for metric_name, metric_value in losses.items():
                                validation_metric_lists[metric_name].append(metric_value)
                            eval_steps += 1
                            if eval_steps == max_eval_steps:
                                break

                        if loss_metric in validation_metric_lists:
                            current_validation_metric = np.mean(validation_metric_lists[loss_metric])
                        log_from_dict(logger, validation_metric_lists, "valid")

                    # Now run any extra validation steps, independent of the validation dataset.
                    validation_extras = self._validation_extras(path, self._steps, validation_dataloader)
                    if loss_metric in validation_extras:
                        current_validation_metric = validation_extras[loss_metric]
                    log_from_dict(logger, validation_extras, "valid")

                    # Evaluation episodes
                    if self.eval_env is not None and eval_fn is not None:
                        eval_metrics = eval_fn(self.eval_env, self, **eval_kwargs)
                        if loss_metric in eval_metrics:
                            current_validation_metric = eval_metrics[loss_metric]
                        log_from_dict(logger, eval_metrics, "eval")

                    if current_validation_metric is None:
                        pass
                    elif loss_metric in MAX_VALID_METRICS and current_validation_metric > best_validation_metric:
                        self.save(path, "best_model")
                        best_validation_metric = current_validation_metric
                    elif current_validation_metric < best_validation_metric:
                        self.save(path, "best_model")
                        best_validation_metric = current_validation_metric

                    # Eval Logger Dump to CSV
                    logger.dump(step=current_step, eval=True)  # Mark True on the eval flag
                    last_validation_log = current_step
                    self.save(path, "final_model")  # Also save the final model every eval period.
                    self.train_mode()

                # Increment the number of training steps.
                # Technically we have done one more, but this keeps logs at uniform positions.
                self._steps += 1
                if current_step >= total_steps:
                    break

                # Profiling
                if profile_freq > 0 and self._steps % profile_freq == 0:
                    current_time = time.time()

            self._epochs += 1
        logger.close()

    @abstractmethod
    def _train_step(self, batch: Any) -> Dict:
        """
        Train the model. Should return a dict of loggable values
        """
        pass

    @abstractmethod
    def _validation_step(self, batch: Any) -> Dict:
        """
        perform a validation step. Should return a dict of loggable values.
        """
        pass

    def _setup_train(self) -> None:
        """
        Does nothing by default. Is called prior to running the training loop.
        """
        pass

    def _validation_extras(self, path: str, step: int, dataloader: Optional[torch.utils.data.DataLoader]) -> Dict:
        """
        perform any extra validation operations
        """
        return {}

    def train_mode(self) -> None:
        self.network.train()
        self.processor.train()

    def eval_mode(self) -> None:
        self.network.eval()
        self.processor.eval()

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
        batch = self._format_batch(batch)
        pred = self._predict(batch, **kwargs)
        if not is_batched:
            pred = utils.get_from_batch(pred, 0)
        if is_np:
            pred = utils.to_np(pred)
        return pred
