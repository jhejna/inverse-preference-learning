import os
import random
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Union

import gym
import numpy as np
import torch

from research.algs.base import Algorithm

from . import evaluate
from .logger import Logger

MAX_VALID_METRICS = {"reward", "accuracy", "success", "is_success"}


def log_from_dict(logger: Logger, metric_lists: Dict[str, Union[List, float]], prefix: str) -> None:
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


def log_wrapper(fn: Callable, metric_lists: Dict[str, List]):
    def wrapped_fn(*args, **kwargs):
        metrics = fn(*args, **kwargs)
        for name, value in metrics.items():
            metric_lists[name].append(value)

    return wrapped_fn


def time_wrapper(fn: Callable, name: str, profile_lists: Dict[str, List]):
    def wrapped_fn(*args, timeit=False, **kwargs):
        if timeit:
            start_time = time.time()
            output = fn(*args, **kwargs)
            end_time = time.time()
            profile_lists[name].append(end_time - start_time)
        else:
            output = fn(*args, **kwargs)
        return output

    return wrapped_fn


def _worker_init_fn(worker_id: int) -> None:
    seed = torch.utils.data.get_worker_info().seed
    seed = seed % (2**32 - 1)  # Reduce to valid 32bit unsigned range
    np.random.seed(seed)
    random.seed(seed)


class Trainer(object):
    def __init__(
        self,
        eval_env: Optional[gym.Env] = None,
        total_steps: int = 1000,
        log_freq: int = 100,
        eval_freq: int = 1000,
        profile_freq: int = -1,
        max_eval_steps: Optional[int] = None,
        loss_metric: Optional[str] = "loss",
        x_axis: str = "steps",
        benchmark: bool = False,
        subproc_eval: bool = False,
        torch_compile: bool = False,
        torch_compile_kwargs: Dict = {},
        eval_fn: Optional[Any] = None,
        eval_kwargs: Dict = {},
        train_dataloader_kwargs: Dict = {},
        validation_dataloader_kwargs: Dict = {},
    ) -> None:
        self._model = None
        self.eval_env = eval_env

        # Logging parameters
        self.total_steps = total_steps
        self.log_freq = log_freq
        self.eval_freq = eval_freq
        self.profile_freq = profile_freq
        self.max_eval_steps = max_eval_steps
        self.loss_metric = loss_metric
        self.x_axis = x_axis

        # Performance parameters
        self.benchmark = benchmark
        assert subproc_eval == False, "Subproc eval not yet supported"
        assert torch_compile == False, "Torch Compile currently exhibits bugs. Do not use."
        self.torch_compile = torch_compile
        self.torch_compile_kwargs = torch_compile_kwargs

        # Eval parameters
        self.eval_fn = eval_fn
        self.eval_kwargs = eval_kwargs

        # Dataloader parameters
        self._train_dataloader = None
        self.train_dataloader_kwargs = train_dataloader_kwargs
        self._validation_dataloader = None
        self.validation_dataloader_kwargs = validation_dataloader_kwargs
        self._validation_iterator = None

    def set_model(self, model: Algorithm):
        assert self._model is None, "Model has already been set."
        self._model = model

    @property
    def model(self):
        if self._model is None:
            raise ValueError("Model has not yet been set! use `set_model` before calling trainer functionality.")
        return self._model

    @property
    def train_dataloader(self) -> torch.utils.data.DataLoader:
        if not hasattr(self.model, "dataset"):
            self.model.setup_train_dataset()
        if self.model.dataset is None:
            return None
        if self._train_dataloader is None:
            shuffle = not isinstance(self.model.dataset, torch.utils.data.IterableDataset)
            pin_memory = self.model.device.type == "cuda"
            self._train_dataloader = torch.utils.data.DataLoader(
                self.model.dataset,
                shuffle=shuffle,
                pin_memory=pin_memory,
                worker_init_fn=_worker_init_fn,
                **self.train_dataloader_kwargs,
            )
        return self._train_dataloader

    @property
    def validation_dataloader(self) -> torch.utils.data.DataLoader:
        if not hasattr(self.model, "validation_dataset"):
            self.model.setup_validation_dataset()
        if self.model.validation_dataset is None:
            return None
        if self._validation_dataloader is None:
            kwargs = self.train_dataloader_kwargs.copy()
            kwargs.update(self.validation_dataloader_kwargs)
            shuffle = not isinstance(self.model.validation_dataset, torch.utils.data.IterableDataset)
            pin_memory = self.model.device.type == "cuda"
            self._validation_dataloader = torch.utils.data.DataLoader(
                self.model.dataset, shuffle=shuffle, pin_memory=pin_memory, worker_init_fn=_worker_init_fn, **kwargs
            )
        return self._validation_dataloader

    def check_compilation(self):
        # If the model has not been compiled, compile it.
        if not self.model.compiled and self.torch_compile:
            self.model.compile(**self.torch_compile_kwargs)

    def train(self, path: str):
        # Prepare the model for training by initializing the optimizers and the schedulers
        self.model.setup_optimizers()
        self.check_compilation()
        self.model.setup_schedulers()
        self.model.setup()  # perform any other arbitrary setup needs.
        print("[research] Training a model with", self.model.num_params, "trainable parameters.")
        print("[research] Estimated size: {:.2f} GB".format(self.model.nbytes / 1024**3))

        # First, we should detect if the path already contains a model and a checkpoint
        if os.path.exists(os.path.join(path, "final_model.pt")):
            # If so, we can finetune from that initial checkpoint. When we do this we should load strictly.
            # If we can't load it, we should immediately throw an error.
            metadata = self.model.load(os.path.join(path, "final_model.pt"), strict=True)
            current_step, steps, epochs = metadata["current_step"], metadata["steps"], metadata["epochs"]
            # Try to load the xaxis value if we need to.
        else:
            current_step, steps, epochs = 0, 0, 0

        # Setup benchmarking.
        if self.benchmark:
            torch.backends.cudnn.benchmark = True

        # Setup the Logger
        writers = ["tb", "csv"]
        try:
            # Detect if wandb has been setup. If so, log it.
            import wandb

            if wandb.run is not None:
                writers.append("wandb")
        except:
            pass

        logger = Logger(path=path, writers=writers)

        # Construct all of the metric lists to be used during training
        # Construct all the metric lists to be used during training
        train_metric_lists = defaultdict(list)
        extras_metric_lists = defaultdict(list)
        profiling_metric_lists = defaultdict(list)
        # Wrap the functions we use in logging and profile wrappers
        train_step = log_wrapper(self.model.train_step, train_metric_lists)
        train_step = time_wrapper(train_step, "train_step", profiling_metric_lists)
        extras_step = log_wrapper(self.model.train_extras, extras_metric_lists)
        extras_step = time_wrapper(extras_step, "extras_step", profiling_metric_lists)
        format_batch = time_wrapper(self.model.format_batch, "processor", profiling_metric_lists)

        # Compute validation trackers
        using_max_valid_metric = self.loss_metric in MAX_VALID_METRICS
        best_valid_metric = -1 * float("inf") if using_max_valid_metric else float("inf")

        # Compute logging frequencies
        last_train_log = -self.log_freq  # Ensure that we log on the first step
        last_validation_log = (
            0 if self.benchmark else -self.eval_freq
        )  # Ensure that we log the first step, except if we are benchmarking.

        profile = True if self.profile_freq > 0 else False  # must profile to get all keys for csv log
        self.model.train()

        start_time = time.time()
        current_time = start_time

        while current_step <= self.total_steps:
            for batch in self.train_dataloader:
                if profile:
                    profiling_metric_lists["dataset"].append(time.time() - current_time)

                # Run any pre-train steps, like stepping the enviornment or training auxiliary networks.
                # Realistically this is just going to be used for environment stepping, but hey! Good to have.
                extras_step(current_step, self.total_steps, timeit=profile)

                # Next, format the batch
                batch = format_batch(batch, timeit=profile)

                # Run the train step
                train_step(batch, current_step, self.total_steps, timeit=profile)

                # Update the schedulers
                for scheduler in self.model.schedulers.values():
                    scheduler.step()

                steps += 1
                if self.x_axis == "steps":
                    new_current_step = steps + 1
                elif self.x_axis == "epoch":
                    new_current_step = epochs
                elif self.x_axis in train_metric_lists:
                    new_current_step = train_metric_lists[self.x_axis][-1]  # Get the most recent value
                elif self.x_axis in extras_metric_lists:
                    new_current_step = extras_metric_lists[self.x_axis][-1]  # Get the most recent value
                else:
                    raise ValueError("Could not find train value for x_axis " + str(self.x_axis))

                # Now determine if we should dump the logs
                if (current_step - last_train_log) >= self.log_freq:
                    # Record timing metrics
                    current_time = time.time()
                    logger.record("time/steps", steps)
                    logger.record("time/epochs", epochs)
                    logger.record(
                        "time/steps_per_second", (current_step - last_train_log) / (current_time - start_time)
                    )
                    log_from_dict(logger, profiling_metric_lists, "time")
                    start_time = current_time
                    # Record learning rates
                    for name, scheduler in self.model.schedulers.items():
                        logger.record("lr/" + name, scheduler.get_last_lr()[0])
                    # Record training metrics
                    log_from_dict(logger, extras_metric_lists, "train_extras")
                    log_from_dict(logger, train_metric_lists, "train")
                    logger.dump(step=current_step)
                    # Update the last time we logged.
                    last_train_log = current_step

                if (current_step - last_validation_log) >= self.eval_freq:
                    self.model.eval()
                    current_valid_metric = None
                    model_metadata = dict(current_step=current_step, epochs=epochs, steps=steps)

                    # Run and time validation step
                    current_time = time.time()
                    validation_metrics = self.validate(path, current_step)
                    logger.record("time/validation", time.time() - current_time)
                    if self.loss_metric in validation_metrics:
                        current_valid_metric = validation_metrics[self.loss_metric]
                    log_from_dict(logger, validation_metrics, "validation")

                    # Run and time eval step
                    current_time = time.time()
                    eval_metrics = self.evaluate(path, current_step)
                    logger.record("time/eval", time.time() - current_time)
                    if self.loss_metric in eval_metrics:
                        current_valid_metric = eval_metrics[self.loss_metric]
                    log_from_dict(logger, eval_metrics, "eval")

                    # Determine if we have a new best self.model.
                    if current_valid_metric is None:
                        pass
                    elif (using_max_valid_metric and current_valid_metric > best_valid_metric) or (
                        not using_max_valid_metric and current_valid_metric < best_valid_metric
                    ):
                        best_valid_metric = current_valid_metric
                        self.model.save(path, "best_model", model_metadata)

                    # Eval Logger dump to CSV
                    logger.dump(step=current_step, eval=True)  # Mark True on the eval flag
                    last_validation_log = current_step
                    self.model.save(path, "final_model", model_metadata)  # Also save the final model every eval period.

                    # Put the model back in train mode.
                    self.model.train()
                    last_validation_log = current_step

                current_step = new_current_step  # Update the current step
                if current_step >= self.total_steps:
                    break  # We need to break in the middle of an epoch.

                profile = self.profile_freq > 0 and steps % self.profile_freq == 0
                if profile:
                    current_time = time.time()  # update current time only, not start time

            epochs += 1

    def validate(self, path: str, step: int):
        assert not self.model.training
        self.check_compilation()
        # Setup the dataset
        validation_metrics = {}
        if self.validation_dataloader is not None:
            eval_steps = 0
            validation_metric_lists = defaultdict(list)
            validation_step = log_wrapper(self.model.validation_step, validation_metric_lists)
            # Get the iterator or continue from where we just left off.
            if self._validation_iterator is None:
                self._validation_iterator = iter(self.validation_dataloader)
            while True:
                try:
                    batch = next(self._validation_iterator)
                except StopIteration:
                    if self.max_eval_steps is None:
                        self._validation_iterator = None  # Set to None for next validation.
                        break
                    else:
                        self._validation_iterator = iter(self.validation_dataloader)
                        batch = next(self._validation_iterator)
                batch = self.model.format_batch(batch)
                validation_step(batch)
                eval_steps += 1
                if eval_steps == self.max_eval_steps:
                    break
            # Return the the average metrics.
            for k, v in validation_metric_lists.items():
                validation_metrics[k] = np.mean(v)
        # Update with any extras.
        validation_metrics.update(self.model.validation_extras(path, step))
        return validation_metrics

    def evaluate(self, path: str, step: int):
        assert not self.model.training
        self.check_compilation()
        eval_fn = None if self.eval_fn is None else vars(evaluate)[self.eval_fn]
        if eval_fn is None:
            return dict()
        eval_metrics = eval_fn(self.eval_env, self.model, path, step, **self.eval_kwargs)
        return eval_metrics
