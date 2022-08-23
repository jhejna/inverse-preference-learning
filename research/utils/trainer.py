import os
import subprocess

import gym
import torch

import research

from . import schedules
from .config import Config
from .logger import Logger


def get_env(env, env_kwargs, wrapper, wrapper_kwargs):
    # Try to get the environment
    try:
        env = vars(research.envs)[env](**env_kwargs)
    except:
        env = gym.make(env, **env_kwargs)
    if wrapper is not None:
        env = vars(research.envs)[wrapper](env, **wrapper_kwargs)
    return env


def get_model(config, device="auto"):
    assert isinstance(config, Config)
    config = config.parse()  # Parse the config
    alg_class = vars(research.algs)[config["alg"]]
    dataset_class = None if config["dataset"] is None else vars(research.datasets)[config["dataset"]]
    network_class = None if config["network"] is None else vars(research.networks)[config["network"]]
    optim_class = None if config["optim"] is None else vars(torch.optim)[config["optim"]]
    processor_class = None if config["processor"] is None else vars(research.processors)[config["processor"]]

    # TODO: Construct the train env as a vector env
    """
    if config['parallel_envs']:
        env = research.envs.SubprocVecEnv(config['env'], config['env_kwargs'], config['wrapper'],
                                          config['wrapper_kwargs'], **config['vec_kwargs'])
    else:
        env = research.envs.DummyVecEnv(config['env'], config['env_kwargs'], config['wrapper'],
                                        config['wrapper_kwargs'], **config['vec_kwargs'])
    """
    env = (
        None
        if config["env"] is None
        else get_env(config["env"], config["env_kwargs"], config["wrapper"], config["wrapper_kwargs"])
    )

    # Construct the eval env. Note that wrappers are assumed to be shared.
    eval_env = (
        None
        if config["eval_env"] is None
        else get_env(config["eval_env"], config["eval_env_kwargs"], config["wrapper"], config["wrapper_kwargs"])
    )

    algo = alg_class(
        env,
        network_class,
        dataset_class,
        network_kwargs=config["network_kwargs"],
        dataset_kwargs=config["dataset_kwargs"],
        validation_dataset_kwargs=config["validation_dataset_kwargs"],
        device=device,
        processor_class=processor_class,
        processor_kwargs=config["processor_kwargs"],
        optim_class=optim_class,
        optim_kwargs=config["optim_kwargs"],
        collate_fn=config["collate_fn"],
        batch_size=config["batch_size"],
        checkpoint=config["checkpoint"],
        eval_env=eval_env,
        **config["alg_kwargs"],
    )

    return algo


def train(config, path, device="auto"):
    # Create the save path and save the config
    print("[research] Training agent with config:")
    print(config)
    os.makedirs(path, exist_ok=False)
    print("[research] Model Directory:", path)

    config.save(path)
    # save the git hash
    process = subprocess.Popen(["git", "rev-parse", "HEAD"], shell=False, stdout=subprocess.PIPE)
    git_head_hash = process.communicate()[0].strip()
    with open(os.path.join(path, "git_hash.txt"), "wb") as f:
        f.write(git_head_hash)

    # Setup wandb here if we have it configured in setup_shell
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if wandb_api_key is not None and wandb_api_key != "":
        # We have wandb, get the project name and initialize
        import wandb

        project_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        wandb.init(
            project=os.path.basename(project_dir),
            name=os.path.basename(path),
            config=config.flatten(separator="-"),
            dir=os.path.join(os.path.dirname(project_dir), "wandb"),
        )
        use_wandb = True
    else:
        use_wandb = False

    model = get_model(config, device=device)
    assert issubclass(type(model), research.algs.base.Algorithm)
    print("[research] Using device", model.device)

    if config["seed"] is not None:
        # Seed the model if provided.
        model.seed(config["seed"])

    # Fetch the scheduler
    schedule = None if config["schedule"] is None else vars(schedules)[config["schedule"]]

    print(
        "[research] Training a model with",
        sum(p.numel() for p in model.network.parameters() if p.requires_grad),
        "trainable parameters.",
    )

    model.train(
        path,
        schedule=schedule,
        schedule_kwargs=config["schedule_kwargs"],
        use_wandb=use_wandb,
        **config["train_kwargs"],
    )

    print("[research] finished training for", model.steps, "steps.")
    return model


def load(config, model_path, device="auto", strict=True):
    model = get_model(config, device=device)
    model.load(model_path, strict=strict)
    return model


def load_from_path(checkpoint_path, device="auto", strict=True):
    config_path = os.path.join(os.path.dirname(checkpoint_path), "config.yaml")
    config = Config.load(config_path)
    return load(config, checkpoint_path, device=device, strict=strict)
