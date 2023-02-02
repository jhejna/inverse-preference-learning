import argparse
import os

from research.utils.config import Config


def try_wandb_setup(path, config):
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if wandb_api_key is not None and wandb_api_key != "":
        try:
            import wandb
        except:
            return
        project_dir = os.path.dirname(os.path.dirname(__file__))
        wandb.init(
            project=os.path.basename(project_dir),
            name=os.path.basename(path),
            config=config.flatten(separator="-"),
            dir=os.path.join(os.path.dirname(project_dir), "wandb"),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, default=None)
    parser.add_argument("--path", "-p", type=str, default=None)
    parser.add_argument("--device", "-d", type=str, default="auto")
    args = parser.parse_args()

    config = Config.load(args.config)
    try_wandb_setup(args.path, config)
    config = config.parse()
    model = config.get_model(device=args.device)
    trainer = config.get_trainer()
    trainer.set_model(model)
    trainer.train(args.path)
