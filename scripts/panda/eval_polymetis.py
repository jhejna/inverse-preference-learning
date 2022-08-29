import argparse
import os

from research.utils.config import Config
from research.utils.evaluate import eval_policy
from research.utils.trainer import load

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, help="path to the policy")
parser.add_argument("--num-ep", type=int, default=10)
parser.add_argument("--ip", type=str, default="localhost")

args = parser.parse_args()

# Load the config
config = Config.load(args.path)
new_env = {
    "PyBulletPandaReach-v0": "PolyMetisReach-v0",
    "RobosuitePandaReach-v0": "PolyMetisReach-v0",
    "PyBulletPandaBlockPush-v0": "PolyMetisBlockPush-v0",
    "PyBulletPandaBlockPushFix-v0": "PolyMetisBlockPushFix-v0",
    "RobosuitePandaBlockPush-v0": "PolyMetisBlockPush-v0",
}[config["env"]]
assert "goal" in config["env_kwargs"]

config["env"] = new_env
config["env_kwargs"]["ip_address"] = args.ip
# del config['env_kwargs']['gripper']
config["checkpoint"] = None  # Remove the checkpoint

model = load(config, os.path.join(args.path, "best_model.pt"), strict=False)
eval_policy(model.env, model, args.num_ep)
