import research
from research.datasets import ReplayBuffer
from research.utils.trainer import load, load_from_path
import gym
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--num-ep", "-n", type=int, default=1)
parser.add_argument("--random-ep", "-r", type=int, default=0)
parser.add_argument("--path", "-p", type=str, required=True)
parser.add_argument("--model", "-m", type=str, required=True)
parser.add_argument("--vision", action='store_true', default=False)
parser.add_argument("--goal", "-g", type=float, nargs='+')
parser.add_argument("--env", type=str, choices=['point_mass', 'reacher'], required=True)
args = parser.parse_args()
assert len(args.goal) == 2

os.makedirs(args.path, exist_ok=False)
if os.path.isdir(args.model):
    models = [os.path.join(args.model, p, "best_model.pt") for p in os.listdir(args.model)]
else:
    models = [args.model]

model = load_from_path(models.pop(0))
capacity = args.num_ep*502 if args.vision else args.num_ep*1002
dataset = ReplayBuffer(model.env.observation_space, model.env.action_space, capacity=*args.num_ep, cleanup=False) # Make sure we save all the episodes.
env_id = {
    'point_mass': "Goal_point_massReach_l2",
    'reacher': "Goal_reacherReach_l2",
}[args.env]
if args.vision:
    env_id += '-vision'
env_id += '-v0'

env = gym.make(env_id, task_kwargs=dict(target_pos=args.goal))

def collect_ep(env, dataset, predict_fn):
    obs = env.reset()
    done = False
    episode_length = 0
    dataset.add(obs)
    while not done:
        action = predict_fn(obs)
        obs, reward, done, info = env.step(action)
        # Determine the discount factor.
        if 'discount' in info:
            discount = info['discount']
        elif hasattr(env, "_max_episode_stes") and episode_length == env._max_episode_steps:
            discount = 1.0
        else:
            discount = 1 - float(done)
        dataset.add(obs, action, reward, done, discount)

while True:
    for _ in range(args.num_ep):
        collect_ep(env, dataset, model.predict)
    if len(models) == 0:
        break
    else:
        model = load_from_path(models.pop(0))

for _ in range(args.random_ep):
    collect_ep(env, dataset, lambda x: env.action_space.sample())

dataset.save(args.path)
