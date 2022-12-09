# Register environment classes here
from .base import Empty

# If we want to register environments in gym.
# These will be loaded when we import the research package.
from gym.envs import register


# Register the DM Control environments.
from dm_control import suite

# Custom DM Control domains can be registered as follows:
# from . import <custom dm_env module>
# assert hasattr(<custom dm_env module>, 'SUITE')
# suite._DOMAINS['<custom dm_env module>'] = <custom dm_env module>

from metaworld.envs.mujoco.env_dict import ALL_V2_ENVIRONMENTS


from . import point_mass
from . import reacher

assert hasattr(point_mass, "SUITE")
suite._DOMAINS["goal_point_mass"] = point_mass


assert hasattr(reacher, "SUITE")
suite._DOMAINS["goal_reacher"] = reacher

# Register all of the DM control tasks
for domain_name, task_name in suite._get_tasks(tag=None):
    # Import state domains
    ID = f"{domain_name.capitalize()}{task_name.capitalize()}-v0"
    register(
        id=ID,
        entry_point="research.envs.dm_control:DMControlEnv",
        kwargs={
            "domain_name": domain_name,
            "task_name": task_name,
            "action_minimum": -1.0,
            "action_maximum": 1.0,
            "action_repeat": 1,
            "from_pixels": False,
            "flatten": True,
            "stack": 1,
        },
    )

# Add the meta world test environments.
# For each one, register the different tasks.

for env_name, env_cls in ALL_V2_ENVIRONMENTS.items():
    ID = f"mw_{env_name}"
    register(id=ID, entry_point="research.envs.metaworld:SawyerEnv", kwargs={"env_name": env_name})

# Add the PolyMetis Envs
register(
    id="PolyMetisReach-v0",
    entry_point="research.envs.polymetis:PolyMetisReach",
    max_episode_steps=400,
    kwargs={"use_quat": False, "fix_gripper": True},
)

register(
    id="PolyMetisBlockPush-v0",
    entry_point="research.envs.polymetis:PolyMetisBlockPush",
    max_episode_steps=200,
    kwargs={"use_quat": False},
)

register(
    id="PolyMetisBlockPushFix-v0",
    entry_point="research.envs.polymetis:PolyMetisBlockPush",
    max_episode_steps=200,
    kwargs={"use_quat": False, "fix_gripper": True},
)

# PyBullet Envs

register(
    id="PyBulletPandaReach-v0",
    entry_point="research.envs.pybullet_panda:Reach",
    max_episode_steps=100,
    kwargs={
        "fix_gripper": True,
        "initialization_noise": 0.35,
    },
)

register(
    id="PyBulletPandaBlockPush-v0",
    entry_point="research.envs.pybullet_panda:BlockPush",
    max_episode_steps=300,
    kwargs={
        "fix_gripper": False,
        "initialization_noise": 0.35,
        "randomize_block": True,
        "block_obs": True,
        "grasp_bonus": 0.01,
    },
)

register(
    id="PyBulletPandaBlockPushFix-v0",
    entry_point="research.envs.pybullet_panda:BlockPush",
    max_episode_steps=300,
    kwargs={
        "fix_gripper": True,
        "initialization_noise": 0.35,
        "randomize_block": True,
        "block_obs": True,
        "grasp_bonus": 0.0,
    },
)

# Cleanup extra imports
del suite
del register
