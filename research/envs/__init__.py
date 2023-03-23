# Register environment classes here
from .base import Empty

# Import the D4RL environments
import d4rl

# Import the robomimic enviroments
from .robomimic_env import RoboMimicEnv

# If we want to register environments in gym.
# These will be loaded when we import the research package.
from gym.envs import register


# Custom DM Control domains can be registered as follows:
# from . import <custom dm_env module>
# assert hasattr(<custom dm_env module>, 'SUITE')
# suite._DOMAINS['<custom dm_env module>'] = <custom dm_env module>

from metaworld.envs.mujoco.env_dict import ALL_V2_ENVIRONMENTS

# Add the meta world test environments.
# For each one, register the different tasks.

for env_name, env_cls in ALL_V2_ENVIRONMENTS.items():
    ID = f"mw_{env_name}"
    register(id=ID, entry_point="research.envs.metaworld:SawyerEnv", kwargs={"env_name": env_name})

del register
