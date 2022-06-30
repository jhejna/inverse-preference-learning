# Register Network Classes here.
<<<<<<< HEAD
from .base import ActorCriticPolicy, ActorCriticRewardPolicy
from .mlp import ContinuousMLPActor, ContinuousMLPCritic, DiagonalGaussianMLPActor, MLPValue, MLPEncoder, DiscreteMLPCritic, RewardEnsemble
=======
from .base import ActorCriticPolicy
from .mlp import ContinuousMLPActor, ContinuousMLPCritic, DiagonalGaussianMLPActor, MLPValue, MLPEncoder, DiscreteMLPCritic
from .drqv2 import DRQV2Encoder, DRQV2Critic, DRQV2Actor
>>>>>>> 7c642a01d48b463eee9781661d3f73ab4c4020f8
