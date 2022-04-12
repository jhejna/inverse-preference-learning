import torch
import numpy as np
import itertools

from .base import Algorithm
from research.networks.base import ActorCriticPolicy

class PPO(Algorithm):
    
    def __init__(self, *args, 
            num_steps=2048,
            num_epochs=10,
            clip_range=0.2,
            clip_range_vf=None,
            ent_coeff=0.0,
            vf_coeff=0.5,
            max_grad_norm=0.5,
            **kwargs
            ):
        super().__init__(*args, **kwargs)
        self.num_steps = num_steps
        self.num_epochs = num_epochs
        self.clip_range = clip_range
        self.ent_coeff = ent_coeff
        self.vf_coeff = vf_coeff
        self.max_grad_norm = max_grad_norm

    def _train_step(self, batch):
        pass

    def _validation_step(self, batch):
        raise NotImplementedError("RL Algorithm does not have a validation dataset.")
