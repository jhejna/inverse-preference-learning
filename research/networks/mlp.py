import torch
from torch import nn
from torch import distributions
from torch.nn import functional as F

from .common import MLP

def weight_init(m, gain=1):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data, gain=gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)

class MLPEncoder(nn.Module):
    def __init__(self, observation_space, action_space, hidden_layers=[256, 256], act=nn.ReLU, ortho_init=False):
        assert len(hidden_layers) > 1, "Must have at least one hidden layer for a shared MLP Extractor"
        self.mlp = MLP(observation_space.shape[0], hidden_layers[-1], hidden_layers=hidden_layers[:-1], act=act)
        if ortho_init:
            self.apply(weight_init, gain=float(ortho_init)) # use the fact that True converts to 1.0
        

class ContinuousMLPCritic(nn.Module):

    def __init__(self, observation_space, action_space, hidden_layers=[256, 256], act=nn.ReLU, num_q_fns=2, ortho_init=False):
        super().__init__()
        self.qs = nn.ModuleList([
            MLP(observation_space.shape[0] + action_space.shape[0], 1, hidden_layers=hidden_layers, act=act)
         for _ in range(num_q_fns)])
        if ortho_init:
            self.apply(weight_init, gain=float(ortho_init)) # use the fact that True converts to 1.0

    def forward(self, obs, action):
        # TODO: convert this to an ensemble model to support an arbitrary number of Q functions with no computation time cost
        x = torch.cat((obs, action), dim=-1)
        return [q(x).squeeze(-1) for q in self.qs]

class MLPValue(nn.Module):

    def __init__(self, observation_space, action_space, hidden_layers=[256, 256], act=nn.ReLU, ortho_init=False):
        super().__init__()
        self.mlp = MLP(observation_space.shape[0], 1, hidden_layers=hidden_layers, act=act)
        if ortho_init:
            self.apply(weight_init, gain=float(ortho_init)) # use the fact that True converts to 1.0
        
    def forward(self, obs):
        return self.mlp(obs).squeeze(-1) # Return only scalar values, no final dim

class ContinuousMLPActor(nn.Module):

    def __init__(self, observation_space, action_space, hidden_layers=[256, 256], act=nn.ReLU, output_act=nn.Tanh, ortho_init=False):
        super().__init__()
        self.mlp = MLP(observation_space.shape[0], action_space.shape[0], hidden_layers=hidden_layers, act=act, output_act=output_act)
        if ortho_init:
            self.apply(weight_init, gain=float(ortho_init)) # use the fact that True converts to 1.0
        
    def forward(self, obs):
        return self.mlp(obs)

class SquashedNormal(distributions.TransformedDistribution):

    def __init__(self, loc, scale):
        self._loc = loc
        self.scale = scale
        self.base_dist = distributions.Normal(loc, scale)
        transforms = [distributions.transforms.TanhTransform(cache_size=1)]
        super().__init__(self.base_dist, transforms)

    @property
    def loc(self):
        loc = self._loc
        for transform in self.transforms:
            loc = transform(loc)
        return loc

class DiagonalGaussianMLPActor(nn.Module):

    def __init__(self, observation_space, action_space, hidden_layers=[256, 256], act=nn.ReLU, ortho_init=False, 
                       log_std_bounds=[-5, 2], state_dependent_log_std=True):
        super().__init__()
        self.state_dependent_log_std = state_dependent_log_std
        self.log_std_bounds = log_std_bounds
        if log_std_bounds is not None:
            assert log_std_bounds[0] < log_std_bounds[1]
        
        if self.state_dependent_log_std:
            self.mlp = MLP(observation_space.shape[0], 2*action_space.shape[0], hidden_layers=hidden_layers, act=act, output_act=None)
        else:
            self.mlp = MLP(observation_space.shape[0], 2*action_space.shape[0], hidden_layers=hidden_layers, act=act, output_act=None)
            self.log_std = nn.Parameter(torch.zeros(action_space.shape[0]), requires_grad=True) # initialize a single parameter vector
        
        if ortho_init:
            self.apply(weight_init, gain=float(ortho_init)) # use the fact that True converts to 1.0
        self.action_range = [float(action_space.low.min()), float(action_space.high.max())]
        
    def forward(self, obs):
        if self.state_dependent_log_std:
            mu, log_std = self.mlp(obs).chunk(2, dim=-1)
        else:
            mu, log_std = self.mlp(obs), self.log_std
        
        if self.log_std_bounds is not None:
            log_std = torch.tanh(log_std)
            log_std_min, log_std_max = self.log_std_bounds
            log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
            dist_class = SquashedNormal
        else:
            dist_class = distributions.Normal
        
        dist = dist_class(mu, log_std.exp())
        return dist

    def predict(self, obs, sample=False):
        dist = self(obs)
        if sample:
            action = dist.sample()
        else:
            action = dist.loc
        action = action.clamp(*self.action_range)
        return action
