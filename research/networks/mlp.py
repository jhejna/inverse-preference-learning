import math
from functools import partial
from typing import List, Optional, Type

import gym
import torch
from torch import distributions, nn
from torch.nn import functional as F

from .common import MLP, EnsembleMLP, LinearEnsemble


def weight_init(m: nn.Module, gain: int = 1) -> None:
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data, gain=gain)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    if isinstance(m, LinearEnsemble):
        for i in range(m.ensemble_size):
            # Orthogonal initialization doesn't care about which axis is first
            # Thus, we can just use ortho init as normal on each matrix.
            nn.init.orthogonal_(m.weight.data[i], gain=gain)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


class MLPEncoder(nn.Module):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        hidden_layers: List[int] = [256, 256],
        act: Type[nn.Module] = nn.ReLU,
        dropout: float = 0.0,
        normalization: Optional[Type[nn.Module]] = None,
        ortho_init: bool = False,
    ):
        assert len(hidden_layers) > 1, "Must have at least one hidden layer for a shared MLP Extractor"
        self.mlp = MLP(
            observation_space.shape[0],
            hidden_layers[-1],
            hidden_layers=hidden_layers[:-1],
            act=act,
            dropout=dropout,
            normalization=normalization,
        )
        if ortho_init:
            self.apply(partial(weight_init, gain=float(ortho_init)))  # use the fact that True converts to 1.0


class ContinuousMLPCritic(nn.Module):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        hidden_layers: List[int] = [256, 256],
        act: Type[nn.Module] = nn.ReLU,
        dropout: float = 0.0,
        normalization: Optional[Type[nn.Module]] = None,
        ensemble_size: int = 2,
        ortho_init: bool = False,
        output_gain: Optional[float] = None,
    ):
        super().__init__()
        self.ensemble_size = ensemble_size
        if self.ensemble_size > 1:
            self.q = EnsembleMLP(
                observation_space.shape[0] + action_space.shape[0],
                1,
                ensemble_size=ensemble_size,
                hidden_layers=hidden_layers,
                act=act,
                dropout=dropout,
                normalization=normalization,
            )
        else:
            self.q = MLP(
                observation_space.shape[0] + action_space.shape[0],
                1,
                hidden_layers=hidden_layers,
                act=act,
                dropout=dropout,
                normalization=normalization,
            )

        if ortho_init:
            self.apply(partial(weight_init, gain=float(ortho_init)))  # use the fact that True converts to 1.0
            if output_gain is not None:
                self.mlp.last_layer.apply(partial(weight_init, gain=output_gain))

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat((obs, action), dim=-1)
        q = self.q(x).squeeze(-1)  # Remove the last dim
        if self.ensemble_size == 1:
            q = q.unsqueeze(0)  # add in the ensemble dim
        return q


class DiscreteMLPCritic(nn.Module):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        hidden_layers: List[int] = [256, 256],
        act: Type[nn.Module] = nn.ReLU,
        dropout: float = 0.0,
        normalization: Optional[Type[nn.Module]] = None,
        ortho_init: bool = False,
        output_gain: Optional[float] = None,
    ):
        super().__init__()
        self.q = MLP(
            observation_space.shape[0],
            action_space.n,
            hidden_layers=hidden_layers,
            act=act,
            dropout=dropout,
            normalization=normalization,
        )
        if ortho_init:
            self.apply(partial(weight_init, gain=float(ortho_init)))  # use the fact that True converts to 1.0
            if output_gain is not None:
                self.mlp.last_layer.apply(partial(weight_init, gain=output_gain))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.q(obs)


class MLPValue(nn.Module):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        hidden_layers: List[int] = [256, 256],
        act: Type[nn.Module] = nn.ReLU,
        output_act: Optional[Type[nn.Module]] = None,
        dropout: float = 0.0,
        normalization: Optional[Type[nn.Module]] = None,
        ortho_init: bool = False,
        output_gain: Optional[float] = None,
    ):
        super().__init__()
        self.mlp = MLP(
            observation_space.shape[0],
            1,
            hidden_layers=hidden_layers,
            act=act,
            dropout=dropout,
            normalization=normalization,
            output_act=output_act,
        )
        if ortho_init:
            self.apply(partial(weight_init, gain=float(ortho_init)))  # use the fact that True converts to 1.0
            if output_gain is not None:
                self.mlp.last_layer.apply(partial(weight_init, gain=output_gain))

    def forward(self, obs):
        return self.mlp(obs).squeeze(-1)  # Return only scalar values, no final dim


class ContinuousMLPActor(nn.Module):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        hidden_layers: List[int] = [256, 256],
        act: Type[nn.Module] = nn.ReLU,
        dropout: float = 0.0,
        normalization: Optional[Type[nn.Module]] = None,
        output_act: Optional[Type[nn.Module]] = None,
        ortho_init: bool = False,
        output_gain: Optional[float] = None,
    ):
        super().__init__()
        assert isinstance(observation_space, gym.spaces.Box) and len(observation_space.shape) == 1

        self.mlp = MLP(
            observation_space.shape[0],
            action_space.shape[0],
            hidden_layers=hidden_layers,
            act=act,
            dropout=dropout,
            normalization=normalization,
            output_act=output_act,
        )
        if ortho_init:
            self.apply(partial(weight_init, gain=float(ortho_init)))  # use the fact that True converts to 1.0
            if output_gain is not None:
                self.mlp.last_layer.apply(partial(weight_init, gain=output_gain))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(obs)


class SquashedNormal(distributions.TransformedDistribution):
    def __init__(self, loc: torch.Tensor, scale: torch.Tensor) -> None:
        self._loc = loc
        self.scale = scale
        self.base_dist = distributions.Normal(loc, scale)
        transforms = [distributions.transforms.TanhTransform(cache_size=1)]
        super().__init__(self.base_dist, transforms)

    @property
    def loc(self) -> torch.Tensor:
        loc = self._loc
        for transform in self.transforms:
            loc = transform(loc)
        return loc


class DiagonalGaussianMLPActor(nn.Module):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        hidden_layers: List[int] = [256, 256],
        act: Type[nn.Module] = nn.ReLU,
        dropout: float = 0.0,
        normalization: Optional[Type[nn.Module]] = None,
        ortho_init: bool = False,
        output_gain: Optional[float] = None,
        log_std_bounds: List[int] = [-5, 2],
        state_dependent_log_std: bool = True,
        squash_normal: bool = True,
        log_std_tanh: bool = True,
        output_act: Optional[Type[nn.Module]] = None,
    ):
        super().__init__()
        # If we have a dict space, concatenate the input dims
        assert isinstance(observation_space, gym.spaces.Box) and len(observation_space.shape) == 1

        self.state_dependent_log_std = state_dependent_log_std
        self.log_std_bounds = log_std_bounds
        self.squash_normal = squash_normal
        self.log_std_tanh = log_std_tanh

        # Perform checks to make sure arguments are consistent
        assert log_std_bounds is None or log_std_bounds[0] < log_std_bounds[1], "invalid log_std bounds"
        assert not (output_act is not None and squash_normal), "Cannot use output act and squash normal"

        if self.state_dependent_log_std:
            action_dim = 2 * action_space.shape[0]
        else:
            action_dim = action_space.shape[0]
            self.log_std = nn.Parameter(
                torch.zeros(action_space.shape[0]), requires_grad=True
            )  # initialize a single parameter vector

        self.mlp = MLP(
            observation_space.shape[0],
            action_dim,
            hidden_layers=hidden_layers,
            act=act,
            dropout=dropout,
            normalization=normalization,
            output_act=output_act,
        )

        if ortho_init:
            self.apply(partial(weight_init, gain=float(ortho_init)))  # use the fact that True converts to 1.0
            if output_gain is not None:
                self.mlp.last_layer.apply(partial(weight_init, gain=output_gain))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if self.state_dependent_log_std:
            mu, log_std = self.mlp(obs).chunk(2, dim=-1)
        else:
            mu, log_std = self.mlp(obs), self.log_std
        if self.log_std_bounds is not None:
            if self.log_std_tanh:
                log_std = torch.tanh(log_std)
                log_std_min, log_std_max = self.log_std_bounds
                log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
            else:
                log_std = torch.clamp(log_std, *self.log_std_bounds)

        dist_class = SquashedNormal if self.squash_normal else distributions.Normal
        dist = dist_class(mu, log_std.exp())
        return dist


class RewardMLPEnsemble(nn.Module):
    def __init__(
        self,
        observation_space,
        action_space,
        hidden_layers=[256, 256],
        act=nn.LeakyReLU,
        ensemble_size=3,
        output_act=nn.Tanh,
    ):
        super().__init__()
        self.net = EnsembleMLP(
            observation_space.shape[0] + action_space.shape[0],
            1,
            ensemble_size=ensemble_size,
            hidden_layers=hidden_layers,
            act=act,
            output_act=output_act,
        )

    def forward(self, obs, action):
        obs_action = torch.cat((obs, action), dim=1)
        return self.net(obs_action).squeeze(-1)


class MetaRewardMLPEnsemble(nn.Module):
    def __init__(
        self,
        observation_space,
        action_space,
        hidden_layers=[256, 256, 256],
        ensemble_size=3,
        act=F.leaky_relu,
        output_act=torch.tanh,
    ):
        super().__init__()
        params = {}
        last_dim = observation_space.shape[0] + action_space.shape[0]
        self.num_layers = len(hidden_layers) + 1
        for i, dim in enumerate(
            hidden_layers
            + [
                1,
            ]
        ):
            weight = torch.empty(ensemble_size, last_dim, dim)
            for w in weight:
                w.transpose_(0, 1)
                nn.init.kaiming_uniform_(w, a=math.sqrt(5))
                w.transpose_(0, 1)
            params[f"linear_w_{i}"] = nn.Parameter(weight)
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(params[f"linear_w_{i}"][0].T)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            params[f"linear_b_{i}"] = nn.Parameter(
                nn.init.uniform_(torch.empty(ensemble_size, 1, dim, requires_grad=True), -bound, bound)
            )
            last_dim = dim

        self.params = nn.ParameterDict(params)
        self.ensemble_size = ensemble_size
        self.act = act
        self.output_act = output_act

    def forward(self, obs, action, params=None):
        if params is None:
            params = self.params
        x = torch.cat((obs, action), dim=1)
        x = x.repeat(self.ensemble_size, 1, 1)
        for i in range(self.num_layers):
            x = torch.baddbmm(params[f"linear_b_{i}"], x, params[f"linear_w_{i}"])
            if i == self.num_layers - 1:
                x = self.output_act(x)
            else:
                x = self.act(x)
        return x.squeeze(-1)
