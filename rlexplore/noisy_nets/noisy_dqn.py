from typing import Optional, List, Type, Dict, Any

import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

import gym
from stable_baselines3.dqn.policies import QNetwork, DQNPolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    NatureCNN
)
from stable_baselines3.common.type_aliases import Schedule

class NoisyLinear(nn.Module):

    def __init__(self, in_features: int, out_features: int, bias: bool = True, std_init: float = 0.5):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.bias = bias

        self.weight_mu = nn.Parameter(torch.Tensor(self.out_features, self.in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(self.out_features, self.in_features))

        self.register_buffer(
            "weight_epsilon", torch.Tensor(self.out_features, self.in_features)
        )

        if self.bias:
            self.bias_mu = nn.Parameter(torch.Tensor(self.out_features))
            self.bias_sigma = nn.Parameter(torch.Tensor(self.out_features))
            self.register_buffer(
                "bias_epsilon", torch.Tensor(self.out_features)
            )

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(
            self.std_init / np.sqrt(self.in_features)
        )

        if self.bias:
            self.bias_mu.data.uniform_(-mu_range, mu_range)
            self.bias_sigma.data.fill_(
                self.std_init / np.sqrt(self.out_features)
            )

    def reset_noise(self):
        epsilon_in = NoisyLinear.scale_noise(self.in_features)
        epsilon_out = NoisyLinear.scale_noise(self.out_features)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        if self.bias:
            self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor, train: bool = True) -> torch.Tensor:
        if train:
            return F.linear(
                x,
                self.weight_mu + self.weight_sigma * self.weight_epsilon,
                self.bias_mu + self.bias_sigma * self.bias_epsilon if self.bias else 0,
            )
        else:
            return F.linear(
                x,
                self.weight_mu,
                self.bias_mu if self.bias else 0,
            )
        
    @staticmethod
    def scale_noise(size: int) -> torch.Tensor:
        x = torch.randn(size)

        return x.sign().mul(x.abs().sqrt())

def create_noisy_mlp(
    input_dim: int,
    output_dim: int,
    net_arch: List[int],
    activation_fn: Type[nn.Module] = nn.ReLU,
    squash_output: bool = False,
    with_bias: bool = True,
) -> List[nn.Module]:
    
    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0], bias=with_bias), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        if idx == len(net_arch) - 2:
            modules.append(NoisyLinear(net_arch[idx], net_arch[idx + 1], bias=with_bias))
        else:
            modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1], bias=with_bias))
        modules.append(activation_fn())
        
    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(NoisyLinear(last_layer_dim, output_dim, bias=with_bias))

    if squash_output:
        modules.append(nn.Tanh())

    return modules

class NoisyQNetwork(QNetwork):

    def __init__(
        self, 
        observation_space: gym.spaces.Space, 
        action_space: gym.spaces.Space, 
        features_extractor: nn.Module, 
        features_dim: int, 
        net_arch: Optional[List[int]] = None, 
        activation_fn: Type[nn.Module] = nn.ReLU, 
        normalize_images: bool = True
    ):
        super().__init__(
            observation_space, 
            action_space, 
            features_extractor, 
            features_dim, 
            net_arch, 
            activation_fn, 
            normalize_images
        )
        q_net = create_noisy_mlp(self.features_dim, self.action_space.n, self.net_arch, self.activation_fn)
        self.q_net = nn.Sequential(*q_net)

class NoisyDQNPolicy(DQNPolicy):
    def __init__(self, observation_space: gym.spaces.Space, action_space: gym.spaces.Space, lr_schedule: Schedule, net_arch: Optional[List[int]] = None, activation_fn: Type[nn.Module] = nn.ReLU, features_extractor_class: Type[BaseFeaturesExtractor] = ..., features_extractor_kwargs: Optional[Dict[str, Any]] = None, normalize_images: bool = True, optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam, optimizer_kwargs: Optional[Dict[str, Any]] = None):
        super().__init__(observation_space, action_space, lr_schedule, net_arch, activation_fn, features_extractor_class, features_extractor_kwargs, normalize_images, optimizer_class, optimizer_kwargs)

    def make_q_net(self) -> QNetwork:
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return NoisyQNetwork(**net_args).to(self.device)
    
class NoisyCnnPolicy(NoisyDQNPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )