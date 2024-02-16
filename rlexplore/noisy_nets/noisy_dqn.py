from typing import Optional, List, Type, Dict, Any

import torch
from torch import nn


import gymnasium as gym
from stable_baselines3.dqn.policies import QNetwork, DQNPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN
from stable_baselines3.common.type_aliases import Schedule

from rlexplore.noisy_nets.noisy_mlp import create_noisy_mlp


class NoisyQNetwork(QNetwork):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        features_extractor: nn.Module,
        features_dim: int,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor,
            features_dim,
            net_arch,
            activation_fn,
            normalize_images,
        )
        q_net = create_noisy_mlp(
            self.features_dim, self.action_space.n, self.net_arch, self.activation_fn
        )
        self.q_net = nn.Sequential(*q_net)


class NoisyDQNPolicy(DQNPolicy):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
        )

    def make_q_net(self) -> QNetwork:
        net_args = self._update_features_extractor(
            self.net_args, features_extractor=None
        )
        return NoisyQNetwork(**net_args).to(self.device)


class NoisyDQNCnnPolicy(NoisyDQNPolicy):
    def __init__(
        self,
        *args,
        features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
        **kwargs,
    ):
        super().__init__(
            *args,
            features_extractor_class=features_extractor_class,
            **kwargs,
        )
