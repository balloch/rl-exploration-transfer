from typing import Union, List, Type, Dict, Optional, Any

import torch
from torch import nn

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    NatureCNN,
    FlattenExtractor,
)
from stable_baselines3.common.distributions import CategoricalDistribution


import gymnasium as gym

from .noisy_mlp import NoisyMlpExtractor
from .noisy_layer import NoisyLinear


class NoisyNetCategoricalDistribution(CategoricalDistribution):
    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        action_logits = NoisyLinear(latent_dim, self.action_dim)
        return action_logits


class NoisyActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        *args,
        num_noisy_layers: int = 2,
        **kwargs,
    ):
        self.num_noisy_layers = num_noisy_layers

        super(NoisyActorCriticPolicy, self).__init__(
            *args,
            **kwargs,
        )

    def _build(self, lr_schedule: Schedule) -> None:
        if self.num_noisy_layers > 0:
            if isinstance(self.action_dist, CategoricalDistribution):
                self.action_dist = NoisyNetCategoricalDistribution(self.action_space.n)
            else:
                """
                To implement more action spaces following these steps:
                    1. Create a sub class of the distribution and name it Noisy{Distribution Name}
                    2. Override the proba_distribution_net method and replace all Linear layers with NoisyLayers
                    3. Add an elif statement to this block as followings --> elif isinstance(self.action_dist, {Distribution Name}):
                    4. Within the elif block add the following line --> self.action_dist = Noisy{Distribution Name}(self.action_space.n, {any other required args})
                """
                raise NotImplementedError(
                    f"Error: noisy probability distribution, not implement for action space of type {type(self.action_space)}. Must be Discrete."
                )

        return super()._build(lr_schedule)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = NoisyMlpExtractor(
            self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
            num_noisy_layers=self.num_noisy_layers - 1,
        )


class NoisyActorCriticCnnPolicy(NoisyActorCriticPolicy):
    def __init__(
        self,
        *args,
        features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
        num_noisy_layers: int = 2,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
            features_extractor_class=features_extractor_class,
            num_noisy_layers=num_noisy_layers,
        )
