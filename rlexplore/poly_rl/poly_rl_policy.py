from typing import Union, List, Type, Dict, Optional, Any, Tuple

import torch
from torch import nn
import gym
import numpy as np

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    NatureCNN,
    FlattenExtractor,
)

from rlexplore.poly_rl.poly_rl import PolyRL

from tensorboardX import SummaryWriter


class PolyRLActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        betta: float = 0.0001,
        epsilon: float = 0,
        sigma_squared: float = 0.00007,
        lambda_: float = 0.035,
        gamma: float = 0.99,
        start_steps: int = 10000,
        logdir=None,
    ):
        super(PolyRLActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            sde_net_arch,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )
        self.betta = (betta,)
        self.epsilon = (epsilon,)
        self.sigma_squared = (sigma_squared,)
        self.lambda_ = lambda_
        self.start_steps = start_steps
        self.poly_rl = PolyRL(
            gamma=gamma,
            nb_actions=action_space.shape[0],
            nb_observations=observation_space.shape[0],
            max_action=float(action_space.high[0]),
            min_action=float(min(action_space.low)),
            actor_target_function=self,
            betta=betta,
            epsilon=epsilon,
            sigma_squared=sigma_squared,
            lambda_=lambda_,
        )
        self.counter_actions = 0
        assert logdir is not None
        self.writer = SummaryWriter(logdir=logdir)
        self.writer.STOP = True
        self.previous_action = None

    def get_exploration_percentage(self):
        return self.poly_rl.percentage_exploration

    def _predict(
        self, observation: torch.Tensor, deterministic: bool = False
    ) -> torch.Tensor:
        self.counter_actions += 1
        if self.start_steps < self.counter_actions:
            action = super()._predict(
                observation=observation, deterministic=deterministic
            )
            return action
        else:
            state = np.array(observation.cpu())
            self.previous_state = state
            action = self.poly_rl.select_action(
                state,
                self.previous_action,
                tensor_board_writer=self.writer,
                step_number=self.counter_actions,
            )
            action = torch.clamp(action, -1, 1).reshape(-1)
            self.previous_action = action
            return action

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: bool = False,
        deterministic: bool = False,
    ):
        if episode_start:
            self.previous_action = None
            self.previous_state = None
            self.poly_rl.reset_parameters_in_beginning_of_episode(
                self.nb_environment_reset
            )
        return super().predict(observation, state, episode_start, deterministic)


class PolyRLActorCriticCnnPolicy(PolyRLActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        betta: float = 0.0001,
        epsilon: float = 0,
        sigma_squared: float = 0.00007,
        lambda_: float = 0.035,
        gamma: float = 0.99,
        logdir=None,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            sde_net_arch,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            betta,
            epsilon,
            sigma_squared,
            lambda_,
            gamma,
            logdir,
        )
