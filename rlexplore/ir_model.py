from typing import Type, Dict, Any, Optional

import numpy as np
import torch

from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env import VecEnv, VecNormalize
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm

import stable_baselines3 as sb3
import inspect


def create_on_policy_ir_class(model_cls: Type[OnPolicyAlgorithm]):

    class IRModel(model_cls):
        def __init__(
            self,
            *args,
            ir_alg_cls: Optional[Type] = None,
            ir_alg_kwargs: Optional[Dict[str, Any]] = None,
            compute_irs_kwargs: Optional[Dict[str, Any]] = None,
            **kwargs,
        ):
            super().__init__(*args, **kwargs)

            self.compute_irs_kwargs = (
                {} if compute_irs_kwargs is None else compute_irs_kwargs
            )
            ir_alg_kwargs = {} if ir_alg_kwargs is None else ir_alg_kwargs

            if ir_alg_cls is None:
                self.exploration_alg = None
            else:
                self.exploration_alg = ir_alg_cls(**ir_alg_kwargs)

        def collect_rollouts(
            self,
            env: VecEnv,
            callback: BaseCallback,
            rollout_buffer: RolloutBuffer,
            n_rollout_steps: int,
        ) -> bool:
            result = super().collect_rollouts(
                env=env,
                callback=callback,
                rollout_buffer=rollout_buffer,
                n_rollout_steps=n_rollout_steps,
            )
            if result:
                if self.exploration_alg is not None:
                    intrinsic_rewards = self.exploration_alg.compute_irs(
                        rollouts={
                            "observations": self.rollout_buffer.observations,
                            "actions": self.rollout_buffer.actions,
                        },
                        time_steps=self.num_timesteps,
                        **self.compute_irs_kwargs,
                    )
                    self.rollout_buffer.rewards += intrinsic_rewards[:, :, 0]
                return True
            else:
                return False

    IRModel.__name__ = f"IR_{model_cls.__name__}"

    return IRModel


def create_off_policy_ir_class(model_cls: Type[OffPolicyAlgorithm]):

    class IRModel(model_cls):
        def __init__(
            self,
            *args,
            ir_alg_cls: Optional[Type] = None,
            ir_alg_kwargs: Optional[Dict[str, Any]] = None,
            compute_irs_kwargs: Optional[Dict[str, Any]] = None,
            **kwargs,
        ):
            super().__init__(*args, **kwargs)

            self.compute_irs_kwargs = (
                {} if compute_irs_kwargs is None else compute_irs_kwargs
            )
            ir_alg_kwargs = {} if ir_alg_kwargs is None else ir_alg_kwargs

            if ir_alg_cls is None:
                self.exploration_alg = None
            else:
                self.exploration_alg = ir_alg_cls(**ir_alg_kwargs)

            self.replay_buffer.sample_without_ir = self.replay_buffer.sample

            def sample_with_ir(
                replay_buffer: ReplayBuffer,
                batch_size: int,
                env: Optional[VecNormalize] = None,
            ) -> ReplayBufferSamples:
                samples = replay_buffer.sample_without_ir(batch_size, env)
                if self.exploration_alg is not None:
                    inp_obs = (
                        torch.clone(samples.observations)
                        .cpu()
                        .numpy()[:, np.newaxis, :]
                        .astype(np.float32)
                    )
                    inp_actions = (
                        torch.clone(samples.actions)
                        .cpu()
                        .numpy()[:, np.newaxis, :]
                        .astype(np.float32)
                    )
                    intrinsic_rewards = self.exploration_alg.compute_irs(
                        rollouts={"observations": inp_obs, "actions": inp_actions},
                        time_steps=self.num_timesteps,
                        **self.compute_irs_kwargs,
                    )
                    new_rewards = (
                        samples.rewards
                        + torch.from_numpy(intrinsic_rewards[:, :, 0]).cuda()
                    )
                    samples = ReplayBufferSamples(
                        samples.observations,
                        samples.actions,
                        samples.next_observations,
                        samples.dones,
                        new_rewards,
                    )
                return samples

            self.replay_buffer.sample = type(self.replay_buffer.sample)(
                sample_with_ir, self.replay_buffer
            )

    IRModel.__name__ = f"IR_{model_cls.__name__}"

    return IRModel


on_policy_algs = inspect.getmembers(
    sb3, lambda obj: inspect.isclass(obj) and issubclass(obj, OnPolicyAlgorithm)
)
for on_policy_alg_name, on_policy_alg_cls in on_policy_algs:
    globals()[f"IR_{on_policy_alg_name}"] = create_on_policy_ir_class(on_policy_alg_cls)

off_policy_algs = inspect.getmembers(
    sb3, lambda obj: inspect.isclass(obj) and issubclass(obj, OffPolicyAlgorithm)
)
for off_policy_alg_name, off_policy_alg_cls in off_policy_algs:
    globals()[f"IR_{off_policy_alg_name}"] = create_off_policy_ir_class(
        off_policy_alg_cls
    )
