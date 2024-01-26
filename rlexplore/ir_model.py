from typing import Type, Union, Dict, Any, Optional
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm

from stable_baselines3.ppo import PPO


def create_on_policy_ir_class(policy_cls: Type[OnPolicyAlgorithm]):
    class IRModel(policy_cls):
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
                        rollouts={"observations": self.rollout_buffer.observations},
                        time_steps=self.num_timesteps,
                        **self.compute_irs_kwargs,
                    )
                    self.rollout_buffer.rewards += intrinsic_rewards[:, :, 0]
                return True
            else:
                return False

    IRModel.__name__ = f"IR_{policy_cls.__name__}"

    return IRModel


IR_PPO = create_on_policy_ir_class(PPO)
