from typing import Any, Dict, Tuple

import numpy as np
import gymnasium as gym

class DiaynSkillWrapper(gym.ObservationWrapper):

    def __init__(self, env: gym.Env, skill_key="skill", state_key="state"):
        super().__init__(env)
        self.state_key = state_key
        self.skill_key = skill_key
        self.state_space = self.observation_space
        self.skill_space = gym.spaces.Box(0, 1, (5, ), np.float32)
        self.observation_space = gym.spaces.Dict({
            self.state_key: self.state_space,
            self.skill_key: self.skill_space
        })

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None) -> Tuple[Any, Dict[str, Any]]:
        self.current_skill = self.sample_skill()
        return super().reset(seed=seed, options=options)

    def sample_skill(self):
        return self.skill_space.sample()

    def observation(self, observation: Any) -> Any:
        return {
            self.state_key: observation,
            self.skill_key: self.current_skill
        }
    