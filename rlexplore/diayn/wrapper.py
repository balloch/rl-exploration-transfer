from typing import Any, Dict, Tuple, Optional

import numpy as np
import gymnasium as gym

class DiaynSkillWrapper(gym.ObservationWrapper):

    def __init__(self, env: gym.Env, skill_size: int = 20, use_discrete_skills: bool = True, skill_key: str = "skill", state_key: str = "state"):
        super().__init__(env)
        self.state_key = state_key
        self.skill_key = skill_key
        self.state_space = self.observation_space
        if use_discrete_skills:
            self.skill_space = gym.spaces.Discrete(skill_size)
        else:
            self.skill_space = gym.spaces.Box(0, 1, (skill_size, ), np.float32)
        self.observation_space = gym.spaces.Dict({
            self.state_key: self.state_space,
            self.skill_key: self.skill_space
        })

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Any, Dict[str, Any]]:
        self.current_skill = self.sample_skill()
        return super().reset(seed=seed, options=options)

    def sample_skill(self):
        return self.skill_space.sample()

    def observation(self, observation: Any) -> Any:
        return {
            self.state_key: observation,
            self.skill_key: self.current_skill
        }
    