from typing import Any

import numpy as np
import gymnasium as gym

class DiaynSkillWrapper(gym.ObservationWrapper):

    def __init__(self, env: gym.Env, skill_key="skill", state_key="state"):
        super().__init__(env)
        self.skill_key = skill_key
        self.state_key = state_key
        self.state_space = self.observation_space
        self.skill_space = gym.spaces.Box(0, 1, 5, np.float32)
        self.observation_space = gym.spaces.Dict({
            self.skill_key: self.state_space,
            self.state_key: self.skill_space
        })

    def sample_skill(self):
        return self.skill_space.sample()

    def observation(self, observation: Any) -> Any:
        return {
            self.skill_key: observation,
            self.state_key: self.sample_skill()
        }
    