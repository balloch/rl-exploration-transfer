import os
import sys

curren_dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(curren_dir_path, os.pardir))
sys.path.append(parent_dir_path)


from typing import Any, Dict, Optional

from shimmy.dm_control_compatibility import DmControlCompatibilityV0
import realworldrl_suite.environments as rwrl
import gymnasium as gym


class RWRL2Gym(DmControlCompatibilityV0):

    def __init__(
        self,
        rwrl_kwargs: Dict[str, Any],
        render_mode: Optional[str] = None,
        render_kwargs: Optional[Dict[str, Any]] = None,
    ):
        dm_control_env = rwrl.load(**rwrl_kwargs)
        super().__init__(
            env=dm_control_env, render_mode=render_mode, render_kwargs=render_kwargs
        )


gym.register(id="rwrl2gym", entry_point=f"env_converters.rwrl_to_gym:RWRL2Gym")
