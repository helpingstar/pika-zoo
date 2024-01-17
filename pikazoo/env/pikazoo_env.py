import functools
import gymnasium
from gymnasium.spaces import Dict, MultiBinary
from pettingzoo import ParallelEnv
from .physics import *


def env(**kwargs):
    env = raw_env(**kwargs)
    return env


class raw_env(ParallelEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "pikazoo_v0",
        "render_fps": 25,
    }

    def __init__(self):
        self.possible_agents = ["player_1", "player_2"]
        # left, right, up, down, power_hit, (down_right)
        self.action_spaces = {
            self.possible_agents[0]: MultiBinary(6),
            self.possible_agents[1]: MultiBinary(5),
        }

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]

    def step(self, actions):
        pass

    def render(self):
        pass

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return Dict({})

    def action_space(self, agent):
        return self.action_spaces[agent]
