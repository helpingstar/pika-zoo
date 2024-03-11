from typing import Optional
import pettingzoo
from pettingzoo.utils import BaseParallelWrapper
from gymnasium import spaces
import numpy as np


class NormalizeObservation(BaseParallelWrapper):
    def __init__(self, env: pettingzoo.ParallelEnv):
        super().__init__(env)
        self.agents = self.env.agents
        self.high = dict()
        self.low = dict()
        for agent in self.possible_agents:
            self.high[agent] = env.observation_space(agent).high
            self.low[agent] = env.observation_space(agent).low

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = super().reset(seed, options)
        self.agents = self.env.agents
        for agent in self.possible_agents:
            normalized_obs = (obs[agent] - self.low[agent]) / (self.high[agent] - self.low[agent])
            obs[agent] = normalized_obs
        return obs, info

    def step(self, action):
        obs, rews, terminateds, truncateds, infos = super().step(action)
        self.agents = self.env.agents
        for agent in self.possible_agents:
            normalized_obs = (obs[agent] - self.low[agent]) / (self.high[agent] - self.low[agent])
            obs[agent] = normalized_obs
        return obs, rews, terminateds, truncateds, infos

    def observation_space(self, agent) -> spaces.Space:
        return spaces.Box(low=0.0, high=1.0, shape=(35,), dtype=np.float32)
