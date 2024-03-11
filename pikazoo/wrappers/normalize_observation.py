from typing import Optional
import pettingzoo
from pettingzoo.utils import BaseParallelWrapper


class NormalizeObservation(BaseParallelWrapper):
    def __init__(self, env: pettingzoo.ParallelEnv):
        super().__init__(env)
        self.high = dict()
        self.low = dict()
        for agent in self.possible_agents:
            self.high[agent] = self.observation_space(agent).high
            self.low[agent] = self.observation_space(agent).low

    def step(self, action):
        obs, rews, terminateds, truncateds, infos = super().step(action)
        for agent in self.possible_agents:
            normalized_obs = (obs[agent] - self.low[agent]) / (self.high[agent] - self.low[agent])
            obs[agent] = normalized_obs
        return obs, rews, terminateds, truncateds, infos

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = super().reset(seed, options)
        for agent in self.possible_agents:
            normalized_obs = (obs[agent] - self.low[agent]) / (self.high[agent] - self.low[agent])
            obs[agent] = normalized_obs
        return obs, info
