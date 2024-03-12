# The following code was referenced.
# https://github.com/ffelten/CrazyRL/blob/main/learning/cpu_env/wrappers.py

from typing import Optional
import pettingzoo
from pettingzoo.utils import BaseParallelWrapper


class RecordEpisodeStatistics(BaseParallelWrapper):
    """This wrapper will record episode statistics and print them at the end of each episode."""

    def __init__(self, env: pettingzoo.ParallelEnv):
        BaseParallelWrapper.__init__(self, env)
        self.agents = self.env.agents
        self.episode_rewards = {agent: 0 for agent in self.possible_agents}
        self.episode_lengths = {agent: 0 for agent in self.possible_agents}

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Resets the environment, recording episode statistics."""
        obs, info = super().reset(seed, options)
        self.agents = self.env.agents
        for agent in self.possible_agents:
            self.episode_rewards[agent] = 0
            self.episode_lengths[agent] = 0
        return obs, info

    def step(self, action):
        """Steps through the environment, recording episode statistics."""
        obs, rews, terminateds, truncateds, infos = super().step(action)
        self.agents = self.env.agents
        for agent in self.possible_agents:
            self.episode_rewards[agent] += rews[agent]
            self.episode_lengths[agent] += 1
        if all(terminateds.values()) or all(truncateds.values()):
            for agent in self.possible_agents:
                infos[agent]["episode"] = {
                    "r": self.episode_rewards[agent],
                    "l": self.episode_lengths[agent],
                }
        return obs, rews, terminateds, truncateds, infos
