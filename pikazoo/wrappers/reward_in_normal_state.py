from pettingzoo.utils import BaseParallelWrapper
from pettingzoo.utils.env import ParallelEnv


class RewardInNormalState(BaseParallelWrapper):
    def __init__(self, env: ParallelEnv, reward):
        super().__init__(env)
        self.reward = reward

    def step(self, action):
        obs, rews, terminateds, truncateds, infos = super().step(action)
        for i, agent in enumerate(self.possible_agents):
            if rews[agent] == 0:
                rews[agent] = self.reward
        return obs, rews, terminateds, truncateds, infos
