from pettingzoo.utils import BaseParallelWrapper
from pettingzoo.utils.env import ParallelEnv


class RewardBasedOnBallPosition(BaseParallelWrapper):
    def __init__(
        self,
        env: ParallelEnv,
        reward: float,
        player: str,
        x_min: int = 0,
        x_max: int = 432,
        y_min: int = 0,
        y_max: int = 304,
    ):
        super().__init__(env)
        assert player in self.env.possible_agents

        self.reward = reward
        self.player = player
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

    def step(self, action):
        obs, rews, terminateds, truncateds, infos = super().step(action)
        ball_x, ball_y = obs[self.player][26:28]
        if (self.x_min <= ball_x <= self.x_max) and (
            self.y_min <= ball_y <= self.y_max
        ):
            rews[self.player] += self.reward
        return obs, rews, terminateds, truncateds, infos
