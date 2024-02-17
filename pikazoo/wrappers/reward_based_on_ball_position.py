from pettingzoo.utils import BaseParallelWrapper
from pettingzoo.utils.env import ParallelEnv
from typing import Set


class RewardBasedOnBallPosition(BaseParallelWrapper):
    def __init__(
        self,
        env: ParallelEnv,
        position: Set[int],
        reward: float,
        is_player_2: bool,
        x_border: int = 216,  # GROUND_HALF_WIDTH
        y_border: int = 176,  # NET_PILLAR_TOP_TOP_Y_COORD
    ):
        super().__init__(env)
        assert isinstance(is_player_2, bool)
        assert isinstance(position, set)

        self.position = position
        self.reward = reward
        self.is_player_2 = is_player_2
        self.x_border = x_border
        self.y_border = y_border
        self.agent = self.env.agents[int(is_player_2)]

    def step(self, action):
        obs, rews, terminateds, truncateds, infos = super().step(action)
        ball_x, ball_y = obs[self.agent][26:28]
        x_sign = ball_x >= self.x_border
        y_sign = ball_y >= self.y_border

        if self.is_player_2:
            x_sign = x_sign ^ True

        if x_sign:
            if y_sign:
                ball_pos = 4
            else:
                ball_pos = 1
        else:
            if y_sign:
                ball_pos = 3
            else:
                ball_pos = 2

        if ball_pos in self.position:
            rews[self.agent] += self.reward

        return obs, rews, terminateds, truncateds, infos
