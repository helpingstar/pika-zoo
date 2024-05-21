from pettingzoo.utils import BaseParallelWrapper
from pettingzoo.utils.env import ParallelEnv
from typing import Set


class RewardByBallPosition(BaseParallelWrapper):
    def __init__(
        self,
        env: ParallelEnv,
        additional_reward: tuple[float | int],
        x_line: int = 216,  # GROUND_HALF_WIDTH
        y_line: int = 176,  # NET_PILLAR_TOP_TOP_Y_COORD
    ):
        super().__init__(env)
        assert len(additional_reward) == 8
        self.x_line = x_line
        self.y_line = y_line
        self.additional_reward = additional_reward

    def step(self, action):
        obs, rews, terminateds, truncateds, infos = super().step(action)
        ball_x, ball_y = obs["player_1"][26], obs["player_1"][27]
        x_sign = ball_x >= self.x_line
        y_sign = ball_y > self.y_line

        ball_pos = 1 * int(y_sign) + 2 * int(x_sign)

        for i, agent in enumerate(self.possible_agents):
            rews[agent] += self.additional_reward[i * 4 + ball_pos]

        return obs, rews, terminateds, truncateds, infos
