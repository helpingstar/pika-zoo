from typing import Any
import pettingzoo
from pettingzoo.utils import BaseParallelWrapper
from gymnasium import spaces


class SimplifyAction(BaseParallelWrapper):
    """
    Represent actions in relative directions instead of absolute directions,
    exclude actions that are not meaningful in gameplay,
    and reduce the number of valid actions from 18 to 13.
    """

    def __init__(self, env: pettingzoo.ParallelEnv):
        BaseParallelWrapper.__init__(self, env)
        self.action_map = {
            "player_1": (0, 1, 2, 3, 4, 6, 7, 10, 11, 12, 13, 14, 16),
            "player_2": (0, 1, 2, 4, 3, 7, 6, 10, 12, 11, 13, 15, 17),
        }
        self.action_spaces = dict(zip(self.possible_agents, [spaces.Discrete(13)] * 2))

    def step(self, actions: dict) -> tuple[dict, dict[Any, float], dict[Any, bool], dict[Any, bool], dict[Any, dict]]:
        """Map the actions to the original environment's actions."""
        actions = {agent: self.action_map[agent][actions[agent]] for agent in self.possible_agents}
        return super().step(actions)

    def action_space(self, agent: Any) -> spaces.Space:
        return self.action_spaces[agent]
