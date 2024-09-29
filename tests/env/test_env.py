from pikazoo import pikazoo_v0
import numpy as np
from typing import Dict
from numpy.typing import NDArray


def test_env_observation_symmetry():
    env = pikazoo_v0.env(winning_score=15, is_player1_computer=True, is_player2_computer=True, render_mode=None)
    observations, infos = env.reset()
    observation_divide_and_assert(observations)
    while env.agents:
        actions = {agent: 0 for agent in env.agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)
        observation_divide_and_assert(observations)


def observation_divide_and_assert(observations: Dict[str, NDArray]):
    player1_info1, player2_info1 = observations["player_1"][0:13], observations["player_1"][13:26]
    player2_info2, player1_info2 = observations["player_2"][0:13], observations["player_2"][13:26]
    assert np.all(player1_info1 == player1_info2)
    assert np.all(player2_info1 == player2_info2)
