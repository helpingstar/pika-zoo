import functools
import gymnasium
import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv
from .physics import PikaPhysics, PikaUserInput, GROUND_HALF_WIDTH, GROUND_WIDTH, PLAYER_HALF_LENGTH, PLAYER_TOUCHING_GROUND_Y_COORD, BALL_RADIUS, BALL_TOUCHING_GROUND_Y_COORD
from typing import List, Dict

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
            self.possible_agents[0]: spaces.MultiBinary(6),
            self.possible_agents[1]: spaces.MultiBinary(5),
        }
        self.physics = PikaPhysics(True, True)
        self.keyboard_array: List[PikaUserInput] = [PikaUserInput(), PikaUserInput()]
        # [0] for player 1 score, [1] for player 2 score
        self.scores: List[int] = [0, 0]
        # winning score: if either one of the players reaches this score, game ends
        self.winning_score: int = 15
        # Is the game ended?
        self.game_ended: bool = False
        # Is the round ended?
        self.round_ended: bool = False
        # Will player 2 serve?
        self.is_player2_serve: bool = False
        
        
    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        
        self.game_ended = False
        self.round_ended = False
        self.is_player2_serve = False
        self.physics.player1.game_ended = False
        self.physics.player1.is_winner = False
        self.physics.player2.game_ended = False
        self.physics.player2.is_winner = False
        
        self.scores[0] = 0
        self.scores[1] = 0
        
        self.physics.player1.initialize_for_new_round()
        self.physics.player2.initialize_for_new_round()
        self.physics.ball.initialize_for_new_round(self.is_player2_serve)
        
        # TODO : render player, ball, score, cloud, wave
        # TODO : audio play
        
        # TODO : return

    def step(self, actions):
        for i, agent in enumerate(self.agents):
            self.keyboard_array[i].get_input(actions[agent])
        
        is_ball_touching_ground: bool = self.physics.run_engine_for_next_frame(self.keyboard_array)
        
        # TODO : audio play
        # TODO : render player, ball, clouds, wave
        
        if is_ball_touching_ground and self.round_ended and self.game_ended:
            if self.physics.ball.punch_effect_x < GROUND_HALF_WIDTH:
                self.is_player2_serve = True
                self.scores[1] += 1
                if self.scores[1] >= self.winning_score:
                    self.game_ended = True
                    self.physics.player1.isWinner = False
                    self.physics.player2.isWinner = True
                    self.physics.player1.gameEnded = True
                    self.physics.player2.gameEnded = True
            else:
                self.is_player2_serve = False
                self.scores[0] += 1
                if self.scores[0] >= self.winning_score:
                    self.game_ended = True
                    self.physics.player1.isWinner = True
                    self.physics.player2.isWinner = False
                    self.physics.player1.gameEnded = True
                    self.physics.player2.gameEnded = True
            # TODO : render score
            self.round_ended = True
        
        # TODO if is_powerhit else
        
        # TODO
        observations = None
        rewards = None
        terminations = None
        truncations = None
        infos = None
        
        return observations, rewards, terminations, truncations, infos
        
    def render(self):
        pass

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # hs) 108 : The maximum height reachable by the player.
        return spaces.Dict({
            self.possible_agents[0]: {
                "position" : spaces.Box(low=np.array([PLAYER_HALF_LENGTH, 108]), high=np.array([GROUND_HALF_WIDTH - PLAYER_HALF_LENGTH, PLAYER_TOUCHING_GROUND_Y_COORD])),
                "is_diving": spaces.Discrete(2)},
            self.possible_agents[1]: {
                "position" : spaces.Box(low=np.array([GROUND_HALF_WIDTH + PLAYER_HALF_LENGTH, 108]), high=np.array([GROUND_WIDTH - PLAYER_HALF_LENGTH, PLAYER_TOUCHING_GROUND_Y_COORD])),
                "is_diving": spaces.Discrete(2)},
            "ball": {
                "origin" : spaces.Box(low=np.array([BALL_RADIUS, 0]), high=np.array([GROUND_WIDTH, BALL_TOUCHING_GROUND_Y_COORD])),
                "hyper" : spaces.Box(low=np.array([BALL_RADIUS, 0]), high=np.array([GROUND_WIDTH, BALL_TOUCHING_GROUND_Y_COORD])),
                "trail" : spaces.Box(low=np.array([BALL_RADIUS, 0]), high=np.array([GROUND_WIDTH, BALL_TOUCHING_GROUND_Y_COORD]))
            }})

    def action_space(self, agent):
        return self.action_spaces[agent]
