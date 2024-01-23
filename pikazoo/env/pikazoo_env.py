import functools
import gymnasium
import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv
from .physics import PikaPhysics, PikaUserInput, Player, Ball, GROUND_HALF_WIDTH, GROUND_WIDTH, PLAYER_HALF_LENGTH, PLAYER_TOUCHING_GROUND_Y_COORD, BALL_RADIUS, BALL_TOUCHING_GROUND_Y_COORD
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
        self.physics = PikaPhysics(False, False)
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
        observations = self._get_obs()
        infos = self._get_infos()
        return observations, infos

    def step(self, actions):
        for i, agent in enumerate(self.agents):
            self.keyboard_array[i].get_input(actions[agent])
        
        is_ball_touching_ground: bool = self.physics.run_engine_for_next_frame(self.keyboard_array)
        
        # TODO : audio play
        # TODO : render player, ball, clouds, wave
        
        if is_ball_touching_ground and not self.round_ended and not self.game_ended:
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

        observations = self._get_obs()
        rewards = {self.agents[0]: int(self.is_player2_serve), self.agents[1]: int(self.is_player2_serve)}
        terminations = {agent: self.game_ended for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = self._get_infos()
        
        return observations, rewards, terminations, truncations, infos
        
    def render(self):
        pass

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # Player1 : x, y, y_velocity, lying_down_duration_left, is_collision_with_ball_happened, state
        # Player2 : x, y, y_velocity, lying_down_duration_left, is_collision_with_ball_happened, state
        # Ball    : x, y, previous_x, previous_y, previous_previous_x, previous_previous_y, x_velocity, y_velocity, is_power_hit
        # hs) 108 : The maximum height reachable by the player.
        return spaces.Box(low=np.array([PLAYER_HALF_LENGTH, 108, -15, -2, 0, 0, 
                                 PLAYER_HALF_LENGTH, 108, -15, -2, 0, 0, 
                                 BALL_RADIUS, 0, BALL_RADIUS, 0, BALL_RADIUS, 0, -20, -123, 0]),
                   high=np.array([GROUND_HALF_WIDTH - PLAYER_HALF_LENGTH, PLAYER_TOUCHING_GROUND_Y_COORD, 16, 3, 1, 6,
                                  GROUND_HALF_WIDTH - PLAYER_HALF_LENGTH, PLAYER_TOUCHING_GROUND_Y_COORD, 16, 3, 1, 6,
                                  GROUND_WIDTH, BALL_TOUCHING_GROUND_Y_COORD, GROUND_WIDTH, BALL_TOUCHING_GROUND_Y_COORD, GROUND_WIDTH, BALL_TOUCHING_GROUND_Y_COORD, 20, 124, 1]), 
                   shape=(21,),
                   dtype=np.int32)

    def action_space(self, agent):
        return self.action_spaces[agent]

    def _get_infos(self):
        return {agent: {} for agent in self.agents}
    
    def _get_obs(self):
        obs1 = np.array(self._get_player_obs(self.agents[0]) 
                        + self._get_player_obs(self.agents[1])
                        + self._get_ball_obs())
        obs2 = obs1.copy()
        
        return {self.agents[0]: obs1, self.agents[1]: obs2}
    
    def _get_player_obs(self, agent: str):
        player: Player = None
        if agent == self.agents[0]:
            player = self.physics.player1
        else:
            player = self.physics.player2
        
        x = player.x
        y = player.y
        y_velocity = player.y_velocity
        lying_down_duration_left = player.lying_down_duration_left
        is_collision_with_ball_happened = int(player.is_collision_with_ball_happened)
        state = player.state
        return [x, y, y_velocity, lying_down_duration_left, is_collision_with_ball_happened, state]
        
        
    def _get_ball_obs(self):
        ball: Ball = self.physics.ball
        x = ball.x
        y = ball.y
        previous_x = ball.previous_x
        previous_y = ball.previous_y
        previous_previous_x = ball.previous_previous_x
        previous_previous_y = ball.previous_previous_y
        x_velocity = ball.x_velocity
        y_velocity = ball.y_velocity
        is_power_hit = ball.is_power_hit
        return [x, y, previous_x, previous_y, previous_previous_x, previous_previous_y, x_velocity, y_velocity, is_power_hit]