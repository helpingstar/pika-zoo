import functools
import gymnasium
import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding
from pettingzoo import ParallelEnv
from .physics import PikaPhysics, PikaUserInput, Player, Ball, GROUND_HALF_WIDTH, GROUND_WIDTH, PLAYER_HALF_LENGTH, PLAYER_TOUCHING_GROUND_Y_COORD, BALL_RADIUS, BALL_TOUCHING_GROUND_Y_COORD
from typing import List, Dict
import pygame
import os

GROUND_HEIGHT = 304

def env(**kwargs):
    env = raw_env(**kwargs)
    return env

def get_image(path):
    cwd = os.path.dirname(__file__)
    image = pygame.image.load(cwd + "/" + path)
    sfc = pygame.Surface(image.get_size(), flags=pygame.SRCALPHA)
    sfc.blit(image, (0, 0))
    return sfc

def blit_center(screen, source, dest):
    x = dest[0] - source.get_width() // 2
    y = dest[1] - source.get_height() // 2
    screen.blit(source, (x, y))

class raw_env(ParallelEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "pikazoo_v0",
        "render_fps": 25,
    }

    def __init__(self, render_mode=None):
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
        
        # Game Status
        # TODO : process about frames, ex) ball, player animation
        self.frames = 0
        self.render_mode = render_mode
        self.screen = None
        
        # Game Constants
        self._seed
        
        if self.render_mode == "human":
            self.clock = pygame.time.Clock()
        
        self.get_all_image()

        
        
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

            self.round_ended = True
        
        if self.render_mode == "human":
            self.render()
        
        if self.round_ended and not self.game_ended:
            self.physics.player1.initialize_for_new_round()
            self.physics.player2.initialize_for_new_round()
            self.physics.ball.initialize_for_new_round(self.is_player2_serve)
            self.round_ended = False

        observations = self._get_obs()
        rewards = {self.agents[0]: int(self.is_player2_serve), self.agents[1]: int(self.is_player2_serve)}
        terminations = {agent: self.game_ended for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = self._get_infos()
        
        return observations, rewards, terminations, truncations, infos

    def draw(self):
        self.draw_background()
        self.draw_player_and_ball()
        self.draw_scores_to_score_boards()

    
    def draw_player_and_ball(self):
        ball: Ball = self.physics.ball
        p1: Player = self.physics.player1
        p2: Player = self.physics.player2
        blit_center(self.screen, self.ball_0, (ball.x, ball.y))
        blit_center(self.screen, self.pikachu_0_0, (p1.x, p1.y))
        blit_center(self.screen, self.pikachu_0_0, (p2.x, p2.y))
        
        if ball.is_power_hit:
            blit_center(self.screen, self.ball_hyper, (ball.previous_x, ball.previous_y))
            blit_center(self.screen, self.ball_trail, (ball.previous_previous_x, ball.previous_previous_y))
            
    def draw_background(self):
        # sky
        for j in range(12):
            for i in range(432 // 16):
                self.screen.blit(self.sky_blue, (16*i, 16*j))
        
        # mountain
        self.screen.blit(self.mountain, (0, 188))

        # ground_red
        for i in range(432 // 16):
            self.screen.blit(self.ground_red, (16 * i, 248))
            
        # ground_line
        for i in range(1, 432 // 16 - 1):
            self.screen.blit(self.ground_line, (16 * i, 264))
        self.screen.blit(self.ground_line_leftmost, (0, 264))
        self.screen.blit(self.ground_line_rightmost, (432 - 16, 264))
        
        # ground_yellow
        for j in range(2):
            for i in range(432 // 16):
                self.screen.blit(self.ground_yellow, (16*i, 280 + 16 * j))

        # net pillar
        self.screen.blit(self.net_pillar_top, (213, 176))
        
        for j in range(12):
            self.screen.blit(self.net_pillar, (213, 184 + 8 * j))
        
    def draw_scores_to_score_boards(self):
        # player1
        if self.scores[0] >= 10:
            self.screen.blit(self.number[1], (14, 10))
        self.screen.blit(self.number[self.scores[0] % 10], (14 + 32, 10))
        
        # player2
        if self.scores[1] >= 10:
            self.screen.blit(self.number[1], (432 - 32 - 32 - 14, 10))
        self.screen.blit(self.number[self.scores[1] % 10], (432 - 32 - 32 - 14 + 32, 10))
        
    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        if self.screen is None:
            pygame.init()
            
            if self.render_mode == "human":
                self.screen = pygame.display.set_mode(
                    [GROUND_WIDTH, GROUND_HEIGHT]
                )
                pygame.display.set_caption("Pika-zoo")
            elif self.render_mode == "rgb_array":
                self.screen = pygame.Surface((GROUND_WIDTH, GROUND_HEIGHT))

        self.draw()
        
        observation = np.array(pygame.surfarray.pixels3d(self.screen))
        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        return (
            np.transpose(observation, axes=(1, 0, 2))
            if self.render_mode == "rgb_array"
            else None
        )
        
    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
        
    def get_all_image(self):
        self.ball_0 = get_image(os.path.join("img", "ball_0.png"))
        self.ball_1 = get_image(os.path.join("img", "ball_1.png"))
        self.ball_2 = get_image(os.path.join("img", "ball_2.png"))
        self.ball_3 = get_image(os.path.join("img", "ball_3.png"))
        self.ball_4 = get_image(os.path.join("img", "ball_4.png"))
        self.ball_hyper = get_image(os.path.join("img", "ball_hyper.png"))
        self.ball_punch = get_image(os.path.join("img", "ball_punch.png"))
        self.ball_trail = get_image(os.path.join("img", "ball_trail.png"))
        self.black = get_image(os.path.join("img", "black.png"))
        self.cloud = get_image(os.path.join("img", "cloud.png"))
        self.fight = get_image(os.path.join("img", "fight.png"))
        self.game_end = get_image(os.path.join("img", "game_end.png"))
        self.game_start = get_image(os.path.join("img", "game_start.png"))
        self.ground_line = get_image(os.path.join("img", "ground_line.png"))
        self.ground_line_leftmost = get_image(os.path.join("img", "ground_line_leftmost.png"))
        self.ground_line_rightmost = get_image(os.path.join("img", "ground_line_rightmost.png"))
        self.ground_red = get_image(os.path.join("img", "ground_red.png"))
        self.ground_yellow = get_image(os.path.join("img", "ground_yellow.png"))
        self.mark = get_image(os.path.join("img", "mark.png"))
        self.mountain = get_image(os.path.join("img", "mountain.png"))
        self.net_pillar = get_image(os.path.join("img", "net_pillar.png"))
        self.net_pillar_top = get_image(os.path.join("img", "net_pillar_top.png"))
        self.number = (get_image(os.path.join("img", "number_0.png")),
        get_image(os.path.join("img", "number_1.png")),
        get_image(os.path.join("img", "number_2.png")),
        get_image(os.path.join("img", "number_3.png")),
        get_image(os.path.join("img", "number_4.png")),
        get_image(os.path.join("img", "number_5.png")),
        get_image(os.path.join("img", "number_6.png")),
        get_image(os.path.join("img", "number_7.png")),
        get_image(os.path.join("img", "number_8.png")),
        get_image(os.path.join("img", "number_9.png")))
        self.pikachu_0_0 = get_image(os.path.join("img", "pikachu_0_0.png"))
        self.pikachu_0_1 = get_image(os.path.join("img", "pikachu_0_1.png"))
        self.pikachu_0_2 = get_image(os.path.join("img", "pikachu_0_2.png"))
        self.pikachu_0_3 = get_image(os.path.join("img", "pikachu_0_3.png"))
        self.pikachu_0_4 = get_image(os.path.join("img", "pikachu_0_4.png"))
        self.pikachu_1_0 = get_image(os.path.join("img", "pikachu_1_0.png"))
        self.pikachu_1_1 = get_image(os.path.join("img", "pikachu_1_1.png"))
        self.pikachu_1_2 = get_image(os.path.join("img", "pikachu_1_2.png"))
        self.pikachu_1_3 = get_image(os.path.join("img", "pikachu_1_3.png"))
        self.pikachu_1_4 = get_image(os.path.join("img", "pikachu_1_4.png"))
        self.pikachu_2_0 = get_image(os.path.join("img", "pikachu_2_0.png"))
        self.pikachu_2_1 = get_image(os.path.join("img", "pikachu_2_1.png"))
        self.pikachu_2_2 = get_image(os.path.join("img", "pikachu_2_2.png"))
        self.pikachu_2_3 = get_image(os.path.join("img", "pikachu_2_3.png"))
        self.pikachu_2_4 = get_image(os.path.join("img", "pikachu_2_4.png"))
        self.pikachu_3_0 = get_image(os.path.join("img", "pikachu_3_0.png"))
        self.pikachu_3_1 = get_image(os.path.join("img", "pikachu_3_1.png"))
        self.pikachu_4_0 = get_image(os.path.join("img", "pikachu_4_0.png"))
        self.pikachu_5_0 = get_image(os.path.join("img", "pikachu_5_0.png"))
        self.pikachu_5_1 = get_image(os.path.join("img", "pikachu_5_1.png"))
        self.pikachu_5_2 = get_image(os.path.join("img", "pikachu_5_2.png"))
        self.pikachu_5_3 = get_image(os.path.join("img", "pikachu_5_3.png"))
        self.pikachu_5_4 = get_image(os.path.join("img", "pikachu_5_4.png"))
        self.pikachu_6_0 = get_image(os.path.join("img", "pikachu_6_0.png"))
        self.pikachu_6_1 = get_image(os.path.join("img", "pikachu_6_1.png"))
        self.pikachu_6_2 = get_image(os.path.join("img", "pikachu_6_2.png"))
        self.pikachu_6_3 = get_image(os.path.join("img", "pikachu_6_3.png"))
        self.pikachu_6_4 = get_image(os.path.join("img", "pikachu_6_4.png"))
        self.pikachu_volleyball = get_image(os.path.join("img", "pikachu_volleyball.png"))
        self.pokemon = get_image(os.path.join("img", "pokemon.png"))
        self.ready = get_image(os.path.join("img", "ready.png"))
        self.sachisoft = get_image(os.path.join("img", "sachisoft.png"))
        self.shadow = get_image(os.path.join("img", "shadow.png"))
        self.sitting_pikachu = get_image(os.path.join("img", "sitting_pikachu.png"))
        self.sky_blue = get_image(os.path.join("img", "sky_blue.png"))
        self.wave = get_image(os.path.join("img", "wave.png"))
        self.with_computer = get_image(os.path.join("img", "with_computer.png"))
        self.with_friend = get_image(os.path.join("img", "with_friend.png"))
    
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

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

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