"""
All of the code for pika-zoo was written based on https://github.com/gorisanson/pikachu-volleyball
"""

from typing import List, Dict
import numpy as np
from numpy.typing import NDArray

# ground width
GROUND_WIDTH: int = 432
# ground half-width, it is also the net pillar x coordinate
GROUND_HALF_WIDTH: int = GROUND_WIDTH // 2  # integer division
# player (Pikachu) length: width = height = 64
PLAYER_LENGTH: int = 64
# player half length
PLAYER_HALF_LENGTH: int = PLAYER_LENGTH // 2  # integer division
# player's y coordinate when they are touching ground
PLAYER_TOUCHING_GROUND_Y_COORD: int = 244
# ball's radius
BALL_RADIUS: int = 20
# ball's y coordinate when it is touching ground
BALL_TOUCHING_GROUND_Y_COORD: int = 252
# net pillar's half width (this value is on this physics engine only, not on the sprite pixel size)
NET_PILLAR_HALF_WIDTH: int = 25
# net pillar top's top side y coordinate
NET_PILLAR_TOP_TOP_Y_COORD: int = 176
# net pillar top's bottom side y coordinate (this value is on this physics engine only)
NET_PILLAR_TOP_BOTTOM_Y_COORD: int = 192

"""
// TODO : Description of INFINITE_LOOP_LIMIT
"""
INFINITE_LOOP_LIMIT: int = 1000


class PikaUserInput:
    """
    Class representing user input (from keyboard or joystick, whatever)
    hs) This class operates similarly to the `PikaKeyboard` in the original code.
        This is because the current environment does not accept keyboard input.
    """

    def __init__(self) -> None:
        # 0: no horizontal-direction input, -1: left-direction input, 1: right-direction input
        self.x_direction: int = 0
        # 0: no vertical-direction input, -1: up-direction input, 1: down-direction input
        self.y_direction: int = 0
        # 0: auto-repeated or no power hit input, 1: not auto-repeated power hit input
        self.power_hit: int = 0

        self.power_hit_key_is_down_previous = False
        self.left_key = False
        self.right_key = False
        self.up_key = False
        self.down_key = False
        self.power_hit_key = False
        self.down_right_key = False

    def get_input(self, action: NDArray[np.int8]) -> None:
        """
        This method does not process keyboard inputs;
        instead, it accepts a list of actions.
        This operates similarly to the `getInput()` method of the `PikaKeyboard` class in the original code.
        [left, right, up, down, powerHit, downRight]
        Args:
            action (NDArray[np.int8]): Whether each key is pressed.
        """
        # if player_2
        if action.shape[0] < 6:
            self.down_right_key = None
        else:
            self.down_right_key = bool(action[5])

        self.left_key = bool(action[0])
        self.right_key = bool(action[1])
        self.up_key = bool(action[2])
        self.down_key = bool(action[3])
        self.power_hit_key = bool(action[4])

        if self.left_key:
            self.x_direction = -1
        elif self.right_key or (self.down_right_key is not None and self.down_right_key):
            self.x_direction = 1
        else:
            self.x_direction = 0

        if self.up_key:
            self.y_direction = -1
        elif self.down_key or (self.down_right_key is not None and self.down_right_key):
            self.y_direction = 1
        else:
            self.y_direction = 0

        is_down: bool = self.power_hit_key
        if not self.power_hit_key_is_down_previous and is_down:
            self.power_hit = 1
        else:
            self.power_hit = 0
        self.power_hit_key_is_down_previous = is_down


class PikaPhysics:
    """Class representing a pack of physical objects i.e. players and ball
    whose physical values are calculated and set by `physics_engine` function
    """

    def __init__(
        self,
        is_player1_computer: bool,
        is_player2_computer: bool,
        np_random: np.random.Generator,
    ) -> None:
        """Create a physics pack

        Args:
            is_player1_computer (bool): Is player on the left (player 1) controlled by computer?
            is_player2_computer (bool): Is player on the right (player 2) controlled by computer?
            np_random (np.random.Generator): The environment-dependent np.random.generator
        """
        self.player1 = Player(False, is_player1_computer, np_random)
        self.player2 = Player(True, is_player2_computer, np_random)
        self.ball = Ball(False)
        self.np_random = np_random

    def run_engine_for_next_frame(self, user_input_array: List[PikaUserInput]) -> bool:
        """run `physicsEngine` function with this physics object and user input

        Args:
            user_input_array (List[PikaUserInput]): userInputArray[0]: PikaUserInput object for player 1, userInputArray[1]: PikaUserInput object for player 2

        Returns:
            bool: Is ball touching ground?
        """
        is_ball_touching_ground: bool = physics_engine(
            self.player1, self.player2, self.ball, user_input_array, self.np_random
        )
        return is_ball_touching_ground


class Player:
    """Class representing a player"""

    def __init__(self, is_player2: bool, is_computer: bool, np_random: np.random.Generator):
        """create a player

        Args:
            is_player2 (bool): Is this player on the right side?
            is_computer (bool): Is this player controlled by computer?
            np_random (np.random.Generator): The environment-dependent np.random.generator
        """
        # Is this player on the right side?
        self.is_player2: bool = is_player2
        # Is controlled by computer?
        self.is_computer: bool = is_computer
        self.np_random = np_random
        self.initialize_for_new_round()

        # -1: left, 0: no diving, 1: right
        self.diving_direction: int = 0
        self.lying_down_duration_left: int = -1
        self.is_winner: bool = False
        self.game_ended: bool = False

        """
        It flips randomly to 0 or 1 by the `let_computer_decide_user_input` function
        when ball is hanging around on the other player's side.
        If it is 0, computer player stands by around the middle point of their side.
        If it is 1, computer player stands by adjacent to the net.
        0 or 1
        """
        self.computer_where_to_stand_by: int = 0

        """
        This dictionary is not in the player pointers of the original source code.
        But for sound effect (especially for stereo sound),
        it is convenient way to give sound property to a Player.
        The original name is stereo sound.
        """
        self.sound: Dict[str, bool] = {"pipikachu": False, "pika": False, "chu": False}

    def initialize_for_new_round(self):
        # x coord, initialized to 36 (player1) or 396 (player2)
        self.x: int = 36
        if self.is_player2:
            self.x = GROUND_WIDTH - 36

        # y coord, initialized to 244
        self.y: int = PLAYER_TOUCHING_GROUND_Y_COORD
        # y direction velocity, initialized to 0
        self.y_velocity: int = 0
        # initialized to 0 i.e False
        self.is_collision_with_ball_happened: bool = False

        """Player's state
        0: normal, 1: jumping, 2: jumping_and_power_hitting, 3: diving
        4: lying_down_after_diving, 5: win!, 6: lost
        0, 1, 2, 3, 4, 5 or 6
        initialized to 0
        """
        self.state: int = 0
        # initialized to 0
        self.frame_number: int = 0
        # initialized to 1
        self.normal_status_arm_swing_direction: int = 1
        # initialized to 0
        self.delay_before_next_frame: int = 0

        """
        This value is initialized to (_rand() % 5) before the start of every round.
        The greater the number, the bolder the computer player.
        If computer has higher boldness,
        judges more the ball is hanging around the other player's side,
        has greater distance to the expected landing point of the ball,
        jumps more,
        and dives less.
        See the source code of the `let_computer_decide_user_input` function.
        """
        self.computer_boldness: int = self.np_random.integers(0, 5)


class Ball:
    """Class representing a ball"""

    def __init__(self, is_player2_serve: bool):
        """Create a ball

        Args:
            is_player2_serve (bool): Will player 2 serve on this new round?
        """
        self.initialize_for_new_round(is_player2_serve)
        # x coord of expected landing point
        self.expected_landing_point_x: int = 0
        """ball rotation frame number selector
        During the period where it continues to be 5, hyper ball glitch occur.
        0, 1, 2, 3, 4 or 5
        """
        self.rotation: int = 0
        self.fine_rotation: int = 0
        # x coord for punch effect
        self.punch_effect_x: int = 0
        # y coord for punch effect
        self.punch_effect_y: int = 0

        """Following previous values are for trailing effect for power hit
        """
        self.previous_x: int = 0
        self.previous_previous_x: int = 0
        self.previous_y: int = 0
        self.previous_previous_y: int = 0

        """this property is not in the ball pointer of the original source code.
        But for sound effect (especially for stereo sound),
        it is convenient way to give sound property to a Ball.
        The original name is stereo sound.
        """
        self.sound = {"power_hit": False, "ball_touches_ground": False}

    def initialize_for_new_round(self, is_player2_serve: bool):
        """Initialize for new round

        Args:
            is_player2_serve (bool): will player on the right side serve on this new round?
        """
        # x coord, initialized to 56 or 376
        self.x: int = 56
        if is_player2_serve:
            self.x = GROUND_WIDTH - 56
        # y coord, initialized to 0
        self.y: int = 0
        # x direction velocity, initialized to 0
        self.x_velocity: int = 0
        # y direction velocity, initialized to 1
        self.y_velocity: int = 1
        # punch effect radius, initialized to 0
        self.punch_effect_radius: int = 0
        # is power hit, initialized to 0 i.e. False
        self.is_power_hit: bool = False


def physics_engine(
    player1: Player,
    player2: Player,
    ball: Ball,
    user_input_array: List[PikaUserInput],
    np_random: np.random.Generator,
) -> bool:
    """This is the Pikachu Volleyball physics engine!
    This physics engine calculates and set the physics values for the next frame.

    Args:
        player1 (Player): player on the left side
        player2 (Player): player on the right side
        ball (Ball): ball
        user_input_array (List[PikaUserInput]): userInputArray[0]: user input for player 1, userInputArray[1]: user input for player 2
        np_random (np.random.Generator): The environment-dependent np.random.generator

    Returns:
        bool: Is ball touching ground?
    """
    is_ball_touching_ground: bool = process_collision_between_ball_and_world_and_set_ball_position(ball)

    player = None
    the_other_player = None
    for i in range(2):
        if i == 0:
            player = player1
            the_other_player = player2
        else:
            player = player2
            the_other_player = player1

        # calculate expected x
        # https://github.com/helpingstar/pika-zoo/pull/5
        if player1.is_computer or player2.is_computer:
            calculate_expected_landing_point_x_for(ball)

        process_player_movement_and_set_player_position(player, user_input_array[i], the_other_player, ball, np_random)

    for i in range(2):
        if i == 0:
            player = player1
        else:
            player = player2

        is_happened: bool = is_collision_between_ball_and_player_happened(ball, player.x, player.y)

        if is_happened:
            if not player.is_collision_with_ball_happened:
                process_collision_between_ball_and_player(ball, player.x, user_input_array[i], player.state, np_random)
                # https://github.com/helpingstar/pika-zoo/pull/5
                if player1.is_computer or player2.is_computer:
                    calculate_expected_landing_point_x_for(ball)
                player.is_collision_with_ball_happened = True
        else:
            player.is_collision_with_ball_happened = False

    return is_ball_touching_ground


def is_collision_between_ball_and_player_happened(ball: Ball, player_x: int, player_y: int) -> bool:
    """Is collision between ball and player happened?

    Args:
        ball (Ball): ball
        player_x (int): player.x
        player_y (int): player.y

    Returns:
        bool:
    """
    diff = ball.x - player_x
    if abs(diff) <= PLAYER_HALF_LENGTH:
        diff = ball.y - player_y
        if abs(diff) <= PLAYER_HALF_LENGTH:
            return True
    return False


def process_collision_between_ball_and_world_and_set_ball_position(ball: Ball) -> bool:
    """Process collision between ball and world and set ball position

    Args:
        ball (Ball): ball

    Returns:
        bool: Is ball touching ground?
    """
    ball.previous_previous_x = ball.previous_x
    ball.previous_previous_y = ball.previous_y
    ball.previous_x = ball.x
    ball.previous_y = ball.y

    future_fine_rotation = ball.fine_rotation + (ball.x_velocity // 2)
    """
    If future_fine_rotation == 50, it skips next if statement finely.
    Then ball.fine_rotation = 50, and then ball.rotation = 5 (which designates hyper ball sprite!).
    In this way, hyper ball glitch occur!
    If this happen at the end of round,
    since ball.xVelocity is 0-initialized at each start of round,
    hyper ball sprite is rendered continuously until a collision happens.
    """
    if future_fine_rotation < 0:
        future_fine_rotation += 50
    elif future_fine_rotation > 50:
        future_fine_rotation += -50

    ball.fine_rotation = future_fine_rotation
    ball.rotation = ball.fine_rotation // 10

    future_ball_x: int = ball.x + ball.x_velocity

    """
    If the center of ball would get out of left world bound or right world bound, bounce back.
    In this if statement, when considering left-right symmetry,
    "future_ball_x > GROUND_WIDTH" should be changed to "future_ball_x > (GROUND_WIDTH - BALL_RADIUS)",
    or "future_ball_x < BALL_RADIUS" should be changed to "future_ball_x < 0".
    Maybe the former one is more proper when seeing Pikachu player's x-direction boundary.
    Is this a mistake of the author of the original game?
    Or, was it set to this value to resolve infinite loop problem? (See comments on the constant INFINITE_LOOP_LIMIT.)
    If apply (future_ball_x > (GROUND_WIDTH - BALL_RADIUS)), and if the maximum number of loop is not limited,
    it is observed that infinite loop in the function `expected_landing_point_x_when_power_hit` does not terminate.
    """
    if (future_ball_x < BALL_RADIUS) or (future_ball_x > GROUND_WIDTH):
        ball.x_velocity = -ball.x_velocity

    future_ball_y = ball.y + ball.y_velocity
    # if the center of ball would get out of upper world bound
    if future_ball_y < 0:
        ball.y_velocity = 1

    if abs(ball.x - GROUND_HALF_WIDTH) < NET_PILLAR_HALF_WIDTH and ball.y > NET_PILLAR_TOP_TOP_Y_COORD:
        if ball.y <= NET_PILLAR_TOP_BOTTOM_Y_COORD:
            if ball.y_velocity > 0:
                ball.y_velocity = -ball.y_velocity
        else:
            if ball.x < GROUND_HALF_WIDTH:
                ball.x_velocity = -abs(ball.x_velocity)
            else:
                ball.x_velocity = abs(ball.x_velocity)

    future_ball_y = ball.y + ball.y_velocity
    # if ball would touch ground
    if future_ball_y > BALL_TOUCHING_GROUND_Y_COORD:
        ball.sound["ball_touches_ground"] = True

        ball.y_velocity = -ball.y_velocity
        ball.punch_effect_x = ball.x
        ball.y = BALL_TOUCHING_GROUND_Y_COORD
        ball.punch_effect_radius = BALL_RADIUS
        ball.punch_effect_y = BALL_TOUCHING_GROUND_Y_COORD + BALL_RADIUS
        return True
    ball.y = future_ball_y
    ball.x = ball.x + ball.x_velocity
    ball.y_velocity += 1

    return False


def process_player_movement_and_set_player_position(
    player: Player,
    user_input: PikaUserInput,
    the_other_player: Player,
    ball: Ball,
    np_random: np.random.Generator,
) -> None:
    """Process player movement according to user input and set player position

    Args:
        player (Player): player
        user_input (PikaUserInput): user_input
        the_other_player (Player): the_other_player
        ball (Ball): ball
    """
    if player.is_computer:
        let_computer_decide_user_input(player, ball, the_other_player, user_input, np_random)

    # if player is lying down.. don't move
    if player.state == 4:
        player.lying_down_duration_left += -1
        if player.lying_down_duration_left < -1:
            player.state = 0
        return

    # process x-direction movement
    player_velocity_x: int = 0
    if player.state < 5:
        if player.state < 3:
            player_velocity_x = user_input.x_direction * 6
        else:
            # player.state == 3 i.e. player is diving.
            player_velocity_x = player.diving_direction * 8

    future_player_x: int = player.x + player_velocity_x
    player.x = future_player_x

    # process player's x-direction world boundary
    if not player.is_player2:
        # if player is player1
        if future_player_x < PLAYER_HALF_LENGTH:
            player.x = PLAYER_HALF_LENGTH
        elif future_player_x > GROUND_HALF_WIDTH - PLAYER_HALF_LENGTH:
            player.x = GROUND_HALF_WIDTH - PLAYER_HALF_LENGTH
    else:
        # if player is player2
        if future_player_x < GROUND_HALF_WIDTH + PLAYER_HALF_LENGTH:
            player.x = GROUND_HALF_WIDTH + PLAYER_HALF_LENGTH
        elif future_player_x > GROUND_WIDTH - PLAYER_HALF_LENGTH:
            player.x = GROUND_WIDTH - PLAYER_HALF_LENGTH

    # jump
    if (
        player.state < 3
        and user_input.y_direction == -1  # up-direction input
        and player.y == PLAYER_TOUCHING_GROUND_Y_COORD  # player is touching on the ground
    ):
        player.y_velocity = -16
        player.state = 1
        player.frame_number = 0
        player.sound["chu"] = True

    # gravity
    future_player_y: int = player.y + player.y_velocity
    player.y = future_player_y
    if future_player_y < PLAYER_TOUCHING_GROUND_Y_COORD:
        player.y_velocity += 1
    elif future_player_y > PLAYER_TOUCHING_GROUND_Y_COORD:
        # if player is landing..
        player.y_velocity = 0
        player.y = PLAYER_TOUCHING_GROUND_Y_COORD
        player.frame_number = 0
        if player.state == 3:
            # if player is diving..
            player.state = 4
            player.frame_number = 0
            player.lying_down_duration_left = 3
        else:
            player.state = 0

    if user_input.power_hit == 1:
        if player.state == 1:
            # if player is jumping
            # then player do power hit
            player.delay_before_next_frame = 5
            player.frame_number = 0
            player.state = 2
            player.sound["pika"] = True
        elif player.state == 0 and user_input.x_direction != 0:
            # then player do diving!
            player.state = 3
            player.frame_number = 0
            player.diving_direction = user_input.x_direction
            player.y_velocity = -5
            player.sound["chu"] = True

    if player.state == 1:
        player.frame_number = (player.frame_number + 1) % 3
    elif player.state == 2:
        if player.delay_before_next_frame < 1:
            player.frame_number += 1
            if player.frame_number > 4:
                player.frame_number = 0
                player.state = 1
        else:
            player.delay_before_next_frame -= 1
    elif player.state == 0:
        player.delay_before_next_frame += 1
        if player.delay_before_next_frame > 3:
            player.delay_before_next_frame = 0
            future_frame_number: int = player.frame_number + player.normal_status_arm_swing_direction
            if future_frame_number < 0 or future_frame_number > 4:
                player.normal_status_arm_swing_direction = -player.normal_status_arm_swing_direction
            player.frame_number = player.frame_number + player.normal_status_arm_swing_direction
    # hs) If player.game_ended == True, the environment terminates immediately, so the code below does not execute.
    if player.game_ended:
        if player.state == 0:
            if player.is_winner:
                player.state = 5
                player.sound["pipikachu"] = True
            else:
                player.state = 6
            player.delay_before_next_frame = 0
            player.frame_number = 0

        process_game_end_frame_for(player)


def process_game_end_frame_for(player: Player):
    """Process game end frame (for winner and loser motions) for the given player

    Args:
        player (Player): player
    """
    if player.game_ended and player.frame_number < 4:
        player.delay_before_next_frame += 1
        if player.delay_before_next_frame > 4:
            player.delay_before_next_frame = 0
            player.frame_number += 1


def process_collision_between_ball_and_player(
    ball: Ball,
    player_x: int,
    user_input: PikaUserInput,
    player_state: int,
    np_random: np.random.Generator,
):
    """Process collision between ball and player.
    This function only sets velocity of ball and expected landing point x of ball.
    This function does not set position of ball.
    The ball position is set by `process_collision_between_ball_and_world_and_set_ball_position` function

    Args:
        ball (Ball): ball
        player_x (int): player.x
        user_input (PikaUserInput): user_input
        player_state (int): player.state
        np_random (np.random.Generator): The environment-dependent np.random.generator
    """

    """
    player_x is pika's x position
    if collision occur,
    greater the x position difference between pika and ball,
    greater the x velocity of the ball
    """
    if ball.x < player_x:
        ball.x_velocity = -(abs(ball.x - player_x) // 3)
    elif ball.x > player_x:
        ball.x_velocity = abs(ball.x - player_x) // 3

    # If ball velocity x is 0, randomly choose one of -1, 0, 1.
    if ball.x_velocity == 0:
        ball.x_velocity = np_random.integers(0, 3) - 1

    ball_abs_y_velocity: int = abs(ball.y_velocity)
    ball.y_velocity = -ball_abs_y_velocity

    if ball_abs_y_velocity < 15:
        ball.y_velocity = -15

    # player is jumping and power hitting
    if player_state == 2:
        if ball.x < GROUND_HALF_WIDTH:
            ball.x_velocity = (abs(user_input.x_direction) + 1) * 10
        else:
            ball.x_velocity = -(abs(user_input.x_direction) + 1) * 10

        ball.punch_effect_x = ball.x
        ball.punch_effect_y = ball.y

        ball.y_velocity = abs(ball.y_velocity) * user_input.y_direction * 2
        ball.punch_effect_radius = BALL_RADIUS

        ball.sound["power_hit"] = True

        ball.is_power_hit = True
    else:
        ball.is_power_hit = False
    # https://github.com/helpingstar/pika-zoo/pull/5
    # calculate_expected_landing_point_x_for(ball)


def calculate_expected_landing_point_x_for(ball: Ball):
    """Calculate x coordinate of expected landing point of the ball

    Args:
        ball (Ball): ball
    """
    copy_ball: Dict[str, int] = {
        "x": ball.x,
        "y": ball.y,
        "x_velocity": ball.x_velocity,
        "y_velocity": ball.y_velocity,
    }
    loop_counter = 0
    while True:
        loop_counter += 1

        future_copy_ball_x: int = copy_ball["x_velocity"] + copy_ball["x"]
        if future_copy_ball_x < BALL_RADIUS or future_copy_ball_x > GROUND_WIDTH:
            copy_ball["x_velocity"] = -copy_ball["x_velocity"]
        if copy_ball["y"] + copy_ball["y_velocity"] < 0:
            copy_ball["y_velocity"] = 1

        # If copy ball touches net
        if (
            abs(copy_ball["x"] - GROUND_HALF_WIDTH) < NET_PILLAR_HALF_WIDTH
            and copy_ball["y"] > NET_PILLAR_TOP_TOP_Y_COORD
        ):
            if copy_ball["y"] < NET_PILLAR_TOP_BOTTOM_Y_COORD:
                if copy_ball["y_velocity"] > 0:
                    copy_ball["y_velocity"] = -copy_ball["y_velocity"]
            else:
                if copy_ball["x"] < GROUND_HALF_WIDTH:
                    copy_ball["x_velocity"] = -abs(copy_ball["x_velocity"])
                else:
                    copy_ball["x_velocity"] = abs(copy_ball["x_velocity"])

        copy_ball["y"] = copy_ball["y"] + copy_ball["y_velocity"]
        # if copy_ball would touch ground
        if copy_ball["y"] > BALL_TOUCHING_GROUND_Y_COORD or loop_counter >= INFINITE_LOOP_LIMIT:
            break

        copy_ball["x"] = copy_ball["x"] + copy_ball["x_velocity"]
        copy_ball["y_velocity"] += 1
    ball.expected_landing_point_x = copy_ball["x"]


def let_computer_decide_user_input(
    player: Player,
    ball: Ball,
    the_other_player: Player,
    user_input: PikaUserInput,
    np_random: np.random.Generator,
):
    """
    Computer controls its player by this function.
    Computer decides the user input for the player it controls,
    according to the game situation it figures out
    by the given parameters (player, ball and theOtherPlayer),
    and reflects these to the given user input object.
    Args:
        player (Player): The player whom computer controls
        ball (Ball): ball
        the_other_player (Player): The other player
        user_input (PikaUserInput): user input of the player whom computer controls
        np_random (np.random.Generator): The environment-dependent np.random.generator
    """
    user_input.x_direction = 0
    user_input.y_direction = 0
    user_input.power_hit = 0

    virtual_expected_landing_point_x = ball.expected_landing_point_x
    if abs(ball.x - player.x) > 100 and abs(ball.x_velocity) < player.computer_boldness + 5:
        left_boundary: int = int(player.is_player2) * GROUND_HALF_WIDTH
        if (
            ball.expected_landing_point_x <= left_boundary
            or ball.expected_landing_point_x >= int(player.is_player2) * GROUND_WIDTH + GROUND_HALF_WIDTH
        ) and player.computer_where_to_stand_by == 0:
            # If conditions above met, the computer estimates the proper location to stay as the middle point of their side
            virtual_expected_landing_point_x = left_boundary + (GROUND_HALF_WIDTH // 2)

    if abs(virtual_expected_landing_point_x - player.x) > player.computer_boldness + 8:
        if player.x < virtual_expected_landing_point_x:
            user_input.x_direction = 1
        else:
            user_input.x_direction = -1
    elif np_random.integers(0, 20) == 0:
        player.computer_where_to_stand_by = np_random.integers(0, 2)

    if player.state == 0:
        if (
            abs(ball.x_velocity) < player.computer_boldness + 3
            and abs(ball.x - player.x) < PLAYER_HALF_LENGTH
            and ball.y > -36
            and ball.y < 10 * player.computer_boldness + 84
            and ball.y_velocity > 0
        ):
            user_input.y_direction = -1

        left_boundary: int = int(player.is_player2) * GROUND_HALF_WIDTH
        right_boundary: int = (int(player.is_player2) + 1) * GROUND_HALF_WIDTH

        if (
            ball.expected_landing_point_x > left_boundary
            and ball.expected_landing_point_x < right_boundary
            and abs(ball.x - player.x) > player.computer_boldness * 5 + PLAYER_LENGTH
            and ball.x > left_boundary
            and ball.x < right_boundary
            and ball.y > 174
        ):
            # If conditions above met, the computer decides to dive!
            user_input.power_hit = 1
            if player.x < ball.x:
                user_input.x_direction = 1
            else:
                user_input.x_direction = -1
    elif player.state == 1 or player.state == 2:
        if abs(ball.x - player.x) > 8:
            if player.x < ball.x:
                user_input.x_direction = 1
            else:
                user_input.x_direction = -1
        if abs(ball.x - player.x) < 48 and abs(ball.y - player.y) < 48:
            will_input_power_hit: bool = decide_whether_input_power_hit(
                player, ball, the_other_player, user_input, np_random
            )
            if will_input_power_hit:
                user_input.power_hit = 1
                if abs(the_other_player.x - player.x) < 80 and user_input.y_direction != -1:
                    user_input.y_direction = -1


def decide_whether_input_power_hit(
    player: Player,
    ball: Ball,
    the_other_player: Player,
    user_input: PikaUserInput,
    np_random: np.random.Generator,
) -> bool:
    """This function is called by `let_computer_decide_user_input`,
    and also sets x and y direction user input so that it participate in
    the decision of the direction of power hit.

    Args:
        player (Player): the player whom computer controls
        ball (Ball): ball
        the_other_player (Player): The other rplayer
        user_input (PikaUserInput): user input for the player whom computer controls
        np_random (np.random.Generator): The environment-dependent np.random.generator

    Returns:
        bool: Will input power hit?
    """
    if np_random.integers(0, 2) == 0:
        for x_direction in range(1, -1, -1):
            for y_direction in range(-1, 2, 1):
                expected_landing_point_x: int = expected_landing_point_x_when_power_hit(x_direction, y_direction, ball)
                if (
                    expected_landing_point_x <= int(player.is_player2) * GROUND_HALF_WIDTH
                    or expected_landing_point_x >= int(player.is_player2) * GROUND_WIDTH + GROUND_HALF_WIDTH
                ) and abs(expected_landing_point_x - the_other_player.x) > PLAYER_LENGTH:
                    user_input.x_direction = x_direction
                    user_input.y_direction = y_direction
                    return True
    else:
        for x_direction in range(1, -1, -1):
            for y_direction in range(1, -2, -1):
                expected_landing_point_x = expected_landing_point_x_when_power_hit(x_direction, y_direction, ball)
                if (
                    expected_landing_point_x <= int(player.is_player2) * GROUND_HALF_WIDTH
                    or expected_landing_point_x >= int(player.is_player2) * GROUND_WIDTH + GROUND_HALF_WIDTH
                ) and abs(expected_landing_point_x - the_other_player.x) > PLAYER_LENGTH:
                    user_input.x_direction = x_direction
                    user_input.y_direction = y_direction
                    return True
    return False


def expected_landing_point_x_when_power_hit(
    user_input_x_direction: int, user_input_y_direction: int, ball: Ball
) -> int:
    """This function is called by `decide_whether_input_power_hit`,
    and calculates the expected x coordinate of the landing point of the ball
    when power hit

    Args:
        user_input_x_direction (int):
        user_input_y_direction (int):
        ball (Ball): ball

    Returns:
        int: x coord of expected landing point when power hit the ball
    """
    copy_ball: Dict[str, int] = {
        "x": ball.x,
        "y": ball.y,
        "x_velocity": ball.x_velocity,
        "y_velocity": ball.y_velocity,
    }
    if copy_ball["x"] < GROUND_HALF_WIDTH:
        copy_ball["x_velocity"] = (abs(user_input_x_direction) + 1) * 10
    else:
        copy_ball["x_velocity"] = -(abs(user_input_x_direction) + 1) * 10
    copy_ball["y_velocity"] = abs(copy_ball["y_velocity"]) * user_input_y_direction * 2

    loop_counter = 0
    while True:
        loop_counter += 1

        future_copy_ball_x: int = copy_ball["x"] + copy_ball["x_velocity"]
        if future_copy_ball_x < BALL_RADIUS or future_copy_ball_x > GROUND_WIDTH:
            copy_ball["x_velocity"] = -copy_ball["x_velocity"]
        if copy_ball["y"] + copy_ball["y_velocity"] < 0:
            copy_ball["y_velocity"] = 1
        if (
            abs(copy_ball["x"] - GROUND_HALF_WIDTH) < NET_PILLAR_HALF_WIDTH
            and copy_ball["y"] > NET_PILLAR_TOP_TOP_Y_COORD
        ):
            """
            The code below maybe is intended to make computer do mistakes.
            The player controlled by computer occasionally power hit ball that is bounced back by the net pillar,
            since code below do not anticipate the bounce back.
            """
            if copy_ball["y_velocity"] > 0:
                copy_ball["y_velocity"] = -copy_ball["y_velocity"]
            """
            An alternative code for making the computer not do those mistakes is as below. 
            
            if copy_ball["y"] <= NET_PILLAR_TOP_BOTTOM_Y_COORD:
                if copy_ball["y_velocity"] > 0:
                    copy_ball["y_velocity"] = -copy_ball["y_velocity"]
            else:
                if copy_ball["x"] < GROUND_HALF_WIDTH:
                    copy_ball["x_velocity"] = -abs(copy_ball["x_velocity"])
                else:
                    copy_ball["x_velocity"] = abs(copy_ball["x_velocity"])
            """

        copy_ball["y"] = copy_ball["y"] + copy_ball["y_velocity"]
        if copy_ball["y"] > BALL_TOUCHING_GROUND_Y_COORD or loop_counter >= INFINITE_LOOP_LIMIT:
            return copy_ball["x"]
        copy_ball["x"] = copy_ball["x"] + copy_ball["x_velocity"]
        copy_ball["y_velocity"] += 1
