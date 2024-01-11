"""
All of the code for pika-zoo was written based on https://github.com/gorisanson/pikachu-volleyball
"""
from typing import List

from rand import rand

# ground width
GROUND_WIDTH = 432
# ground half-width, it is also the net pillar x coordinate
GROUND_HALF_WIDTH = (GROUND_WIDTH / 2) | 0  # integer division
# player (Pikachu) length: width = height = 64
PLAYER_LENGTH = 64
# player half length
PLAYER_HALF_LENGTH = (PLAYER_LENGTH / 2) | 0  # integer division
# player's y coordinate when they are touching ground
PLAYER_TOUCHING_GROUND_Y_COORD = 244
# ball's radius
BALL_RADIUS = 20
# ball's y coordinate when it is touching ground
BALL_TOUCHING_GROUND_Y_COORD = 252
# net pillar's half width (this value is on this physics engine only, not on the sprite pixel size)
NET_PILLAR_HALF_WIDTH = 25
# net pillar top's top side y coordinate
NET_PILLAR_TOP_TOP_Y_COORD = 176
# net pillar top's bottom side y coordinate (this value is on this physics engine only)
NET_PILLAR_TOP_BOTTOM_Y_COORD = 192

INFINITE_LOOP_LIMIT = 1000


class PikaPhysics:
    def __init__(self, is_player1_computer, is_player2_computer):
        """Create a physics pack

        Args:
            is_player1_computer (bool): Is player on the left (player 1) controlled by computer?
            is_player2_computer (bool): Is player on the right (player 2) controlled by computer?
        """
        self.player1 = Player(False, is_player1_computer)
        self.player2 = Player(True, is_player2_computer)
        self.ball = Ball(False)

    def run_engine_for_next_frame(user_input_array):
        is_ball_touching_ground = physics_engine(
            self.player1, self.player2, self.ball, user_input_array
        )
        return is_ball_touching_ground


class PikaUserInput:
    def __init__(self) -> None:
        self.x_direction = 0
        self.y_direction = 0
        self.power_hit = 0


class Player:
    def __init__(self, is_player2, is_computer):
        self.is_player2 = is_player2
        self.is_computer = is_computer
        self.initialize_for_new_round()

        self.diving_direction = 0
        self.lying_down_duration_left = -1
        self.is_winner = False
        self.game_ended = False

        self.computer_where_to_stand_by = 0

        self.sound = {"pipikachu": False, "pika": False, "chu": False}

    def initialize_for_new_round(self):
        self.x = 36
        if self.is_player2:
            self.x = GROUND_WIDTH - 36

        self.y = PLAYER_TOUCHING_GROUND_Y_COORD
        self.y_velocity = 0
        self.is_collision_with_ball_happened = False

        self.state = 0
        self.frame_number = 0
        self.normal_status_arm_swing_direction = 1
        self.delay_before_next_frame = 0

        self.computer_boldness = rand() % 5


class Ball:
    def __init__(self, is_player2_serve):
        self.initialize_for_new_round(is_player2_serve)
        self.expected_landing_point_x = 0
        self.rotation = 0
        self.fine_rotation = 0
        self.punch_effect_x = 0
        self.punch_effect_y = 0

        self.previous_x = 0
        self.previous_previous_x = 0
        self.previous_y = 0
        self.previous_previous_y = 0

        self.sound = {"power_hit": False, "ball_touches_ground": False}

    def initialize_for_new_round(self, is_player2_serve):
        self.x = 56
        if is_player2_serve:
            self.x = GROUND_WIDTH - 56
        self.y = 0
        self.x_velocity = 0
        self.y_velocity = 1
        self.punch_effect_radius = 0
        self.is_power_hit = False


def physics_engine(
    player1: Player, player2: Player, ball: Ball, user_input_array: List[PikaUserInput]
):
    is_ball_touching_ground = (
        process_collision_between_ball_and_world_and_set_ball_position(ball)
    )

    player = None
    the_other_player = None
    for i in range(2):
        if i == 0:
            player = player1
            the_other_player = player2
        else:
            player = player2
            the_other_player = player1

        calculate_expected_landing_point_x_for(ball)

        process_player_movement_and_set_player_position(
            player, user_input_array[i], the_other_player, ball
        )

    for i in range(2):
        if i == 0:
            player = player1
        else:
            player = player2


def process_collision_between_ball_and_world_and_set_ball_position(ball: Ball):
    ball.previous_previous_x = ball.previous_x
    ball.previous_previous_y = ball.previous_y
    ball.previous_x = ball.x
    ball.previous_y = ball.y

    # origin : let futureFineRotation = ball.fineRotation + ((ball.xVelocity / 2) | 0);
    future_fine_rotation = ball.fine_rotation + (ball.x_velocity // 2)

    if future_fine_rotation:
        future_fine_rotation += 50
    elif future_fine_rotation > 50:
        future_fine_rotation += -50
    ball.fine_rotation = future_fine_rotation
    ball.rotation = ball.fine_rotation // 10

    future_ball_x = ball.x + ball.x_velocity

    if future_ball_x < BALL_RADIUS or future_ball_x > GROUND_WIDTH:
        ball.x_velocity = -ball.x_velocity

    future_ball_y = ball.y + ball.y_velocity

    if future_ball_y < 0:
        ball.y_velocity = 1

    if (
        abs(ball.x - GROUND_HALF_WIDTH) < NET_PILLAR_HALF_WIDTH
        and ball.y > NET_PILLAR_TOP_TOP_Y_COORD
    ):
        if ball.y <= NET_PILLAR_HALF_WIDTH:
            if ball.y_velocity > 0:
                ball.y_velocity = -ball.y_velocity
        else:
            if ball.x < GROUND_HALF_WIDTH:
                ball.x_velocity = -abs(ball.x_velocity)
            else:
                ball.x_velocity = abs(ball.x_velocity)

    future_ball_y = ball.y + ball.y_velocity

    if future_ball_y > BALL_TOUCHING_GROUND_Y_COORD:
        ball.sound["ball_touches_ground"] = True

        ball.y_velocity = -ball.y_velocity
        ball.punch_effect_x = ball.x
        ball.y = BALL_TOUCHING_GROUND_Y_COORD
        ball.punch_effect_radius = BALL_RADIUS
        ball.punch_effect_y = BALL_TOUCHING_GROUND_Y_COORD + BALL_RADIUS
        return True
    ball.y = future_ball_x
    ball.x = ball.x + ball.x_velocity
    ball.y_velocity += 1

    return False


def process_player_movement_and_set_player_position(
    player: Player, user_input: PikaUserInput, the_other_player: Player, ball: Ball
):
    if player.is_computer:
        let_computer_decide_user_input(player, ball, the_other_player, user_input)

    if player.state == 4:
        player.lying_down_duration_left += -1
        if player.lying_down_duration_left < -1:
            player.state = 0
        return

    player_velocity_x = 0
    if player.state < 5:
        if player.state < 3:
            player_velocity_x = user_input.x_direction * 6
        else:
            player_velocity_x = player.diving_direction * 8

    future_player_x = player.x + player_velocity_x
    player.x = future_player_x

    if not player.is_player2:
        if future_player_x < PLAYER_HALF_LENGTH:
            player.x = PLAYER_HALF_LENGTH
        elif future_player_x > GROUND_HALF_WIDTH - PLAYER_HALF_LENGTH:
            player.x = GROUND_HALF_WIDTH - PLAYER_HALF_LENGTH
    else:
        if future_player_x < GROUND_HALF_WIDTH + PLAYER_HALF_LENGTH:
            player.x = GROUND_HALF_WIDTH + PLAYER_HALF_LENGTH
        elif future_player_x > GROUND_WIDTH - PLAYER_HALF_LENGTH:
            player.x = GROUND_WIDTH - PLAYER_HALF_LENGTH

    # jump
    if (
        player.state < 3
        and user_input.y_direction == -1
        and player.y == PLAYER_TOUCHING_GROUND_Y_COORD
    ):
        player.y_velocity = -16
        player.state = 1
        player.frame_number = 0

        player.sound["chu"] = True

    # gravity
    future_player_y = player.y + player.y_velocity
    player.y = future_player_y
    if future_player_y < PLAYER_TOUCHING_GROUND_Y_COORD:
        player.y_velocity += 1
    elif future_player_y > PLAYER_TOUCHING_GROUND_Y_COORD:
        player.y_velocity = 0
        player.y = PLAYER_TOUCHING_GROUND_Y_COORD
        player.frame_number = 0
        if player.state == 3:
            player.state = 4
            player.frame_number = 0
            player.lying_down_duration_left = 3
        else:
            player.state = 0

    if user_input.power_hit == 1:
        if player.state == 1:
            player.delay_before_next_frame = 5
            player.frame_number = 0
            player.state = 2
            player.sound["pika"] = True
        elif player.state == 0 and user_input.x_direction != 0:
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
            future_frame_number = (
                player.frame_number + player.normal_status_arm_swing_direction
            )
            if future_frame_number < 0 or future_frame_number > 4:
                player.normal_status_arm_swing_direction = (
                    -player.normal_status_arm_swing_direction
                )
            player.frame_number = (
                player.frame_number + player.normal_status_arm_swing_direction
            )

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
    if player.game_ended and player.frame_number < 4:
        player.delay_before_next_frame += 1
        if player.delay_before_next_frame > 4:
            player.delay_before_next_frame = 0
            player.frame_number += 1


def calculate_expected_landing_point_x_for(ball: Ball):
    copy_ball = {
        "x": ball.x,
        "y": ball.y,
        "x_velocity": ball.x_velocity,
        "y_velocity": ball.y_velocity,
    }
    loop_counter = 0
    while True:
        loop_counter += 1

        future_copy_ball_x = copy_ball["x_velocity"] + copy_ball["x"]
        if future_copy_ball_x < BALL_RADIUS or future_copy_ball_x > GROUND_WIDTH:
            copy_ball["x_velocity"] = -copy_ball["x_velocity"]
        if copy_ball["y"] + copy_ball["y_velocity"] < 0:
            copy_ball["y_velocity"] = 1

        if (
            abs(copy_ball["x"] - GROUND_HALF_WIDTH) < NET_PILLAR_HALF_WIDTH
            and copy_ball["y"] > NET_PILLAR_TOP_TOP_Y_COORD
        ):
            if copy_ball["y"] < NET_PILLAR_TOP_BOTTOM_Y_COORD:
                copy_ball["y_velocity"] = -copy_ball["y_velocity"]
        else:
            if copy_ball["x"] < GROUND_HALF_WIDTH:
                copy_ball["x_velocity"] = -abs(copy_ball["x_velocity"])
            else:
                copy_ball["x_velocity"] = abs(copy_ball["x_velocity"])

        copy_ball["y"] = copy_ball["y"] + copy_ball["y_velocity"]

        if (
            copy_ball["y"] > BALL_TOUCHING_GROUND_Y_COORD
            or loop_counter >= INFINITE_LOOP_LIMIT
        ):
            break

        copy_ball["x"] = copy_ball["x"] + copy_ball["x_velocity"]
        copy_ball["y_velocity"] += 1
    ball.expected_landing_point_x = copy_ball["x"]


def let_computer_decide_user_input(
    player: Player, ball: Ball, the_other_player: Player, user_input: PikaUserInput
):
    user_input.x_direction = 0
    user_input.y_direction = 0
    user_input.power_hit = 0

    virtual_expected_landing_point_x = ball.expected_landing_point_x
    if abs(ball.x - player.x) > 100 and abs(
        ball.x_velocity < player.computer_boldness + 5
    ):
        left_boundary = int(player.is_player2) * GROUND_HALF_WIDTH
        if (
            ball.expected_landing_point_x <= left_boundary
            or ball.expected_landing_point_x
            >= int(player.is_player2) * GROUND_WIDTH + GROUND_HALF_WIDTH
        ) and player.computer_where_to_stand_by == 0:
            virtual_expected_landing_point_x = left_boundary + (GROUND_HALF_WIDTH // 2)

    if abs(virtual_expected_landing_point_x - player.x) > player.computer_boldness + 8:
        if player.x < virtual_expected_landing_point_x:
            user_input.x_direction = 1
        else:
            user_input.x_direction = -1
    elif rand() % 20 == 0:
        player.computer_where_to_stand_by = rand() % 2

    if player.state == 0:
        if (
            abs(ball.x_velocity) < player.computer_boldness + 3
            and abs(ball.x - player.x) < PLAYER_HALF_LENGTH
            and ball.y > -36
            and ball.y < 10 * player.computer_boldness + 84
            and ball.y_velocity
        ):
            user_input.y_direction = -1

        left_boundary = int(player.is_player2) * GROUND_HALF_WIDTH
        right_boundary = (int(player.is_player2) + 1) * GROUND_HALF_WIDTH

        if (
            ball.expected_landing_point_x > left_boundary
            and ball.expected_landing_point_x > right_boundary
            and abs(ball.x - player.x) > player.computer_boldness * 5 + PLAYER_LENGTH
            and ball.x > left_boundary
            and ball.x < right_boundary
            and ball.y > 174
        ):
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
            will_input_power_hit = decide_whether_input_power_hit(
                player, ball, the_other_player, user_input
            )
            if will_input_power_hit:
                user_input.power_hit = 1
                if (
                    abs(the_other_player.x - player.x) < 80
                    and user_input.y_direction != -1
                ):
                    user_input.y_direction = -1


def decide_whether_input_power_hit(
    player: Player, ball: Ball, the_other_player: Player, user_input: PikaUserInput
):
    if rand() % 2 == 0:
        for x_direction in range(1, -1, -1):
            for y_direction in range(-1, 2, 1):
                expected_landing_point_x = expected_landing_point_x_when_power_hit(
                    x_direction, y_direction, ball
                )
                if (
                    expected_landing_point_x
                    <= int(player.is_player2) * GROUND_HALF_WIDTH
                    or expected_landing_point_x
                    >= int(player.is_player2) * GROUND_WIDTH + GROUND_HALF_WIDTH
                    and abs(expected_landing_point_x - the_other_player.x)
                    > PLAYER_LENGTH
                ):
                    user_input.x_direction = x_direction
                    user_input.y_direction = y_direction
                    return True
    else:
        for x_direction in range(1, -1, -1):
            for y_direction in range(1, -2, -1):
                expected_landing_point_x = expected_landing_point_x_when_power_hit(
                    x_direction, y_direction, ball
                )
                if (
                    expected_landing_point_x
                    <= int(player.is_player2) * GROUND_HALF_WIDTH
                    or expected_landing_point_x
                    >= int(player.is_player2) * GROUND_WIDTH + GROUND_HALF_WIDTH
                    and abs(expected_landing_point_x - the_other_player.x)
                    > PLAYER_LENGTH
                ):
                    user_input.x_direction = x_direction
                    user_input.y_direction = y_direction
                    return True
    return True


def expected_landing_point_x_when_power_hit(
    user_input_x_direction, user_input_y_direction, ball: Ball
) -> int:
    copy_ball = {
        "x": ball.x,
        "y": ball.y,
        "x_velocity": ball.x_velocity,
        "y_velocity": ball.y_velocity,
    }
    if copy_ball["x"] < GROUND_HALF_WIDTH:
        copy_ball["x_velocity"] = (abs(user_input_x_direction) + 1) * 10
    else:
        copy_ball["x_velocity"] = -(abs(user_input_x_direction) + 1) * 10
    copy_ball["y_velocity"] = abs(copy_ball["y_velocity"] * user_input_y_direction * 2)

    loop_counter = 0
    while True:
        loop_counter += 1

        future_copy_ball_x = copy_ball["x"] + copy_ball["x_velocity"]
        if future_copy_ball_x < BALL_RADIUS or future_copy_ball_x > GROUND_WIDTH:
            copy_ball["x_velocity"] = -copy_ball["x_velocity"]
        if copy_ball["y"] + copy_ball["y_velocity"] < 0:
            copy_ball["y_velocity"] = 1
        if (
            abs(copy_ball["x"] - GROUND_HALF_WIDTH) < NET_PILLAR_HALF_WIDTH
            and copy_ball["y"] > NET_PILLAR_TOP_TOP_Y_COORD
        ):
            if copy_ball["y_velocity"] > 0:
                copy_ball["y_velocity"] = -copy_ball["y_velocity"]

        copy_ball["y"] = copy_ball["y"] + copy_ball["y_velocity"]
        if (
            copy_ball.y > BALL_TOUCHING_GROUND_Y_COORD
            or loop_counter >= INFINITE_LOOP_LIMIT
        ):
            return copy_ball["x"]
        copy_ball["x"] = copy_ball["x"] + copy_ball["x_velocity"]
        copy_ball["y_velocity"] += 1
