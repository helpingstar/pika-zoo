"""
All of the code for pika-zoo was written based on https://github.com/gorisanson/pikachu-volleyball

hs) For multithreaded optimization and experimental reproducibility, I used a generator for random number extraction.
    As an argument to the class/function, the environment's self.np_random is used as a generator.
"""

import numpy as np
from typing import List


class Cloud:
    """class represents a cloud"""

    def __init__(self, np_random: np.random.Generator) -> None:
        self.top_left_point_x = -68 + (np_random.integers(0, 432 + 68))
        self.top_left_point_y = np_random.integers(0, 152)
        self.top_left_point_x_velocity = 1 + np_random.integers(0, 2)
        self.size_diff_turn_number = np_random.integers(0, 11)

    @property
    def size_diff(self):
        return 5 - abs(self.size_diff_turn_number - 5)

    @property
    def sprite_top_left_point_x(self):
        return self.top_left_point_x - self.size_diff

    @property
    def sprite_top_left_point_y(self):
        return self.top_left_point_y - self.size_diff

    @property
    def sprite_width(self):
        return 48 + 2 * self.size_diff

    @property
    def sprite_height(self):
        return 24 + 2 * self.size_diff


class Wave:
    """Class representing wave"""

    def __init__(self) -> None:
        self.vertical_coord = 0
        self.vertical_coord_velocity = 2
        self.y_coords = []
        for i in range(432 // 16):
            self.y_coords.append(314)


def cloud_and_wave_engine(cloud_array: List[Cloud], wave: Wave, np_random: np.random.Generator):
    """Move clouds and wave

    Args:
        cloud_array (List[Cloud]): cloud array
        wave (Wave): wave
        np_random (np.random.Generator): random number Generator, ex) `self.np_random`
    """
    for i in range(10):
        cloud_array[i].top_left_point_x += cloud_array[i].top_left_point_x_velocity
        if cloud_array[i].top_left_point_x > 432:
            cloud_array[i].top_left_point_x = -68
            cloud_array[i].top_left_point_y = np_random.integers(0, 152)
            cloud_array[i].top_left_point_x_velocity = 1 + np_random.integers(0, 2)
        cloud_array[i].size_diff_turn_number = (cloud_array[i].size_diff_turn_number + 1) % 11

    wave.vertical_coord += wave.vertical_coord_velocity
    if wave.vertical_coord > 32:
        wave.vertical_coord = 32
        wave.vertical_coord_velocity = -1
    elif wave.vertical_coord < 0 and wave.vertical_coord_velocity < 0:
        wave.vertical_coord_velocity = 2
        wave.vertical_coord = -(np_random.integers(0, 40))

    for i in range(432 // 16):
        wave.y_coords[i] = 314 - wave.vertical_coord + np_random.integers(0, 3)
