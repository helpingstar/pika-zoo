import math
import random

custom_rng = None


def rand():
    global custom_rng
    if custom_rng is None:
        return random.randint(0, 32767)
    return int(32768 * custom_rng())


def set_custom_rng(rng):
    global custom_rng
    custom_rng = rng
