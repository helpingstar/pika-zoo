from pikazoo import pikazoo_v0
from pettingzoo.test import parallel_api_test


def test_parallel_api_test():
    env = pikazoo_v0.env()
    parallel_api_test(env, num_cycles=1_000_000)
