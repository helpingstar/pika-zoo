from pikazoo import pikazoo_v0
from pettingzoo.test import parallel_api_test


if __name__ == "__main__":
    env = pikazoo_v0.env()
    parallel_api_test(env, num_cycles=1_000_000)
