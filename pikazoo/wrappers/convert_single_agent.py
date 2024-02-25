from pettingzoo.utils import BaseParallelWrapper
from pettingzoo.utils.env import ParallelEnv


class ConvertSingleAgent(BaseParallelWrapper):
    def __init__(self, env: ParallelEnv, side: str):
        super().__init__(env)
        assert side in ("player_1", "player_2")
        self.side = side
        self.other_side = "player_1" if side == "player_2" else "player_2"

    def reset(self, seed=None, options=None):
        obs, infos = super().reset(seed=seed, options=options)
        return obs[self.side], infos[self.side]

    def step(self, action):
        actions = {
            self.side: action,
            self.other_side: self.action_space(self.other_side).sample(),
        }
        obs, rews, terminateds, truncateds, infos = super().step(actions)
        return (
            obs[self.side],
            rews[self.side],
            terminateds[self.side],
            truncateds[self.side],
            infos[self.side],
        )
