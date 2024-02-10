# pika-zoo

The original Pikachu Volleyball (対戦ぴかちゅ～　ﾋﾞｰﾁﾊﾞﾚｰ編) was developed by

* 1997 (C) SACHI SOFT / SAWAYAKAN Programmers
* 1997 (C) Satoshi Takenouchi

**All of the code for pika-zoo was written based on [gorisanson/pikachu-volleyball](https://github.com/gorisanson/pikachu-volleyball).**

You can play the game at the site below.

* https://gorisanson.github.io/pikachu-volleyball/en/
* P2P Online : https://gorisanson.github.io/pikachu-volleyball-p2p-online/en/

https://github.com/helpingstar/pika-zoo/assets/54899900/b253555e-20bd-4449-bf34-130e0d8902fa

| Import             | `from pikazoo import pikazoo_v0`         |
|--------------------|------------------------------------------|
| Actions            | Discrete                                 |
| Parallel API       | Yes                                      |
| Manual Control     | No                                       |
| Agents             | agents= ['player_1', 'player_2']         |
| Agents             | 2                                        |
| Action Shape       | player_1 : (6,) or (5,), player_2 : (5,) |
| Action Values      | 0 or 1                                   |
| Observation Shape  | (36,)                                    |
| Observation Values | [-inf,inf]                               |

## Action Space

| Index | Description                         |
|-------|-------------------------------------|
| 0     | left key                            |
| 1     | right key                           |
| 2     | up key                              |
| 3     | down key                            |
| 4     | power hit key                       |
| 5     | down rigth key (only for player1)   |

## Observation Space

| Index | Description                                  | min  | max  |
|-------|----------------------------------------------|------|------|
| 0     | X position of player1                        | 32   | 184  |
| 1     | Y position of player1                        | 108  | 244  |
| 2     | Y Velocity of player1                        | -15  | 16   |
| 3     | Diving direction of player1                  | -1   | 1    |
| 4     | Player1's remaining duration of lying down   | -2   | 3    |
| 5     | Player1 is collision with ball               | 0    | 1    |
| 6     | State of player1 (normal)                    | 0    | 1    |
| 7     | State of player1 (jumping)                   | 0    | 1    |
| 8     | State of player1 (jumping and power hitting) | 0    | 1    |
| 9     | State of player1 (diving)                    | 0    | 1    |
| 10    | State of player1 (lying down after diving)   | 0    | 1    |
| 11    | State of player1 (win)                       | 0    | 1    |
| 12    | State of player1 (lost)                      | 0    | 1    |
| 13    | X position of player2                        | 248  | 400  |
| 14    | Y position of player2                        | 108  | 244  |
| 15    | Y Velocity of player2                        | -15  | 16   |
| 16    | Diving direction of player2                  | -1   | 1    |
| 17    | Player2's remaining duration of lying down   | -2   | 3    |
| 18    | Player2 is collision with ball               | 0    | 1    |
| 19    | State of player2 (normal)                    | 0    | 1    |
| 20    | State of player2 (jumping)                   | 0    | 1    |
| 21    | State of player2 (jumping and power hitting) | 0    | 1    |
| 22    | State of player2 (diving)                    | 0    | 1    |
| 23    | State of player2 (lying down after diving)   | 0    | 1    |
| 24    | State of player2 (win)                       | 0    | 1    |
| 25    | State of player2 (lost)                      | 0    | 1    |
| 26    | X position of ball                           | 20   | 432  |
| 27    | Y position of ball                           | 0    | 252  |
| 28    | Previous X position of ball                  | 32   | 432  |
| 29    | Previous Y position of ball                  | 0    | 252  |
| 30    | Previous previous X position of ball         | 32   | 432  |
| 31    | Previous previous Y position of ball         | 0    | 252  |
| 32    | X Velocity of ball                           | -20  | 20   |
| 33    | Y Velocity of ball                           | -124 | 124  |
| 34    | If the ball is in   POWER HIT status         | 0    | 1    |
| 35    | Is player 2? (Whether you play on the right.)| 0    | 1    |

Since I do not know the exact minimum and maximum values of the ball's y velocity, I used the minimum and maximum values I observed.

## Arguments

```python
env = pikazoo_v0.env(
    use_f_key=False,
)
```

* `use_f_key` : In the original game, the F key functions to press the down direction (V key) and the right direction key (G key) simultaneously. This is unnecessary from the perspective of reinforcement learning. Considering the symmetry of the game and for vectorizing the environment, it is advisable to set this option to `False`. When this option is set to `False`, the action_space of player1 changes from [`MultiBinary(6)`](https://gymnasium.farama.org/api/spaces/fundamental/#multibinary) to [`MultiBinary(5)`](https://gymnasium.farama.org/api/spaces/fundamental/#multibinary). Related information can be found in the link below.
  * https://github.com/gorisanson/pikachu-volleyball/issues/3

<!-- TODO: Install, Sample Code -->
