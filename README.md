# pika-zoo

**All of the code for pika-zoo was written based on [gorisanson/pikachu-volleyball](https://github.com/gorisanson/pikachu-volleyball).**

You can play the game at the site below.

* https://gorisanson.github.io/pikachu-volleyball/en/
* P2P Online : https://gorisanson.github.io/pikachu-volleyball-p2p-online/en/


| Import             | `from pikazoo import pikazoo_v0`   |
|--------------------|----------------------------------|
| Actions            | Discrete                         |
| Parallel API       | Yes                              |
| Manual Control     | No                               |
| Agents             | agents= ['player_1', 'player_2'] |
| Agents             | 2                                |
| Action Shape       | player_1 : (6,), player_2 : (6,) |
| Action Values      | 0 or 1                           |
| Observation Shape  | (21,)                            |
| Observation Values | [-inf,inf]                       |

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

| Index | Description                                  | min | max |
|-------|----------------------------------------------|-----|-----|
| 0     | X position of player1                        | 32  | 184 |
| 1     | Y position of player1                        | 108 | 244 |
| 2     | Y Velocity of player1                        | -15 | 16  |
| 3     | player1's remaining   duration of lying down | -2  | 3   |
| 4     | player1 is collision   with ball             | 0   | 1   |
| 5     | state of player1                             | 0   | 6   |
| 6     | X position of player2                        | 248 | 400 |
| 7     | Y position of player2                        | 108 | 244 |
| 8     | Y Velocity of player2                        | -15 | 16  |
| 9     | player2's remaining   duration of lying down | -2  | 3   |
| 10    | player2 is collision   with ball             | 0   | 1   |
| 11    | state of player2                             | 0   | 6   |
| 12    | X position of ball                           | 20  | 432 |
| 13    | Y position of ball                           | 0   | 252 |
| 14    | Previous X position   of ball                | 32  | 432 |
| 15    | Previous Y position   of ball                | 0   | 252 |
| 16    | Previous previous X   position of ball       | 32  | 432 |
| 17    | Previous previous Y   position of ball       | 0   | 252 |
| 18    | X Velocity of ball                           | -20 | 20  |
| 19    | Y Velocity of ball                           | -inf | inf |
| 20    | If the ball is in   POWER HIT status         | 0   | 1   |

I don't know exactly what the maximum and minimum values of the ball's y velocity are, so I used `inf`. The maximum and minimum values I observed are `-123` and `124`.

<!-- TODO: Install, Sample Code -->