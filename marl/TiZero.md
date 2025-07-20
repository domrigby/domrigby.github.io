# TiZero: Mastering Multi-Agent Football with Curriculum Learning and Self-Play

Date: 20th July 2025

[arXiv link](https://arxiv.org/abs/2302.07515)

## Key Points
* Aims:
  * TiZero is an algorithm which learns to play 11 vs 11 football in Google Research Football from scratch.
  * Football is a very hard problem: it includes multi-agent coordination, both long and short-term planning and non-transivity.
  It is both cooperative and competitive. Rewards for agents without the ball for example are _extremely_ sparse.
* How:
  * Tackles large search space using a **curriculum learning**. The curriculum is set using the speed of the opponent, 
  position of ball on the pitch and the position of the players (ball close to our goal = hard, close to their goal = easy).
  * Tackles non-transivity, competitive and cooperativeness using **self-play learning**. They create a probability distribution 
  over next agent scales linearly which biases agents which are more difficult. 
* Interesting implementation details:
  * They implement a tweak of MAPPO called Joint-ratio Policy Optimisation. This optimised a joint policy which is the product of all 
  the individual policies. This supposedly achieves better coordination.
  * Use different observation for policy (individual) and critic (team).
  * As mentioned above: utilises a curriculum.
  * Two stage self-play:
    1. Challenge self-play: plays most recent agent 80% of the time, 20% an older agent
    2. Generalise self-play: plays against entire opponent pool. Opponent is sample using probability distribution given 
    in section 4.3
  * Utilises an LSTM to learn from earlier observations in episode. This gives insight about opponent strategy and helps
  with long term planning.
  * Masks out actions which aren't possible to reduce search space.
  * Utilises reward shaping but ensure the game is still zero-sum.
  * **Processes each feature of the game with a separate MLP** and then processes the results using a **player-conditioned** 
  LSTM. This saves a bunch of compute as well as you can reuse the observations between agents.
  * Random actions at start to create varied scenarios
  * **Utilises actor-learner distributed framework**