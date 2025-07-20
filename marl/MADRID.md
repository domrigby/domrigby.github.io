# Multi-Agent Diagnostics for Robustness via Illuminated  (MADRID)

Date: 19th July 2025

[arXiv Link](https://arxiv.org/abs/2401.13460)

## Key Points
* Aims: to adversarially find scenarios in which multi-agent reinforcement learning policies fail.
* Why?: MARL algorithms can lack robustness and overfit to specific scenarios seen during training.
* How?: 
  * Uses TiZero as test-bed (algorithm which can learn 11vs11 football)
  * Implements **MAP-Elites on the scenario search space**, with **regret as the fitness**.
  * Regret: difference between current policy and optimal policy's reward. We don't know the optimal policy so we use 
    a set of policies to get a lower bound. These reference policies tend to be old checkpoints and/or heuristics.
  * MAP-Elites descriptor: x and y start position of the ball. This gives a range of scenarios across the pitch
* Results: identified a diverse set of, sometimes unexpected, failure modes in TiZero football.