# Polychromic Objectives for Reinforcement Learning

Date read: 2nd October 2025

[ArXiv link](https://www.arxiv.org/abs/2509.25424)

## Key Points
* RL Fine-Tuning:
  * Suffers from mode collapse. This hinders exploration and limits enhancement of capabilities (just focuses on capabilities already in the model).
  * Often performs poorly on high k pass@k compared to the base model (has not yet suffered mode collapse)
  * Regularisation promotes token level variation rather than trajectory level.
* Paper introduces **Set Reinforcement Learning**:
  * Optimises over a **set** of trajectories rather than just one
  * Encourage diversity of trajectories as part of objective
  * Uses **same advantage value across whole set**, i.e. it optimises each trajectory based on how the whole set performed. If whole set did well, then encourage.
* Implementation details:
  * Value function is sum of the reward of all possible trajectories from that state
  * The above is approximate via **vine sampling**: sample trajectories -> sample states -> sample N trajectories from those states
  * Where N trajectories are sample... SRL is performed... else just normal PPO