# Hierarchical Reasoning Model

Date read: 4th August 2025

[arXiv link](https://arxiv.org/abs/2506.21734)

## Key Points
* Introduces a brain inspired hierarchical reasoning model
* Brain inspired?
  * Two levels of planning:
    1. High level (every T time steps) does abstract planning
    2. Low level (every time step and conditioned on the high level planner) does execution of high level plan.
  * Plans in abstract latent space rather than in words (like brain)
* Avoids Back Propagation Through Time (BPTT) by online optimising the final step of the rollout (assumes this is a stable point and approximates the whole 
gradient, check the page 6 for more details)
* Avoids early convergence by constantly reconditioning the lower level RNN from the high level RNN once it has converged.
* DQN decides whether we stop rolling out... allows for system 1 and 2 thinking and **scalable test time computer**
  * Exploration is done by picking a minimum time to end as the beginning of each episode