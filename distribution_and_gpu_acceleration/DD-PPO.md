#  DD-PPO: LEARNING NEAR-PERFECT POINTGOAL NAVIGATORS FROM 2.5 BILLION FRAMES

Date read: 23rd August 2025

[ArXiv link](https://arxiv.org/pdf/1911.00357)

## Key Points
* Scales up PPO RL almost linearly to 128 GPUs (107x speed up) for navigation
* Argues for **decentralised training** with no lockstep, rather than single learner, multi-actor centralised PPO
* Why? Synchronous PPO has to wait for the slowest worker, sending data back to main server can be memory intensive when 
using images as an observation.
* Each worker runs the environment independently, before calculating their own gradients which are then broadcast to the other agents (synchronous).
* Uses **preemption threshold** of around 60% which stops running the remaining environments if 60% have declared their results. This means
they have to share memory. This really helps the scaling as it prevents stragglers wasting too much time.
* **Shows massive performance increase with scale** which is also trained more quickly

