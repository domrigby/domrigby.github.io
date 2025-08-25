# Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning

Date read: 21st August 2025

[ArXiv Link](https://arxiv.org/pdf/2109.11978)

## Key Points
* Aims to train continuous control RL in minutes using massively parallelised GPU environments. It achieves this by training 
walking policies in a matter of minutes.
* Full on GPU training (no comms overhead)
* For robotics: they suggest maximising number of robots and therefore reducing the length of each rollout (but not too much, they found that
the algorithm struggles with less than 25 steps).
* Make batch-size as large as possible to denoise the gradients (data is abundant )
* Episodes span multiple-policy updates

* Implementation details:
  * Auto-curriculum to ensure tasks are correct level of difficulty. Begins with flat surfaces for robot locomotion an
  * Uses many robots in one environment to save memory
  * Do not reset envs after rollouts
  * Ensures truncated states are bootstrapped and not set to 0 (no time given in observation)