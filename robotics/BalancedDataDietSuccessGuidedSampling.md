# A Balanced Data Diet: Addressing the Exploration Bottleneck in Mega-Scale RL for Robot Control

**Date:** 20th October 2026

[arXiv Link](https://arxiv.org/abs/2610.12465) | [PDF](https://arxiv.org/pdf/2610.12465) | [Project Page](https://sgs-rl.github.io/)

Related: [AlphaNIST: Emergent Assembly Skills from Binary Task Rewards](AlphaNISTBinaryRewardAssembly.md) | [Prioritised Level Replay](../open_endedness_and_auto_curriculums/PrioritisedLevelReplay.md)

## Key Points
* RL pipelines often rely of heavy per task engineering (reward, resets, demos etc)
* **Success Guide Sampling aims to be a generic method for sampling edge of capability reset states**. This has shown itself to be an effective method for solving complex RL tasks.
    * **Scaling up with uniform sampling of states means much of the compute is spent on solved or impossible tasks**
    * Programmatically generate thousands of high quality reset states
    * Non-parametric success rate estimate:: record last N attempts success
    * Utilise **P*(1-P)** as a learnability score for (start state, goal, env config)
    * Utilise beta shaped kernel to generate softmax logit
    * $$ l_{i} = log((p + \epsilon)^{kt} (1 - p + \epsilon)^{k(1-t)}) $$
    * $$ P(state_{i}) = softmax(l_{i}, T) $$
    * t = target success rate, k = concentration (sharpness of peak at t) T = softmax temperaature
    * They use k = 1 and T = 2, t = 0.5 for manipulation and k = 5, T = 2, t= 0.66 for locomotion, Higher k means probability is pushed towards learnability target.

