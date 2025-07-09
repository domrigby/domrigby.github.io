# Illustrated Comparison of Different Distributed Versions of PPO

Date: 7th July 2025

[Blog post](https://medium.com/@maxbrenner-ai/illustrated-comparison-of-different-distributed-versions-of-ppo-18602d27c657)

# Key Points
* Discussed a variety of different ways to distributed PPO.
* This has the challenge that the lag can make the data off-policy
* Discussed themes:
  1. **Synchronous**: waits for all agents to calculate their respective gradients before doing a weights update.
  2. **Asynchronous**: doesn't wait.
  3. **Centralised**: single server does all gradient accumulation and weights updates.
  4. **Decentralised**: all share gradients (all-reduce) but have their own model.
* Discussed algos:
  1. **Asynchronous PPO**: multiple CPU workers collect rollouts to be set to GPU for gradient calculation. (not asynchronous w.r.t gradients)
  2. **Distributed PPO**: mutliple actors, single parameter server. They send rollouts and the server calculates gradients and then applies weight updates.
  3. **Decentralised Distributed PPO**: no parameter server, decentralised updates.
  4. **Resource Flexible Distributed PPO**: workers can either be used for rollout collection of gradient calculation.