# DISCOVER: Automated Curricula for Sparse-Reward Reinforcement Learning

Date read: 26th July 2025

[arXiv link](https://arxiv.org/abs/2505.19850)

## Key Points
* Aims to guide RL agents in sparse reward scenarios by giving the agent sub-goals to try and achieve
* The idea is that sequentially achieving these sub-goals results in optimised performance
* These sub-goals are chosen according to three criteria: **novelty, achievability and relevance**. I.e. it creates
an auto-curriculum which gradually increases difficulty of new tasks.
* How?
  * Actor and critic are **conditioned**: i.e. they take a goal state as an input.
  * Trains an **ensemble of critics**
  * Sub-goals (g) are put through the ensemble and then chosen according to:
    * High V(s0, g) means tasks is likely achievable
    * High std(s0, g) means this is not reliable and therefore likely novel
    * High V(g, g*) means the sub-goal g is close to the target goal g*