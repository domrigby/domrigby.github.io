# Synergizing Quality-Diversity with Descriptor-Conditioned Reinforcement Learning

Date: 15th July 2025

[arXiv link](https://arxiv.org/abs/2401.08632)

## Key Points
* Aims for quality diversity, like MAP-Elites. QD aims to provide diverse range of solutions across a **descriptor space** which is high level description of a policy.
* QD algos such as MAP-Elites tend use evolutionary algorithms which don't expand to high dimensions well. This paper aims to unite QD and reinforcement learning.
* Their algorithm **DCRL-MAP-Elites** uses a **descriptor conditioned** actor and critic. The **policy and target have descriptors**. I.e. it finds different ways to solve different goals. The policy is conditioned to meet a certain target.
* Descriptor are user-defined **desribe the observed trajectories of policies** and allow both actor and critic to become learn different niches for certain areas of the search
* Still maintains a population of policies in which variation is applied in three following ways:
  1. Genetic algorithm mutation
  2. Descriptor-condition policy gradients
  3. Actor injection: they train a generative model to create new policies
* Uses usual MAP-Elites algorithm to ensure that we get the best policy per area of the search space.