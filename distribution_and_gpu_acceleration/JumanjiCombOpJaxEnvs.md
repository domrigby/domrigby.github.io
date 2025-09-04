# Jumanji: a Diverse Suite of Scalable Reinforcement Learning Environments in JAX

Date read: 4th September 2025

[arXiv link](https://arxiv.org/abs/2306.09884)

## Key Points
* Jumanji is a set of highly parallelisable, GPU based NP-hard combinatorial optimisation environments.
* Aims to create a set of open-source environments which are **closely related to real life industrial applications**.
* Jumanji aims to make RL more:
	1. Fast: hardware accelerated on GPU **or** TPU. Allowing for **rapid iteration** of RL algorithms
	2. Flexible: easy customisation to mimic real world situations. This is through custom initialisations and reward functions.
	3. Scalable: set arbitrary difficulty to allow 'faithful representationn of real world challenges'
* Speed: maximum throughput is achieved by:
	1. Training and environment both on GPU and JIT compiled
	2. Massive parallelisation due to hardware
	3. Multi-GPU execution using JAX's pmap function.
* Section 3.2 provides an interesting oversight of how they designed their environments.
