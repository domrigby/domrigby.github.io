# Winner Takes It All: Training Performant RL Populations for Combinatorial Optimization

Date read: 27th July 2025

[arXiv link](https://arxiv.org/abs/2210.03475)

## Key Points
* Introduces **Poppy**: solves combinatorial optimisation problems using a **population of RL agents** (uses whole 
population to solve the problem).
* Why? Single agent RL struggles to search across a range of different possible solutions
* Populations allow the agent to learn a diverse **set of solutions**
* Populations for CO have three problems however:
  1. Compute: training multiple agents is computationally heavy
  2. Population should have complementary policies which lead to different solutions
  3. Should not include any handcrafted notion of diversity (many algorithms do, e.g. MAP Elites)
* Poppy answers these problems by:
  1. Shares feature extraction across whole population
  2. Introduces RL objective which encourage agents to specialise on subset of solution.
* How: **only trains best performing agent** on each problem. Optimises max reward across all policies. This encourages 
diversity as it is unlikely that any one agent is the best on all objectives.
* NOTE: uses Jax for the environment