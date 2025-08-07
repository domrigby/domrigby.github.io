# GOAL: A Generalist Combinatorial Optimization Agent Learner

Date read: 6th August 2025

[arXiv link](https://arxiv.org/abs/2406.15079)

## Key Points
* Theorises that combinatorial optimisation problems have shared features and skills, therefore a generalist agent could 
be created which is _decent_ solution to all of them and could be finetuned to be a very good agent.
* Current methods to solve these problems rely on expert designed, problem specific heuristics to guide search... could this be replaced
by a learnt function?
* Method:
  * Trains transformer on set of combinatorial optimisation problem (travelling salesman, routing etc).
  * This is trained via **supervised learning** on expert trajectories from solved problems. Problems therefore have to be small enough 
  that they could already be solved.
  * Each problem has its own input and output layers,
  * Input is a graph and then task identifier
  * Output: rating over nodes (e.g. next to visit in routing)
  * Architecture: does **mixed attention**: takes separate Qs and Ks of nodes and edges and then combines them before self-attention