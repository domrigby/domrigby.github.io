# Eurekaverse: Environment Curriculum Generation via Large Language Models

Date read: 23rd July 2025

[arXiv link](https://arxiv.org/abs/2411.01775)

## Key Points
* Uses LLMs to create a curriculum for a quadruped robotic dog to tackle obstacles.
* LLMs are used as way of tackling the problem of reliance on low dimensionality for environments.
### How?
  * LLM designs height grids for the agent to navigate through **generating code**.
  * **Agent-environment co-evolution**: follows process of training agents and then generating code... and then back to 
    start.
### Method:
  * Generate original set of examples via a text description of the game in the prompt + some example code
  * Hot sample of the LLMs outputs.
  * Train a set of agents, each on a subset of these levels.
  * Get best performing agent on **all levels**... the levels used to train this agent must be strong.
  * Give each of these tasks to the LLM with the task description and ask for it to make the task harder.
  * Repeat process but with set of the best performing agent