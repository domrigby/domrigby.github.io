# Efficiently Quantifying Individual Agent Importance in Cooperative MARL

Date read: 5th September 2025

[arXiv link](https://arxiv.org/pdf/2312.08466)

## Key Points
* Credit assignment in MARL is important for explainability, training and for robustness. 
* Credit assignment in this context means calculating which agents actions effected the reward for the whole team.
* Shapley values provide a method for calculating how much a single agent's actions contributed to the 'coalition reward',
but they are computationally expensive to calculate (scales exponentially with number of agents) and require that we can 
reset the environment to calculate counterfactuals.
* Paper introduces **Agent Importance**, a metric or quantifying the contribution of an agent which **scales linearly with 
number of agents**. How?
	* Agent Importance is the difference in reward between when the agent did take the action and when it didn't
	* It is calculated by creating N copies of the environment and adding a NOOP action to each agent, such that we can calculate the reward if they didn't take the action.
	* It is calculated per step rather than per episode.
* They also found that algorithms with more evenly distributed agent importance tend to perform better than ones which focus importance
on one or two agents (makes sense).