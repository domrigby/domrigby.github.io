# GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning

Date read: 1st August 2025

[arXiv link](https://arxiv.org/abs/2507.19457)

## Key Points
* **Prompt optimisation can be better than RL on a single task**
* Uses **LLMs to reflect on what went wrong/well after a rollout**... LLMs act as 'intelligent improvement operators'.
Improvement is done is genetic algorithm manner (mutate and then check results, keeping performance improvements). High
performing but differing prompts also go through a merging operation to see if an even better prompt can be made.
* **Takes a Pareto-Front across the set of training tasks** to encourage diversity
* **More sample efficient than RL**: 10% improvement over GRPO in 35x less rollouts
* Large improvements after one rollout