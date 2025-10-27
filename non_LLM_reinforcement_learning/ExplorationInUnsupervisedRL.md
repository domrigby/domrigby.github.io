# Demystifying the Mechanisms Behind Emergent Exploration in Goal-Conditioned RL

Date read: 24th October 2025

[ArXiv link](https://arxiv.org/pdf/2510.14129)

## Key Points
* Paper aims to learn how unsupervised RL algorithms learn to explore

* Unsupervised RL: 
	* Agents get **no external reward**
	* Often goal conditioned: agents get a goal state they have to reach 
	* Agents learn to manipulate environment and emergent behaviour often appears

* SGCRL: Single-Goal Contrastive RL
	* Agent receives a goal state
	* Critic learns **two separate embeddings**: one for the current state and action and one for the goal
	* Result of dot product of embeddings = likelihood that action at state leads to goal state.
	* **Trained with contrastive loss**, specifically InfoNCE
	* Action is greedy sampled from the critic predictions.

* Exploration:
	* Algorithm implicitly learns to maximise chance of reaching states which look like the goal state.
	* The longer you sit anywhere without reaching the goal, the critic will decrease this value and push you away.
	* You are pushed away from already visited unsuccessful states and pulled towards successful ones