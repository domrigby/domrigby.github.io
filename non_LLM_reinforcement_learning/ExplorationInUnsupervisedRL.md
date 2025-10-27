# Demistifying the Mechanisms Behind Emergent Exploration in Goal-Conditioned RL

Date read: 24th October 2025

[ArXiv link]()

## Key Points
* Paper aims to learn how unsupervised RL algorithms learn to explore

* Unsupervised RL: 
	* Agents get **no external reward**
	* Often goal conditioned: agents get a goal state they have to reach 
	* Agents learn to manipulate environment and emergent behaviour often appears

* SGCRL: Single-Goal Contrastive RL
	* Agent receives a goal state
	* Critic gets a state and action and learned **two separate embeddings**
	* Result of dot product of embeddings = likelihood that action at state leads to goal state.
	* **Trained with contrastive loss**, specifically InfoNCE
	* Action is greedy sampled from the critic predictions