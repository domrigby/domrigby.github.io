# Puffing Up Reinforcement Learning

Date read: 10th November 2025

[Blog link](https://puffer.ai/blog.html#post-11)

## Key Points
* Algorithm changes:
	* Switches from **Adam to Muon** optimiser: solved problems ~30% faster.
	* Utilise **cosine annealing for learning rates**
	* **Data filtering**:got rid of uninformative trajectories and sampled them using prioritsed experience replay
	* **Mixed PPO and IMPALA**:
		* PPO has clipping and GAE, IMPALA corrects off-policy-ness with V-Trace
		* Implementation not truly detailed, but did mention implementations are often slow so they made a custom CUDA kernel to make it faster.

* Performance optimisations:
	* 

* Hyperparameter tuning:
	* Aim: find **Pareto efficient** hyperparameters between **cost and performance**
	* Created modified version of **CARBS algorithm called Protein**
	* CARBS: 
		* Randomly generate Pareto frontier of hyperparameters
		* Mutates them and then train **Gaussian processes** to predict their scores from the HPs.
		* 
