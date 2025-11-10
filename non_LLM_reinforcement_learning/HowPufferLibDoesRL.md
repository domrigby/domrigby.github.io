# Puffing Up Reinforcement Learning

Date read: 9th November 2025

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
	1. **No dynamic allocations**: all allocated at initialisation.
	2. **No observation copies**: write directly to the buffers
	3. **Aggressive caching**: reuse as much data and memory as possible
	4. **Very asynchronous**: inspired by **EnvPool**

* Hyperparameter tuning:
	* Aim: find **Pareto efficient** hyperparameters between **cost and performance**
	* Created modified version of **CARBS algorithm called Protein**
	* CARBS: 
		* Randomly generate Pareto frontier of hyperparameters
		* Mutates them and then train **Gaussian processes** to predict their scores from the HPs.
		* Uses the GPs to identify strong new HPs
	* Issues: Bias towards Pareto front and susceptible to noise

* RL tips:
	1. **Results > methods**: make sure experiments used fast environments or results can be purely noise
	2. PPO is the normally the goto
	3. PPO hyperparameter tips:
		* Sweep learning rate
		* Gamma and lambda: ask yourself how long in the future matters in this game? 
			* Gamma = 1 - 1/ (number of steps in that time)
			* Lambda: bit less than gamma
		* Clipping not too low or experiment will be 'on-rails'
	4. Perform hyperparameter searches on reward scalings
	5. **Always use white-box RL where you can: everything you make will break**










