# POMO: Policy Optimisation with Multiple Optima for Reinforcement Learning

Date read: 16th October 2025

[Paper link]()

## Key Points
* Aim:
	* RL allows us to train neural heuristics which can find near-optimal solutions, without advanced designer domain knowledge
	* POMO encourages a diverse set of optimal solutions to be found 
	* Many DRL based CO solvers are heavily reliant on their first action -> overfit to a small subset of the solutions.

* Why?
	* Deep learning at this point had replaced hand designed features in computer vision and NLP via supervised learning
	* Difficult to do supervised learning with combinatorial optimisation as **it is impossible to get instant access to optimal solutions**
	* We can however get almost instant scores for solutions... fits RL well!

* POMO:	
	* Address issue of overfitting to a small subset of solutions:
		* Most solvers spawn their solver with a START token and it learns a criteria to pick the first node, forcing bias toward certain opening moves.
		* POMO instantiates with a set of chosen N start nodes
	* Generates N trajectories in parallel
		* N start tokens generate N second steps, etc.
	* Utilises **REINFORCE**
	* **Specialist baselines: average reward over all generated trajectories** instead of greedy 
		* Less variance than greedy
		* Zero-mean (with greedy, most trajectories have sub-zero mean)
		* Less computationally expensive than training a critic
		* **Resistant to local minima**: trajectories are compared to N-1 other trajectories rather than just one, encouraging more diverse behaviours
	* Entropy reward built in to reward
	* Architecture agnostic

* Augmentation:
	* Issue with this method: can only generate limited set of trajectories according tot he number of start nodes
	* This is overcome by creating new start nodes with augmented version of the problem, e.g. flipped, rotated, mirrored, exploiting symmetries in the problem.











