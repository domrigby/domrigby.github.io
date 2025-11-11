# EnvPool: A Highly Parallel Reinforcement Learning Environment Execution Engine

Date read: 10th November 2025

[Paper link](https://arxiv.org/pdf/2206.10558)

## Key Points
* Aims to speed up environment execution. Training and inference benefit from speeds up from supervised learning and computer vision, environmnets are RL only.
* Scaling is as important as algorithm improvements
* Done in C++. Python multiprocessing is inefficient in comparison and is not optimised for environment execution.
* Why not GPU? Aims to be able to run any environment, not just matrix based ones.

* How it works:
	* Replaces step with send function which queues actions and then the thread continues without waiting.
	* ThreadPool picks the actions from the queue and then steps the simulation, placing the new state in a queue
	* Main thread then takes all the items from the state queue and predicts new actions.
	* Key point: the main thread **does not wait for all to be done**, only for the first M to be done. This means you don't wait for the slowest to finish (same as DD-PPO) and they are handled in the next batch.

* Performance improvements:
	* Number threads <= number of CPU cores (less context switching)
	* Threads are pinned to a certain core (less context switching)
	* Number of environments = 2-3x number of threads (keep threads fully loaded).