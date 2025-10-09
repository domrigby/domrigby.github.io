# Training Agents Inside of Scalable World Models

Date read: 8th October 2025

[Paper link](https://arxiv.org/abs/2509.24527)

## Key Points
* Summary:
	* World model is learned from offline data
	* Real time inference due to Efficient Transformer architecture
	* Performs RL in 'imagination' inside world model

* World models:
	* Learn to predict effect of actions
	* Allows for planning in imaginations
	* Two main types:
		* Ones trained via interaction: tend to be too domain specific
		* Video ones (e.g. Genie): tend to be imprecise

* Architecture:
	* **Diffusion model with shortcuts**: decides size of step at inference time (can speed up inference)
	* **Utilises diffusion forcing**: applies a different noise signal to each time step, allowing for the model to adapt 
	to varying detail levels of histroy 
	* **Causal tokeniser**: compresses video timeseries into embeddings
	* Efficient transformer:
		* Separate time and space attention (scale down N^2)
		* Quantised
		* **Multiple query heads for one VK head... reduce size of KV cache**
		* Stability: QKNorm, attention logit soft capping

* Training:
	1. Pre-train tokeniser (masked auto-encoder) on videos to get good embeddings
	2. Agent finetuning: finetune world model with task inputs for policy and reward heads
		* Utilises behavioural cloning
	3. RL in imagation to optimise policy and value heads
		* Imagines rollouts
		* Freezes dynamics model, only train policy and value net
		* Uses **PMPO**: only uses **sign of advantage, not magnitude: focus on all tasks independent of their scale of reward**








