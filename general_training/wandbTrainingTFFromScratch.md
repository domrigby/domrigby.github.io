# Current Best Practices for Training LLMs from Scratch

Date read: 1st October 2025

[Blog link](https://wandb.ai/site/wp-content/uploads/2023/09/Current-Best-Practices-for-Training-LLMs-from-Scratch-Final.pdf?utm_source=chatgpt.com)

## Key Points
* Aim of paper: discuss best practices and common pitfalls when training LLMs
* **Scaling Laws**:
	* DeepMind published a paper showing most models are undertrained: didn't see enough data
	* Showed **for optimal performance: model size and number of weights should roughly increase at the same rate**
	* I.e. 10x more compute -> 3.1x data and 3.1x more weights
	* Recommends: 
		* 20x more data than weights
		* Check Chinchilla optimal model size.

* **Parallelism**:
	* Micro-batches: small batches which accumulate gradients before updating
	* Data-parallelism: 
		* Slit batch across GPUs with same model, all-reduce gradients before updates. 
		* Simple but inefficient in that you have to broadcast whole gradient and have many optimisers for one model (memory inefficient).
	* Tensor-parallelism:
		* Splits matmul tiles across many GPUs
		* Asynchronous
		* Memory efficient (tiles only small) but adds additional comms, requires high comms BW
	* Pipeline parallelism:
		* Splits model into sections which can process new data in parallel.
		* Get new data once finished processing 
		* Memory and compute efficient but limited by depth of model
	* Nvidia PTD-P example:
		* 52% utilisation across thousands of GPUs
		* Inside nodes: data and tensor parallelism
		* Across nodes: pipeline parallelism 
	
* Data:
	* Diversity super important for generalisation
	* Upsample high quality data -> quality is important

* Tokenisation:
	* Use subword tokenisation to reduce number of possible tokens

* Training advice:
	1. Start with **smaller model and scale up slowly**
	2. Start with popular architecture and add bits as needed
	3. Perform experiments to find best architecture:
		* weight init, positional embeddings, optimiser, activation, learning rate, etc.
	4. Auto-search LR, batch size, dropout rate
	5. Start with hyperparameters from the literature and tune accordingly
	6. HP to change during training: LR and batch size (start small and increase)

* Typical instability advice:
	* Advice: save checkpoints and lower learning rate if it fails.
	* Biggest batch possible
	* Decreasing LR schedule
	* Use batch normalisation
	* Pre-trained start
	* Augment data
	* Skip data which caused spike
	* QV normalisation
	* Regularisation (L1 and L2)
	* Swap optimiser during training if needed