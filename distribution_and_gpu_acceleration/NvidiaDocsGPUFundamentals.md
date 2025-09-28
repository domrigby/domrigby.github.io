# Nvidia Documentation: GPU Performance Background User's Guide

Date read: 27th September 2025

[Docs link](https://docs.nvidia.com/deeplearning/performance/pdf/Optimizing-Linear-Fully-Connected-Layers-User-Guide.pdf)

## Key Points
* Basic architecture:
	* NV Streaming Multiprocessors: each has its own instruction scheduler
	* On-chip L2 fast cache (mega fast access)
	* High bandwidth DRAM 
* Operations, code and values are stored in DRAM and then loaded to L2 cache when needed to perform operations.
* **Tensor Cores**: accelerate matrix multiplication (only do matmuls)
* **CUDA cores**: perform operations which can't be converted into matmuls

* **Extreme parallelisation**:
	* Two level thread hierarchy:
		* Function threads are grouped into equally sized **blocks**
		* Groups of blocks are launched
		* GPUs hide dependent instruction latency by constantly switching threads.
		This allows linear performance increase to batches far above the number of cores.
	* Why?
		* GPUs have many SMs each with their own scheduler and many threads which can communicate via shared memory.
		* SMs operate one block at a time but can achieve linear performance with many blocks per SM
		* **You want to have a number of blocks an integer multiple to the number of SMs to reduce tail effect**
		* Tail effect: under utilisation at the end of a batch in which not all SMs are being used (e.g. 6 jobs across 4 SMs leaves two in last run).

* **Limiting factors**:
	1. Memory bandwidth (e.g. ReLU)
	2. Maths bandwidth (e.g. large linear layer)
	3. Latency
	* Rule: simple ops tend to be memory limited, complex or large operations tend to be math limited





