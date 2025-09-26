# Performance Tuning Guide (PyTorch)

Date read: 26th September 2025

[Blog link](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)

## Key Points
* Classic advice:
	* Data loading:
		* Take advantage of asynchronous data loading bu setting num_workers > 0
`		* **Set pin_memory to True if loading to GPU**
	* Use no_grad() for inference
	* Disable biases before BatchNorm layers
	* **Set gradients to None** rather than using zero_grad(), or set_to_none as True
	* Create tensors directly on device rather than sending them via 'to()'

* **Kernel fusion**:
	* PyTorch eager mode:
		* Each line calls a kernel from the CPU.
		* **Each kernel loads data, performs operation and then writes** to HBM.
		* This is a lot of reading and writing (slowest bit of GPU)!
	* Answer: **fuse operations into one kernel** which reads and writes data once.
		* **Enable kernel fusion using torch compile**

* **Buffer checkpointing** to save memory:\
	* Save every few activations on the forward pass rather than all of them.
	* Recalculate the rest during backward pass.
	* **Pick layers which will make this easy**, e.g. output of heaviest layer.

* **Tensor cores**: specialist matrix multiplication area of GPU
	* Can take advantage of lower precision data to get major speed ups
	* Use multiples of 8 for sizes when using tensor cores, pad if needs

* **CUDA graphs**:
	* Compile with "max-autotune" or "reduce-overhead" as an argument to torch compile.

* Avoid unnecessary GPU-CPU comms or syncs, e.g. prints

* Enable Automatic Mixed Precision (AMP)

* **Pre-allocate memory if input size varies**:
	* Varying size slows PyTorchs caching allocator.
	* New size means having to release intermediate buffers and re-allocate new ones
	* This is why models are slower on first pass (allocating memory)

* DistributedDataParallel > DataParallel:
	* Automatically does all-reduce unless no-sync is called.