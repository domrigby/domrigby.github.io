# Accelerating PyTorch with CUDA Graphs

Date read: 24th September 2025

## Key Points
* CUDA graphs: 
	* Collect a set of kernels and fuse them into one such that they can called with little overhead.
	* Strings of commands then execute on the GPU
	* Requires static inputs: graph runs and reads ands write to same memory address every time.
* This matters a lot less at large batch sizes when launch time is less of an overhead.
* Therefore might help more during inference when batches are smaller.