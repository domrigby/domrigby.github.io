# Speeding Up Graph Learning Models with PyG and torch.compile

Date read: 13th October 2025

[Blog link]()

## Key Points
* Aim of post: explain how to overcome typical problems that arrive when using torch compile to speed up GNN related activitie, specifically graph transformers.
* Blog focuses on doing this in PyTorch with the following tools:
	* PyTorch Frame: embeds multi-model data (e.g. text, images, video) into shared embedding space
	* PyG: handles message passing 
	* PyTorch Lighting: helps scale training processes onto GPU clusters.
* Torch compile: 
	* Takes standard PyTorch code and performs kernel fusion to create kernels which minimise memory reads amd writes, delivering large speed ups.
	* It does however have its limitations, e.g.:
		* Recompilatio
		* Host-device syncs
		* Learning rate treated as constant

* Recompilation:
	* When the assumptionsmade at compile time break, the code has to be recompiled.
	* Common assumptions: data type, data shape, constants.
	* **Dynamic shapes are very common when using GNNs** due to differing number of nodes.
	* Tips:
		1. Set torch.compile(dynamic=True) to optimise for dynamic shapes. 
		2. Set TORCH_LOGS="recompiles" such that it is logged when you recompile and you can address the issues as required.

* LR treated as a constant:
	* If LR is set to a float ti will be treated as a constant at compile time and require a recompile if changed. 
	* Set LR as a tensor to prevent this

* **Graph breaks**:
	* Some code can't be compiled and torch will return to eager mode for these steps.
	* This breaks the compiled code into a set of kernels rather than just one.
	* This causes unneccessary memory read and writes
	* Tips:
		1. Set TORCH_LOGS='graph_breaks', which will log any graph breaks which can often be easily fixed.

* CUDA graphs:
	* Set torch.compile(mode='reduce-overhead') to creates a CUDA graph which can be launched in one go from the CPU and reduce kernel launch overhead.
	* **Only use if the number of possible shapes is small** as it will have recompile the instruction set for every individual size.

* Asynchronousity:
	* Python usually launches GPU kernels and doesn't wait for them to finish before continuing, unless it is explicitly told to in the code.
	* Reduce this if possiblle by:
		* Keep items as torch tensors rather than floats, ints etc
		* Don't print too often
		* Don't use item
		* Be explicity with expected shape when possible






