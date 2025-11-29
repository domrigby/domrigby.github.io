# ELI5: FlashAttention

Date read: 6th October 2025

[Blog post](https://gordicaleksa.medium.com/eli5-flash-attention-5c44017022ad)

## Key Points
* Flash attention benefits:
	1. Fast
	2. Memory efficient (O(N^2) -> O(N))
	3. **Outputs are exact**: same as vanilla attention
	4. **I/O aware**: 
		* Optimised for the GPU architecture
		* E.g. lowers memory read and writes rather than focusing on FLOPs which are cheaper on GPUs
		* Speed of FLOP increases has rapidly outpaced memory throughput.

* Attention is memory limited:
	* Consists of many elementwise operations -> low arithmetic intensity
	* These take most of the time, despite matmuls taking up most flops

* Normal vs Flash Attention
	* Normal attention treats read and writes as free
	* FlashAttention optimises this by limiting read and writes between the SRAM (very fast, on SM) and the HBM.

* Optimisations:
	* **Kernel fusion**: merges many operations into a single one which keeps the values in SRAM rather than sending them to and from the HBM
	* **Materialisation**: limits memory usage to O(N) rather than O(N^2). Details below
	* **Tiling**: 
		* Breaks matmuls and softmax are broken down into tiles which max out SRAM utilisation
		* Introduces method for doing softmax in tiles (no need for row / column wide exp sum). Instead it can be iteratively updated across blocks
		* Outer and inner loops: outer loop loops across V and K, inner loop across Q and O (O is output, which needs to preallocated in memory)

* Check blog for further implementation details.

* Issue:
	* Only supported on certain GPUs
	* Requires writing of custom CUDA kernels (e.g. triton)






