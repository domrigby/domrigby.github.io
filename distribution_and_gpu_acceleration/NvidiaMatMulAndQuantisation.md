# Nvidia Docs: Matrix Multiplication Background

Date read: 28th September 2025

[Docs link](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#gpu-imple)

## Key Points
* **GEMM definitions**:
	* General Matrix Multiplication (GEMM)
	* Represented as: C = aAB + bC
		* A and B = input matrices
		* a and b = scalar inputs
		* C = vector to be overwritten
	* For linear layers a = 1 and b = 0
	* For skip connections a = 1 and b = 1

* Classic example:
	* A: mxk, B: kxn, C: mxn
	* Computing C requires doing mxn k-length dot products. This is m*n*l=k fused multiply adds (FMAs).
	* As each FMA is two operations this is 2*m*n*k operations.

* **Arithmetic Intensity**:
	* Number of operations per byte loaded: tells us whether layer will be memory or maths limited.
	* This is done by **comparing to GPU max** (from datasheet)
	* If AI > GPU max: mem limited (maths faster than memory loading)
	* If AI < GPU max: maths limited (doesn't matter how fast we load data, we're limited by maths).

* **GPU Implemenations**:
	* GPUs parallelise matmuls by splitting them into tiles which are computed in parallel.
	* These tiles are often **performed on tensors cores**: therefore dimensions must allign with numbers for TCs.
	* There are multiple tiling strategies (decided by PyTorch so don't worry too much):
		1. Large tiles: more data re-use but less parallelisation for small matmuls
		2. Small tiles: more parallel but more read and writes.
	* Larger tiles tend to be better for larger matrices.

* **QUANTISATION EFFECTS**:
	1. **Tile quantisation**: wasted compute as a result of matrices not dividing perfectly into tiles.
		* Tiles should fit perfectly into our matrix.
		* If not, we still have to compute the whole tile, mostly padded, and we waste a lot of compute.
		* E.g. if tile size is 128 and our dim is 257, we have to compute 3 tiles worth, 50% more than a dim of 256
		* Only valid data is loaded, but invalid columns still have to be computed.

	2. **Wave quantisation**: wasted compute as a result of the number of tiles not dividing perfectly into the number of streaming multi-processors.
		* Have to perform not fully utilised last step for the left over tiles.
		* Occurs over large batch size scale as this occurs across whole GPU.