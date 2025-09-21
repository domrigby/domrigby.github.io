# PyTorch Internals by Edward Zang

Date read: 21st September 2025

[Blog post link](https://blog.ezyang.com/2019/05/pytorch-internals/?utm_source=chatgpt.com)

## Key Points

### Tensors
* Tensors are stored in contiguous memory with the shape as metadata (hence reshape is ~free)
* Metadata:
	* dtype
	* size: shape of tensor
	* stride: how to convert indices to memory address is contiguous memory
		* E.g. stride: (2, 1) -> [i, j] -> base address + 2*i + 1*j
* Indexing tensors does not create new tensors, just alters view of current one.
* Storage: tensors contain human readable data, e.g. size and they have a corresponding storage unit 
which stores the actual data: dtype, device etc.
* Tensor types are fully defined by three values:
	1. Device
	2. Layout (strides, size etc. This can change for sparse tensors for example).
	3. Dtype

### Execution Steps
1. Python argument parsing
2. Variable dispatch
3. Dtype and device dispatch:
	* Dynamic dispatch calls the correct implementation for the device and datatype.
4. The assigned kernel is then called
	* Parallelisation occurs inside kernel, whether it be explicit for CPU or implicit in CUDA.

### Autograd
* Uses reverse mode for forward pass: performs forward step backwards.
* Creates a graph of the forward pass to trace backwards
* Leaf nodes of graph are the parameters we are updating.








