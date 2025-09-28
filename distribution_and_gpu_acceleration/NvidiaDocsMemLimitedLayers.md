# Optimising Memory Limited Layers

Date read: 27th September 2025

[Docs link](TODO)

## Key Points
* Memory limited layers: layers with a small number of operations per loaded bit tend to be bottlenecked by reading and writing data
rather than actually performing operations on it.
* Will have linear increases in performance until memory bandwidth is reached.
* Performance is proportional to size of inputs and/or outputs
* How to help:
	* Limit the number of read and writes if you can
	* If batch size is small and layer is light, see if there is an optimised implementation in **cuDNN API reference**
* Examples: norms, pooling, activations etc.