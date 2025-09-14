# Compiling machine learning programs via high-level tracing

Date read: 14th September 2025

[Paper link](https://cs.stanford.edu/~rfrostig/pubs/jax-mlsys2018.pdf)

## Key Points
* JAX is a JIT compiler for writing hardware accelerated code in python.
* JAX is made to mimic Numpy and will compile any matrix operations, slicing etc.
* Assumption: ML workflows can be broken down into sets of **Pure-and-Statically-Composed** sub-routines which are then activated by dynamic logic. What this means:
	* Pure: running the routine only effects the outputs, it does not have side effects.
	* Statically-composed: route can be represented as a graph of unchanging primitive functions
	* Primitive functions: kernel level function which in Numpy include matrix operations which are compilable.
* Technical details:
	* JAX stands for Just After eXecution, as it executes the code before building the graphs etc.
	* User labels the PSC routines with a decorator.
	* JAX includes distribution functions for running many operations in parallel.