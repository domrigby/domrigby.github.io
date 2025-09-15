# How to Think About GPUs

Date read: 14th - 15th September 2025

[Blog link](https://jax-ml.github.io/scaling-book/gpus/)

## Key Points
* Blog post summarises:
  1. GPU architectures: mainly Nvidia and compares to TPUs
  2. Networking: how data is moved around and the bottlenecks
  3. Collective operations across GPU clusters and their theoretical speed limits

* **GPU architecture**:
  * **Streaming Multi-Processors**: GPU's main core. Each one has many, e.g. H100 has 132. Made up of sub-parts:
    * **Tensor cores**: specialise in matrix multiplication. Useful for multiply-and-accumulate operations. Useful for dense layers of neural networks (1-2 FLOP per clock cycle).
    * **CUDA cores**: general purpose cores for vector operations: multiply, add, compare etc. Slower but more multi-purpose (1 FLOP per clock cycle). They can also access non-contiguous memory.
    * **Warp scheduler**: bit of hardware which selects which warp to run this clock cycle.
      * Operate **SIMT**: Single Instruction Multiple Threads -> group data into warps and do the same operation on all, potentially masking out unwanted operations.
      * Warps: collections of 32 threads over which one operation is done per clock cycle. The same instruction is done across the whole warp.
      If they require different operation, then masking is performed.
      * Why? Some operations are slower than others, e.g. fetching data is slower than arithmetic. The warp scheduler minimise SM downtime by constantly scheduling ready warps.
      * Interleaving warps can hide latency
    * **Registers**: extremely fast access memory for storing warps just before calculation. Each CUDA core has access to 256 registers
    * **L1 cache / SMEM**: small (256kB) rapid access memory
    * **L2 cache**: ~50MB of reasonable fast access memory
    * **High Bandwidth Memory**: bit like RAM for the GPU.
* GPU architecture has advantage of many small independent SMs. This means a program can launch many independent processes.

* **Networking**: discusses how data is shared within a cluster of GPUs.
  * **Node**:
    * Usually around 8 GPUs.
    * Extremely fast **all-to-all NVLink** connections
    * Max bandwith ~450GB/s on H100.
  * **Scalable Units**:
    * Collection of nodes, usually around 32
    * Connected by ethernet and InfiniBand switches
    * Stored in **fat tree**, i.e. tree format with nodes as leaves and higher bandwidth the higher up the tree you go.
    There are multiple 'trunks' meaning there are multiple equal cost paths to get from one node to another, decreasing 
    choke points.