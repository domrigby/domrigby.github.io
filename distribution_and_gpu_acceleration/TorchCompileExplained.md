# What is Torch Compile?

Date read: 21st September 2025

[Blog link]()

## Key Points
* What is torch compile?
    * Torch compile allows you to speed up torch code with minimal edits
    * JIT compiles it into optimised kernels
* How does it work?
  * Graph acquisition -> breaks down code into graphs and sub-graphs and classifies which can be compiled or have to be eager
  * Graph lowering -> converts graph into backend kernels
  * Graph compilation -> generate optimised kernels for operation. 
* Optimisations:
  * **Kernel fusion**: combine multiple operations into one kernel to reduce the overhead of calling them from the CPU.
  * **CUDA graph capture**: graph is captured and stored for reuse
  * **Operator fusion**: fuses multiple operation with multiple intermediates into single operators. E.g. relu(fc(x))
* Common pitfalls:
  * Recompilation due to size changes. Fix by:
    * Maintain a constant size, including batch. Dont let last batch be smaller
  * Graph breaks: parts of the graph can't be compiled so are eagerly executed. Fix by using fullgraph=True to identify these
  * Always benchmark to ensure it is quicker
  * Make sure to compile before distributing