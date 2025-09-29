# CUDA Study Log 4: Optimizing Constrained Decoding with Triton Kernel 

Date read: 22nd September 2025

[Blog link](https://gitlostmurali.com/blog/structured-generation-optimizations/)

## Key Points
* Problem: during inference of LLM there are often limitations on the tokens we can pick. Calculating logits for tokens 
which can't be chosen is a waste of compute (same in GPU RL envs which are done or inactive).
* He discusses three levels of optimisation:
  1. Finite State Machines: if there are sets of states which definitely follow one another, they can be compressed down
  to one state.
  2. **Optimised Matrix Multiplication**: identify tokens allowed and then only calculate the logits for these tokens. This is performed as masked matrix multiplication.
    * Reduces memory and computations. 
    * Limit: inflexible when batch. Each item in batch has different mask preventing efficient slicing and reducing parallelism.
  3. Kernel optimisations: uses **triton** to optimise the kernels. Methods used:
   * Block level: before performing block mat mul, the mask is checked to see if this block has any active values. If not, the block is skipped.
   * Thread level: checks s are not masked out when they are being loaded. This can negatively impact performance due to causing warp divergence, and disrupting contiguous memory access.