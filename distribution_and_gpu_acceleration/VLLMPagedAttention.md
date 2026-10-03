# Efficient Memory Management for Large Language Model Serving with PagedAttention

**Date:** 2nd October 2026

[arXiv Link](https://arxiv.org/abs/2309.06180) | [PDF](https://arxiv.org/pdf/2309.06180) | [Code](https://github.com/vllm-project/vllm)

Related: [ELI5: FlashAttention](FlashAttention.md)

## Key Points
* KV cache is veruy inefficient in contiguous, static memory as prompts can be different lengths. Old methods just pre-assign maximum length, which is highly wasteful.
* vLLM introduces **paged memory** and **paged attention**:
    * **Paged Memory**: a scedhuler and system for storing data in non-contiguous chunks. Saves memory by not having to reserve huge chunks.
    * **Paged Attention**: an efficient kernel for computing attention across chunms of the Qs, Ks and Vs
* Bunch of inference methods
