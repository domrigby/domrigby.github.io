## Absolute Zero Reasoner
**Date:** 23rd May 2025 
[arXiv Link](https://arxiv.org/abs/2505.03335) 
**Key Points:**  
- Reinforcement learning where LLM proposes problems and then also solves them in a self-play methodology.  
- This means we don't need any external examples.  
- The problems can be verified using a coding engine. This allows theoretically allows unlimited reasoning as coding languages are Turing complete.

**Key Methods:**  
- Problems are built of three parts: input, code and output. One is hidden and the LLM has to guess what it would be. The code engine can verify this.  
- MuZero-like planning in latent space (using a world model) to guide self-play loops.  
- Unified code‐execution environment that serves both as verifier and data generator, allowing AZR to remain entirely “zero‐data.”  
- Open‐source implementation, demonstrating compatibility across model scales and architectures.  
