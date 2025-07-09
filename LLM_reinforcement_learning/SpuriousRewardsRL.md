## Spurious Rewards: Rethinking Training Signals in RLVR

**Date:** 4th June 2025 

[arXiv link](https://arxiv.org/abs/2506.10947)  

**Key Points:**  
- Observes that applying reinforcement learning with verifiable rewards (RLVR) to Qwen‐2.5 Math models yields large 
improvements on math benchmarks—even when rewards are random or extremely sparse.  
- Demonstrates that “random” or “rare” rewards (e.g., +1 for any code output matching a certain format) still encourage
the model to practice reasoning and generate more code, indirectly improving performance.  
- Suggests that the benefit of RLVR may come largely from elicitation of latent capabilities
(e.g., “just make the model write more code”), rather than from specific reward signals.  
- Challenges the narrative that carefully engineered reward models are always necessary; even random reward signals can bootstrap better performance in math domains.  

**Key Methods:**  
- **Random/Rare reward design:** assign a small reward when the generated code contains a special token or satisfies a
trivial “format” rule, rather than verifying semantic correctness.  
- **GRPO clipping:** use Group Relative Policy Optimization (GRPO) with high clipping thresholds to allow large policy
updates when rare rewards are observed.  
- **One‐shot example RL:** occasionally include a single verified math problem to ensure minimal learning signal,
but rely mainly on random‐format rewards.  
- **Empirical evaluation:** measure MATH‐500 performance gains across Qwen‐2.5 Math models when
trained under random reward regimes.  