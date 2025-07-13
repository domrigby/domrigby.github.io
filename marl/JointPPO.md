## JointPPO: Diving Deeper into the Effectiveness of PPO in Multi-Age Reinforcement Learning 

**Date:** 28th May 2025

[arXiv Link](https://arxiv.org/pdf/2404.11831)

**Key Points:**  
- Addresses multi‐agent reinforcement learning (MARL) in environments requiring high communication by centralizing decision‐making via a joint policy.  
- Decomposes a joint action distribution into a sequence of conditional decisions, each conditioned on prior agents’ choices.  
- Uses a Transformer decoder to model the sequence of agent‐decisions, effectively learning “who decides when” in a centralized yet factorized manner.  
- Decision order (which agent acts first, second, etc.) is dynamic and learned via a Graph Neural Network (GNN) that predicts optimal ordering based on state.  

**Key Methods:**  
- **Conditional policy decomposition:** joint policy $\pi(a_1, a_2, ..., a_n | s)$ factorized as $\pi(a_1|s)\pi(a_2|s,a_1)\dots $, implemented with a Transformer decoder.  
- **Decision‐order GNN:** ingests the global state and agent identifiers to output a permutation (ordering) that maximizes expected team return.  
- **Centralized training with shared parameters:** all agent “slots” share weights in the Transformer; indices distinguish agents at inference.  
- **PPO‐based optimization:** Proximal Policy Optimization on the joint policy, with entropy regularization to encourage diverse ordering.  
