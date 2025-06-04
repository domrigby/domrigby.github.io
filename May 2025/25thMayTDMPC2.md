## TD-MPC2: Scalable, Robust World Models for Continuous Control  
**Date:** 25th May 2025
[arXiv Link](https://arxiv.org/abs/2310.16828)
**Key Points:**  
- Extends TD-MPC (Temporal Difference Model Predictive Control) into TD-MPC², a multi-task RL framework using a world model for planning in latent space.  
- Achieves strong performance across 104 continuous control tasks without per-task hyperparameter tuning.  
- Demonstrates that a single 317M‐parameter world‐model agent can learn to solve 80+ tasks spanning diverse domains and embodiments.  
- Utilizes MuZero‐like latent‐space planning that iteratively refines action sequences via learned dynamics.  

**Key Methods:**  
- World model architecture: decoder-free latent dynamics with MLPs (LayerNorm + Mish), SimNorm on latent states, and an ensemble of Q‐functions with Dropout.  
- MuZero‐style planning: perform latent‐space rollouts to evaluate action sequences; encoder maps observations directly to LSTM‐based latent states.  
- Shared hyperparameters for all tasks: same learning rate, network sizes, planning depth, etc., enabling out-of-the-box multi-task generalization.  
- Open-source code and benchmarks: publicly available repository for reproducibility.  
