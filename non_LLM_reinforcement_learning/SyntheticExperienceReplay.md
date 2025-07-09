## SynthER  
**Date:** 29th May 2025 
[arXiv Link](https://arxiv.org/abs/2303.06614)
**Key Points:**  
- Presents SynthER, a method that trains a residual multi‐layer perceptron as a diffusion model on collected agent experiences.  
- SynthER generates synthetic trajectories (“hallucinated experiences”) to augment the agent’s replay buffer, improving exploration in sparse‐reward tasks.  
- Demonstrates that augmenting real data with diffusion‐generated rollouts yields higher policy performance in fewer environment interactions.  

**Key Methods:**  
- **Residual MLP diffusion model:** models the transition distribution $p(s_{t+1}, r_t | s_t, a_t)$ by learning to reverse a noise‐corrupted version of real transitions.  
- **Experience augmentation:** sample synthetic transitions from the diffusion model and add them to the replay buffer alongside real experience.  
- **Importance weighting:** assign lower weights to synthetic transitions if their log‐probability under the learned diffusion model is low, mitigating model bias.  
- **Agent training loop:** alternate between collecting real experiences with the current policy, retraining the diffusion model, and generating new synthetic data for policy updates.  
