## Demonstration-Augmented RL  
**Date:** 2025-06-04  
**arXiv Link:** https://arxiv.org/abs/ (pending; often seen under “DemoRL” or “Demo-Augmented RL”)  
**Key Points:**  
- Integrates expert demonstrations into RL by adding a demonstration‐constraint term to the policy gradient loss.  
- Ensures that initially, the policy strongly mimics expert trajectories; over time, this imitation weight is annealed to allow exploration.  
- Bridges the gap between pure imitation learning and pure RL, resulting in faster convergence and better final performance.  

**Key Methods:**  
- **Loss function:**  
  \[
    \mathcal{L} = \mathcal{L}_{\text{RL}} + \lambda(t)\,\mathcal{L}_{\text{demo}},  
  \]  
  where $\mathcal{L}_{\text{demo}}$ encourages matching the expert’s action distribution (e.g., cross‐entropy), and $\lambda(t)$ decays from $\lambda_0$ to 0 over training.  
- **Annealing schedule:** often linear or exponential decay for $\lambda(t)$, ensuring early guidance from demonstrations and eventual policy autonomy.  
- **Expert buffer:** store a fixed set of high‐quality trajectories; sample mini‐batches from both on‐policy rollout buffer (for RL loss) and expert buffer (for imitation loss).  
- **PPO‐based optimization:** combine GAE (Generalized Advantage Estimation) for $\mathcal{L}_{\text{RL}}$ and cross‐entropy on expert states for $\mathcal{L}_{\text{demo}}$.  