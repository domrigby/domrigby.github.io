## Multi-Turn Credit Assignment  
**Date:** 30/05/2025
[arXiv Link](https://arxiv.org/abs/2505.11821)
**Key Points:**  
- Addresses the problem of sparse/delayed rewards in multi‐turn (multi‐agent or multi‐step) settings by introducing two separate advantage functions: one for the immediate “turn” and one for the cumulative “trajectory.”  
- Shows that combining both advantage estimates (via a weighted mean) results in more stable policy gradient updates and faster credit assignment across long horizons.  
- Demonstrates empirical improvements on conversational AI benchmarks and multi‐agent cooperative games.  

**Key Methods:**  
- **Per‐turn advantage $A_{\text{turn}}(s_t,a_t)$:** computed as $r_t + \gamma V(s_{t+1}) - V(s_t)$.  
- **Trajectory‐level advantage $A_{\text{traj}}(s_t,a_t)$:** computed using full return $G_t = \sum_{k=t}^{T} \gamma^{\,k-t} r_k$, so $A_{\text{traj}}(s_t,a_t) = G_t - V(s_t)$.  
- **Weighted advantage:** define $A_{\text{combined}} = \alpha\,A_{\text{turn}} + (1-\alpha)\,A_{\text{traj}}$, with $\alpha\in[0,1]$ tuned on a validation set.  
- **Policy optimization:** standard PPO update using $A_{\text{combined}}$; value network is trained to regress the trajectory return $G_t$.  