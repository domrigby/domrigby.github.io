## Capture the Flag  
**Date:** 2025-06-04  
**arXiv Link:** Need to find lin
**Key Points:**  
- Introduces a multi-agent RL environment (“Capture the Flag”) with populations of agents employing both fast and slow LSTMs.  
- Implements a two-tier optimization: each agent learns an internal reward function while an external optimization process adjusts those internal rewards to maximize team win rate.  
- Demonstrates emergent cooperative strategies: agents learn to coordinate roles (offense vs. defense) by balancing internal exploration with external competitive reward.  
- Highlights that an agent’s “internal” objective (e.g., collecting flags efficiently) can diverge from the “external” objective (winning the match), requiring nested RL loops.  

**Key Methods:**  
- Fast‐LSTM policy network: processes high‐frequency observations (e.g., immediate surroundings) for quick reflexes.  
- Slow‐LSTM “strategy” network: processes longer‐term context (e.g., team positions, flag locations) to set internal sub‐goals.  
- Two‐tier optimization:  
  - **Internal RL loop:** each agent maximizes its internal reward (e.g., proximity to enemy flag, survival time).  
  - **External RL loop (“meta‐optimizer”):** adjusts internal reward weights across the population to maximize global win rate.  
- Population‐based training: regularly introduce new agents, replace poorly performing ones, and evolve internal‐reward parameters.  