## ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in LLMs  
**Date:** 4th June 2025  
**arXiv Link:** https://arxiv.org/abs/2505.24864  
**Key Points:**  
- Proposes ProRL, a training paradigm that applies extended RL training (e.g., thousands of steps) to LLMs to uncover novel reasoning strategies not present in the base model.  
- Shows that, contrary to beliefs that RL simply amplifies existing biases, prolonged RL can discover entirely new solution pathways that the base model could not sample.  
- Empirically demonstrates performance gains across math, code, logic puzzle, STEM‐reasoning, and instruction‐following tasks (e.g., +14.7% pass@1 on math, +54.8% on logic puzzles).  
- Correlates improvements with base‐model weakness: ProRL yields the largest gains where the base model struggles most, indicating a genuine expansion of reasoning capabilities.  

**Key Methods:**  
- **KL‐divergence control:** enforce a dynamic KL constraint during RL updates to balance exploration and policy drift.  
- **Reference policy resetting:** periodically reset the policy head to a previous checkpoint to prevent mode collapse and encourage exploration of new reasoning paths.  
- **Diverse task suite:** train on a broad mixture of tasks (136K problems spanning math, code, logic puzzles, STEM, and instructions) so that RL is not overfitting to a single domain.  
- **Extended training schedule:** perform ProRL for 2,000+ gradient steps (unprecedented for reasoning RL), showing continual improvement even after “saturation” of standard metrics.  