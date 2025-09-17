# RL’S RAZOR: WHY ONLINE REINFORCEMENT LEARNING FORGETS LESS

Date read: 16th September 2025

[ArXiv link](https://arxiv.org/pdf/2509.04259)

## Key Points
* Address catastrophic forgetting when tuning LLMs
* Catastrophic forgetting: a model learns to complete a new task, but forgets how to do previous ones
* Paper proposes and proves that **online RL preserves previous knowledge more than Supervised Fine-Tuning (SFT)
* Why?
  * During training, RL only sees actions with non-zero probability.
  * It then slowly moves probability mass towards the good choices, rather than pulling it like it does with SFT.
  * This means good solutions are found closer to the original policy, SFT does not guarantee this.
  * Online RL often explicitly or implicitly minimise KL-divergence against a reference policy.
* How?
  * Paper proposes that the forgetting can be represented by the KL-divergence between the new and old policy **on the new task**.