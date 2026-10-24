# LoGRA: Scaling LLM Reinforcement Learning with Low-Rank Gradient Sketches

**Date:** 18th October 2026

[arXiv Link](https://arxiv.org/abs/2610.06647) | [PDF](https://arxiv.org/pdf/2610.06647) | [Code](https://github.com/skzhang1/labs-molt/tree/logra/examples/scripts/logra)

## Key Points
* Algorithm level memory optimisation
* Intuition is that binary reward contains very little info... this doesn't need full gradient to be represented
* LoGRA using a low rank approximation of the gradient with KL-divergence control
* If G is dxk... it will approximate G as SA with S as dxr and A as rxk. r << d or k
* Then update W <- W - alpha * nu * SA 
* **Saves 45% memory on 27B parameter model**