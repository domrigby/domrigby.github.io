# Scaling Up RL: Unlocking Diverse Reasoning in LLMs via Prolonged Training

**Date:** 26th September 2026

[arXiv Link](https://arxiv.org/abs/2507.12507) | [PDF](https://arxiv.org/pdf/2507.12507) | [Model (Hugging Face)](https://huggingface.co/nvidia/Nemotron-Research-Reasoning-Qwen-1.5B)

Related: [ProRL](ProlongedRL.md), [ProRL V2](ProRL2.md)

## Key Points
* Paper essentially trials DAPO, runs ablations and presents solutions to common issues in training: entropy collapse, instability and plateauing performance.
* Provides some implementation details about how they hosted it on their infrastructure:
    * Sandboxed rewards servers to prevent crashes crashing whole training
    * Heavy use of multi-processing + distributed compute to scale reward calculation
    * Calculate reward on CPU as soon as rollout has finished, in parallel to GPU runs

## Key Methods
* Utilised GRPO with:
    1. Decoupled clipping: epsilon high > epsilon low. Encourages increasing low probability answers
    2. Dynamic prompt sampling: filter out any prompts where model scores 0 or 1 (no information gain)
    3. **Kl regularisation with regular reference model reset**: 
        * KL regularisation term against reference model (starts with basemodel)
        * Change model to current model whenever performance plateaus or entropy collapses
        * Encourages model to maintain higher entropy
        * Works better when model performance is already good


## Results
* Found higher sampling temperature (1.2) starts off more unstable than low (0.6), but high overtakes eventially (fairly obvious)

## Thoughts
* 
