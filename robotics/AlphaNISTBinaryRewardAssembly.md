# AlphaNIST: Emergent Assembly Skills from Binary Task Rewards

**Date:** 19th October 2026

[arXiv Link](https://arxiv.org/abs/2610.13354) | [PDF](https://arxiv.org/pdf/2610.13354) | [Project Page](https://alpha-nist.github.io/)

Related: [DISCOVER: Automated Curricula for Sparse-Reward Reinforcement Learning](../open_endedness_and_auto_curriculums/DiscoverAutoCurriculaForSparseRewards.md)

## Key Points
* **Trains RL agent from scratch to complete multi-task, contact rich and long horizon assembly task** using standard PPO at scale
    * PPO is synchronous with 256k parallel envs, 32 length rollout
    * Separate critic and actor networks post feature extractiomn. Only two ~million MLP residual block each
* Scales RL through parallel environment on GPU in IsaacLab 
* Scaling only effective due to **rich reset states selected to be at frontier of capability**
    * Reset state have their empirical success rate recorded
    * Each GPU has local reset state cache
    * States are sampled from using **p*(1-p)**.
* **Reset states are selected to be high quality**: reaching for object, near-object, stable grasph, near goal
* **Only terminal binary rewards** with action regularisatio each step
* **Agent handles all 20 assembly tasks with one network** and performs better than any one individual, showing transferability in the dexterity tasks