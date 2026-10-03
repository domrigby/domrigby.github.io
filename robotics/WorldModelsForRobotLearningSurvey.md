# World Model for Robot Learning: A Comprehensive Survey

**Date:** 29th September 2026

[arXiv Link](https://arxiv.org/abs/2605.00080) | [PDF](https://arxiv.org/pdf/2605.00080)

Related: [Dreamer 4](../non_LLM_reinforcement_learning/Dreamer4OfflineWorldModel.md) | [TD-MPC2](../non_LLM_reinforcement_learning/TDMPC2.md)

## Key Points
* VLAs are too reactive for long term reasoning as they lack planning capability. World models supply a way to allow policies to plan.
* Language lacks high information content about physical interactions.

* **Video generation models can be used as world models**
    * The large scale internet data provides 'useful priors for motion'. They can be conditioned to create motion paths towards goals.
    * Overtime these methods have **gone from predicting pixels to predicting latent representations** 
* Uses:
    1. Action selection via rollouts: rollout and select best action.
    2. Future conditions action selection: world model (e.g. video model) creates a rollout with a desired goal and then remap it onto the action space.
    3. Joint predictive control
* World models can also be used as cheap simulators (on device).
* Worlds models can be used for RL, but helps to update the world model to make sure it stays accurate (MuZero style).

## Types
* **Decouple design**: separate world model and policy. World model generates trajectory and then policy inversely predicts the actions.
    * e.g. video model generates robot pick up ball -> policy turns into actions
* **Unified backbone**: single model generates state and actions inside generative process.
* **MoE/MoT**: separate video and action generation, but they can run different frequencies, representation scale etc.
* **VLAs**: can predict future images
* **Latent-Space World Modelling**: MuZero style future prediction. Acts in latent space.

