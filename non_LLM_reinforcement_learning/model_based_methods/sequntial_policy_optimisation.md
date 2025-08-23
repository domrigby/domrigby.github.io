# SPO: Sequential Monte Carlo Policy Optimisation

Date read: 14th August 2025

[arXiv link](https://arxiv.org/pdf/2402.07963)

## Key Points
* Introduces SPO model-based reinforcement learning algorithm which utilises expectation-maximisation (trying to maximise
expected reward at a state) via particle filters, which have the advantage of being highly parallelisable.
* Works for both continuous and discrete action spaces with no modifications.
* No need to save a large tree, only require the current particle states.
* This model-based planning is then used as a target for policy optimisation via a KL-divergence based target (minimise KL-divergence
between planner and policy).
* Drawback: expects accurate adn deterministic model of the environment