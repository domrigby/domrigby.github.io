# SPIRAL: Self-Play on Zero-Sum Games Incentivizes Reasoning via Multi-Agent Multi-Turn Reinforcement Learning

Date: 3rd July 2025

[arXiv link](https://arxiv.org/abs/2506.24119)

# Key Points
* Introduces a self-play method for LLMs in self-play methodology for zero-sum text based games.
* **Roll Based Advantage Estimator**: games have multiple roles. This causes a lot of noise and variance in the GAE. They
therefore have a separate GAE for each role in each game, reducing variance. This does however limit generalisation.
* Uses **same LLM policy for all role but conditioned in the prompt**.
* Creates **unlimited non-verifiable rewards**.
* Framework: many parallel agents generate experiences for a single worker.