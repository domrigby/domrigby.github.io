## Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning
[Arxiv link](https://arxiv.org/abs/2506.01939)

### Key Points
- High entropy tokens (entropy of output probability distribution) act as 'forks in the road' for reasoning.
- This is where the crucial decisions are made. The rest of the tokens are low entropy and pretty deterministic
- Examples of high entropy words: suppose, since, actually etc. These are words which alter the direction of the reasoning
- RLVF tends to maintain entropy of the output, where as SFT does not. Hence why RLVF avoids overfitting more.

## Methods
- Calculates entropy of output probability distribution and assosciate it with the chosen token
- When doing DAPO or GRPO backpass, zero out any token gradients with below the threshold entropy. 
- Threshold entropy is set in this paper to be the 80th percentile