## Writing-Zero: Bridge the Gap Between Non-verifiable Tasks and Verifiable Rewards
Date: 12th June 2025  
[arXiv Link](https://arxiv.org/abs/2506.00103)

### Key Points:
- Aims to bridge non-verifiable rewards gap for reinforcement learning on LLMs.
- Subjective tasks such as creative writing are inherently unverifiable and opinion based. This makes giving them rewards
in RL extremely difficult.
- Scoring normally relies on preferences between two pieces. This paper trained a Generative Reward Model (GenRM) to give
a total score of 10 between two pieces of writing.

### Key Methods:
- Bootstrapped Relative Policy Optimisation: as mentioned above, you need some writing to compare the results to.
BRPO uses a random sample from the batch to use as the comparison. This means the reference is always fresh.
- DAPO: filters out any prompts which score 0 or 1 as these don't tell us much.