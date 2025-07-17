# What Has a Foundation Model Found? Using Inductive Bias to Probe for World Models

Date: 16th July 2025

[arXiv link](https://arxiv.org/pdf/2507.06952 )

**Needs visiting**s

## Key Points
* Much of our scientific understanding has come from observing sequences and fitting models to predict them. E.g. Kepler and then Newton's Laws.
Can foundation models do the same?
* Paper measures this using an **inductive bias probe**. This probe measures whether a model is learning actual laws about the system
  (these would be representative functions in the embedding space), or just learning heuristi workarounds.
* This is measured by:
  1. Creating synthetic data using the models
  2. Inputs are put through foundation model
  3. Model is fine-tuned to this new data. 
  4. Find pairs with the same output: check model is outputting the same value for them.