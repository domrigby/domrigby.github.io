# Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time

Date: 14th July 2025

[arXiv link](https://arxiv.org/abs/2203.05482)

## Key Points
* Ensembles of models often work better than the single highest performing model. Issue: they require more inference time compute.
* ModelSoup tries to alleviate this by **averaging the weights** across fine-tuned models trained with different hyperparameters
* Why does this work? Fine-tuned models often end up in **same loss value** meaning that averaging across their positions
can get a lower loss and better performance.
* ModelSoup has three different varieties:
  1. Uniform soup: simple average. Issue: some models end up in different valleys and then negatively impact performance.
  2. **Greedy soup**: only add a model if it makes it perform better (**best performing**).
  3. Learned soup: optimised interpolation between model parameters to maximise performance on a test set. It also has the
  downside that all models have to loaded into RAM.