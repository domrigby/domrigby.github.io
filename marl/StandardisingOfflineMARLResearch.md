# Dispelling the Mirage of Progress in Offline MARL through Standardised Baselines and Evaluation

Date: 28th July 2025

[arXiv link](https://arxiv.org/abs/2406.09068)

## Key Points
* Offline MARL research is plagued wih inconsistencies in baselines and evaluation protocols.
* There is a lack of clarity about changing meaning that comparing performance is impossible.
* Naming MARL algorithms is also difficult: have to specify communications etc.
* Contribution: outlines method papers should follow:
  * Settings: choose 2-3, 3-4 different scenarios in each, range of dataset quality and use common datasets
  * Baselines: 3-4 baselines (including behavioural cloning) and make sure they are well known.
  * Training and eval parameters: set training budget, do regular evals but don't include the data in training as this
  would make it online, baseline set at 32 evals but test
  * Results: do 10 random seeds and include mean and std of results