# Forecaster: Towards Temporally Abstract Tree-Search Planning from Pixels

Date read: 2nd August 2025

[arXiv link](https://arxiv.org/abs/2310.09997)

## Key Points
* RL agents tend to suffer from being attracted to short term rewards over long term objectives. **Forecaster aims to stop
this by creating a temporally abstracted world model**.
* **Manager-Worker Dynamic**:
  * Manager:
    * Picks goals for the worker
    * Performs MCTS ont these time abstracted goals
    * Made up of:
      * World model bit: encoder and decoder, dynamics and reward predictor (on latent states). Similar to MuZero.
      * Goal autoencoder: state -> goal -> back to state
      * Manager network: maps from state to goal
  * Worker: aims to achieve goals set by manager
    * Chooses actions according to state and goal
