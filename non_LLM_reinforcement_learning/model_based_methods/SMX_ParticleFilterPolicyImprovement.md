# SMX: Sequential Monte Carlo Planning for Expert Iteration

Date read: 11th August 2025

[arXiv link]

## Key Points
* Introduces SMX: a particle filter based planning method
* Why?
  * **Parallelisable** (and therefore GPU accelerated) planning method.
  * **Discrete or continuous actions** naturally
  * No need to maintain large tree in memory
  * Sample efficient (compared to AlphaZero)
* How?
  * Expert Iteration framework with two levels of optimisation: 1) game play level 2) planning level. 
  * Rollouts provide a policy improvement target for the networks to learn
  * Particle filter:
    * Start with N particles
    * Propagate forward via action sample
    * Update weights in correlation with advantage (good actions get heavier weights)
    * Every M steps we resample the distribution to overcome _weight degeneracy_ (one particle dominating)
    * Weighted particles should converge to target distribution
  * Particle distribution over there starting actions is used to train the policy by minimsig the KL divergence between them.
* Method has to be stabilised using the following methods:
  1. Normalising advantage updates (similar to MuZero action-value normalisation)
  2. Low softmax temperature when resample distribtion of particle (they used 0.1)
  3. Perform particle resampling every P steps (they found 2 or 4 worked well)