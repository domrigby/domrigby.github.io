# IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures

Date: 5th July 2025

[arXiv link](arxiv.org/abs/1802.01561)

# Key Points
* Distributed RL method: parallel actors (different random seeds) collect trajectories for the learner.
* The learner then broadcasts weight updates for all the actors
* **Problem:** there is a lag between the trajectories being collected and the update... data is therefore **off-policy**
* They introduce **V-trace**: a method for correcting for off-policy data by importance sampling weights. These weights scale down gradients
if the different between new and old policy is too large.