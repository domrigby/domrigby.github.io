# TorchBeast: A PyTorch Platform for Distributed RL

Date: 5th July 2025

[arXiv link](https://arxiv.org/abs/1910.03552)

# Key Points:
* PyTorch version of IMPALA
* Attempts to balance high performance low level code with ease of implementation of PyThon.
  * E.g. batching is done by C++.
* Presents a high performance server set up: 
  * Many environment servers: send observations and rewards
  * Client server: does neural network processing. Receives observations and sends back actions.
  * One learner server: learns.
