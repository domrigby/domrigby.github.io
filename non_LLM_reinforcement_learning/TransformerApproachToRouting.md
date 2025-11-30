# Attention, Learn to Solve Routing Problems

Date read: 11th October 2025

[Paper link](https://arxiv.org/pdf/1803.08475)

## Key Points
* Aims:
	* Learn heuristics for solving combinatorially complex routing problems.
	* The aim is **not to outperform specialist algorithms, but to show flexibility of approach**

* Examples of previous attempt:
	* Pointer networks to pick next actions
	* Many encoder-decoder methods
	* Difference in previous work tends to focus on how to form action space. E.g:
		* Pick turns
		* Order nodes
		* Output adjacency matrix.

* Architecture:
	* Multi-headed attention
	* Input: node embeddings enter fully connected encoder
	* Backbone: **Graph Attention Network**
	* Output: permutation of the nodes (best visiting order)
		* Decoder: receives node embeddings + decoder context (current node + end node) with already visited nodes masked out. A probability distribution over the remaining nodes is the output.
		* Probability is calculated via a single headed self-attention mechanism.
	* Generality: mode backbone might be the same between different problems, but the input and output layers and the decoder context might need to change.

* Algorithm:
	* REINFORCE
	* Baseline: reward from solving the problem in a greedy manner with a recent policy.
	* Reward: distance across entier tour










