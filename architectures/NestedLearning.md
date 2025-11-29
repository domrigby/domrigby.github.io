# Introducing Nested Learning: A new ML paradigm for continual .

Date read: 6th November 2025

[Blog link](https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/)

## Key Points
* Aims to create an paradigm in which **optimising the weights and choosing the architecture are part of the same problem**.
* Views learning as a solving a **smaller optimisation problems**
* Only optimising smalls parts of the network at any one time means we **can avoid catastrophic forgetting**
* Each smaller optimisation problem (model) has its own unique context flow and problem to solve.
* Utilises a **hierarchical architecture** with different levels being called at different frequencies

* Architecture: HOPE
	* Modifies Titan architecture which prioritised memories based on how surprising they are
	* Adds **recurrent element to add unlimited in-context learning** (no limit on context window)