# Illuminating search spaces by mapping elites

Date: 10th July 2025

[arXiv link](arxiv.org/abs/1504.04909)

# Key Point
* Aims to find the best solution from each neighbourhood of the space.
* Tells us about how the best performance changes with different dimensions. 
* Method:
  * Uses evolutionary algorithm. Feature space is description of system in meaningful dimensions.
  * User selects dimensions of interest in which we are interested to see how performance varies.
  * Creates grid in this space
  * Creates random solutions... identifies which grid space they belong in
  * Mutates them and sees if the new mutant outperforms the version in its grid space.
* Result is a set of best solutions at each point in the space