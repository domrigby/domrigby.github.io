# Illusion of Thinking: Understanding Strongths and Limitations of Large Reasoning Models (LRMs)

Date: 8th June 2025

## Key Points
- Uses puzzles with tunable complexify rather than maths problems (these suffer from data contamination).
- Observations at different complexities:
    1. Low: non-reasoning > reasoning (more direct circuits).
    2. Medium: non-reasoning < reasoning
    3. High: both fail! Interesting, the reasoning models stop producing more tokens over a threshold of complexity.
- When given pseudocode for solving problems (e.g. tower of hanoi recursion), the models didn't use it.
- Could do many reasoning steps for tower of hanoi, but not for river crossing where large number of steps is more rare. 
This suggests performance relates to the training data rather than true logical reasoning.
- Paper suggests that reasoning improvements are solving reasoning problems rather than learning generalisable reasoning.
