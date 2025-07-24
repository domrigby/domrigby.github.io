# Model-Based Meta Automatic Curriculum Learning

Date read: 22nd July 2025

[OpenReview link](https://openreview.net/forum?id=Kp716SJ5dbJ)

## Key Points
* Train model to predict performance on new task based on performance on old tasks. Context is given as set of tasks and 
a set of losses which it encodes using an LSTM.
* Takes the current level and then greedily selects a new level based on the highest performance improvement.
* They call this model the **learning dynamics model**
* Why?
  * States: goal of curriculum learning is to choose a set of tasks w that maximise the learning of each step
  * Meta-curriculum learning aims to identify a policy which selects these tasks to maximise learning progress over multiple
  lifetimes.