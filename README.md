# PaperDiary
I have been keeping up to date with deep learning but trying to read at least one paper a day.

I recently decided to keep track of this using this repository!

## Interesting Methods
This section outlines in a sentence or two some of the most interesting methods and common themes.

### Open-Endedness

### Reinforcement Learning
1. **LLM reasoning self-play**, rooted to reality by a coding engine (Absolute Zero Reasoner).
    - The dynamic was for the LLM to both design coding problems as well as solve them.
Interesting to see how this coupd be applied in other domains, with different environments rooting them to reality (e.g. robotics) 
2. Unsupervised self-improvement via self-dialog games. VLMs can self-improve by playing a
Guess Who like game with in domain images.
3. RL performance can be improved (especially in LLMs) by only training on **high entropy** tokens.
These are 'forks in the road' and affect performance the most. Other tokens are pre-decided by thess high entropy tokens.
Training on these alone delivers performance improvements.  

### Training in General
1. **Heterogenous pre-training for robotics**: Pi0.5 trained on video datasets in order to learn transferable skills for robotics.
2. RL-Pre-Training: allow model to reason during pre-training for every token. This can also be used for fine-tuning.
 