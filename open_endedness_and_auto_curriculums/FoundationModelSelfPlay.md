# Foundation Model Self-Play: Open-Ended Strategy Innovation via Foundation Models

Date: 11th July 2025

[arXiv link](https://arxiv.org/abs/2507.06466)

## Key Points
* This paper present **self-play via foundation models** not a self-play foundation model.
* Foundation models write **programmatic policies** i.e. policies written in code.
* The policies often include techniques from the literature such as Kalman filters in Car Tag.
* Method: **foundation models act as intelligent search operators**. They see both teams policies, the score and then suggest improvements.
* Proposes three techniques:
  1. Vanilla Foundation Model Self-Play (vFMSP): single policy for each team, sees policies and suggests improvements.
  2. Novelty-Search Self-Play: aims to create diverse set of solutions which aren't focused on performance
  3. **Quality-Diversity Self Play**: aims for diversity and performance. Two populations compete. When new policies are created, 
  they are always kept if they inhabit an empty bit of the embedding space.
* Implementation detail: FM generates code and then runs a set of unit tests, returning any errors for the FM to fix.
* *Future research**: using LLMs to create reward functions for more complex environments.