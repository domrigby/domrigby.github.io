# Prioritised Level Replay

18th June 2025

[Paper link](https://arxiv.org/pdf/2010.03934)

## Key Points
- Generates 'levels' for games to be as 'learnable' as possible.
- Levels are of most interest to agent.
- Natural curriculum emerges it has to introduce easy levels at 
start which gradually become more difficult.
- Probability distribution to sample old leveks is bias towards interesting levels.

## Key Methods
- Orders levels by their temporal difference error.
- Levels have staleness rating to discourage continuously trying and failing at one environment.
- Each episode decides whether to randomly create new level or replay old one.