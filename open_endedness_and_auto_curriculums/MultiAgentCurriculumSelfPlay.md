# Multi-Agent Training for Pommerman: Curriculum Learning and Population-based Self-Play Approach

Date: 25th June 2025

## Key Points
- Aims to learn the game of Pommerman, a multi-agent competitive game which has:
    1. Sparse rewards
    2. False positives
    3. Delayed action effects
    4. Very complex exploration (e.g. agent must place bombs to blow up walls etc)
- This paper proposed a method to solve this game using two main avenues: curriculum learning and self-play
- Curriculum learning:
    - Three **hardcoded** opponents which encourage the agent to learn specific skills.
    - Once you defeat them, you upgrade to a harder one.
    - **Performance annealed dense exploration reward** rewards the agent for exploring when the agents performance is low, and anneals this when it starts performing well.
- ELO based population self-play:
    - Probability of playing against an another agent is weight by its ELO score.