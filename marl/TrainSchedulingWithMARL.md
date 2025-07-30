# Intelligent Railway Capacity and Traffic Management Using Multi-Agent Deep Reinforcement Learning

Date read: 30th July 2025

[Paper link](https://stefanbschneider.github.io/assets/other/marl_itsc2024.pdf)

## Key Points
* Aim: perform planning and operations for train network to create optimal schedules to increase capacity and robustness
of the rail network.
* Introduces a multi-agent reinforcement learning based approach to Capacity and Traffic Management Systems (CTMS)
* Uses a realistic railway transport simulator which allows you to import railway maps.
* MARL:
  * Treats problem as each train deciding where they will go locally.
  * Uses APEX-DQN, a distributed DQN algorithm, but with **shared weights** such that the agents can share insights learned 
  from all over the map.
  * Observation: a tree graph of what is in front of the train (points, stations, other trains etc.) with varying depth depending on compute.
  It also has access to some observations about distant trains (track numbers, speed destination) but doesn't elaborate how.
  * Action space: agent has a discrete set of options: 1) tune train speed 2) change points (choose between two railways).
  * Rewards: +1 for reaching each stop (more stops serviced = more reward). This is encouraged to be quick by the gamma factor.