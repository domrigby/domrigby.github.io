# Automatic Curriculum Learning for Driving Scenarios: Towards Robust and Efficient Reinforcement Learning
[Arxiv link](https://arxiv.org/pdf/2505.08264v1)

Date: 23rd June 2025.

## Key Points
- Aims to build a self-driving car auto-curriculum learning framework
- Why? Fixed scenarios = overfitting, domain randomisation = sample inefficiency
- Paper introduces graph based method for generating roads and driving scenarios
- Auto-curriculum is made up as followed:
    1. Random generator for creating new levels
    2. Editor: perturbs high learning potential levels
- Learning potential: positive value loss (temporal difference error)