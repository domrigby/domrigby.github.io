# Open-Ended Learning Leads to Generally Capable Agents

Date read: 9th August 2025

[arXiv link](https://arxiv.org/abs/2107.12808)

## Key Points
* Aims to train generally capable agents which can zero-shot tasks or be fine-tuned to create high performing agents quickly
* Introduces XLand:
  * Multi-agent environment with an incredibly large number of possible scenarios
  * Procedurally generated worlds.
  * Observation: pixels of each agents view
  * Actions: 
  * Reward: creates a goal state for each agent such that they get +1 if all the conditions are met (e.g. item X must be 
  next to item Y)
* Contains a lot of maths to formalise ideas like difficulty smoothness (perturbing the environment causes smooth change in difficulty)
* Training process.
  * Broken down into three parts:
    1. Deep RL on current task 
       * Pareto dominant agents are chosen to be trained
       * V-MPO algorithm
       * Trained only on current task distribution
       * Uses RNN to remember information across time-steps
       * Value is conditioned on the goal
    2. Dynamic task generation: **auto-curriculum**
       * Population of tasks and agents with different hyperparameters
       * New tasks are evaluated with the policy and **baseline policy** (random in this case).
       * For a task to be accepted, it must be sufficiency difficult (chance of failure), not too difficult
         (chance of losing) and the baseline must be performing badly.
       * The minimum difficulty etc should be dynamically assigned throughout training
    3. **Generational training**:
       * Generalising to become good at multiple tasks
       * New policy starts from scratch each time, but old policies are distilled into new policy
       * **Self reward-play**: agents are challenged to undo their goal once they achieve it. This improves performance.