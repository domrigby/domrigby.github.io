# Assessing the Zero-Shot Capabilities of LLMs for Action Evaluation in RL

Date: 20th July 2025

[arXiv link](https://arxiv.org/abs/2409.12798)

## Key Points
* Aim:
  * Tackle issue of credit assignment in RL by **using LLMs for reward shaping and options** (this is a job usually done
  by humans). These both breakdown the task into 'smaller, composable sub-goals'.
  * This aims to transfer LLM knowledge of the game into the agent by shaping its behaviours.
  * LLM acts as a critic: it evaluates how useful the action was in achieving the final objective rather than generating
  itself.
* How:
  * LLM gets to decide what the useful sub-goals are in the environment (suggestions are included in the prompt for some 
  experiments).
  * LLM gets given a prompt describing the game and a state-action-state transition. It then has to return a binary signal
  on whether a sub-goal has been reached.
  * If yes... the agent gets an intermediate reward added to its reward.