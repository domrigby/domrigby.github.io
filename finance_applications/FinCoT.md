# FinCoT: Grounding Chain-of-Thought in Expert Financial Reasoning

[arXiv link](https://arxiv.org/abs/2506.16123)

Date: 9th July 2025

# Key Points
* Improves LLM reasoning in finance applications via prompt engineering
* **Structured CoT prompting**:
  * Outlines the reasoning steps the agent should take. Finance workflows are fairly well known. Telling the model to following 
these workflows improves model performance.
  * Obtained by using DeepResearch on finance methods and then o3 to outline the steps required to solve each problem.
This is checked by a human in the loop.
* Benefit: more auditable