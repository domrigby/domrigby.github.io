# OMNI-EPIC: Open-Endedness Via Models of Human Notions of Interestingness With Environments Programmed In Code

Date: 28th June 2025

[arXiv link](https://arxiv.org/abs/2405.15568)

# Key Points
- Creates an open-ended learning framework based around leveraging the intuition and logic baked into LLMs and foundation models
- Process:
    1. Most 'interesting' task is selected (covered later)
    2. A new more interesting task is generated using an FM. The **tasks are all described in text**.
        - RAG is used to retrieve the N most similar tasks to give the LLM context of what the agent has already seen.
        - RAG is done using OpenAI's text-embedding-3-small model
    3. Environment generator converts the description into a gym style environment.
    4. Model of Interestingness rates how interesting the new environment is.
        - Tries to mimic human concepts on interestingness it has learned from human text.
    5. Success detector (LLM or VLM) confirms whether or not the agent succeed at the task. 