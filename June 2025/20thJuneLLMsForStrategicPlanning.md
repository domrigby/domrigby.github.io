# Agents of Change: Self-Evolving LLM Agents for Strategic
Date: 20th June 2025
[Arxiv Link](https://arxiv.org/pdf/2506.04651)

## Key Points
- Introduce a concept of **self-evolving** LLMs to perform long-horizon strategic planning.
- Evolving seemed to rely on the agents ability to reason as to why things went wrong.
- Tested four different types of agents:
  1. BaseAgent: LLM takes in state and gives out actions
  2. StructuredAgen: LLM receives state + lots of human designed info about the game such as basic strategies
  3. PromptEvolver: same as above but there is another agents whos goal it is to adapt the prompt to induce best performance. It
has access to the web, previous results etc.
  4. AgentEvolver: team of agents work together all with different roles: Evolver, Analyser, Research, Coder, and Player.
These agents all converse with the aim of improving performance.
- This was **not SFT** they iteratively improved performance by updating the prompts which the agent was using to make decisions
by giving the agent the results of their previous actions.
- Showed LLMS can team up to improve performance!