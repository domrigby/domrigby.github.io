# Environment Scaling for Interactive Agentic Experience Collection: A Survey

**Date:** 30th September 2026

[arXiv Link](https://arxiv.org/abs/2511.09586) | [PDF](https://arxiv.org/pdf/2511.09586) | [Code](https://github.com/lukahhcm/Awesome_Environment_Scaling)

## Key Points
* Interesting paper om how we 'scale' the environment for the Era of Experience. This is not 'how do we run more environments?', its 'how do we create environments which lead to the most intelligence?'.
* A three staged process is proposed:
    1. **Generate**: environments are collected in which the agent can collect interested experiences.
    2. **Execute**: the agent collects experiencs
    3. **Feedback**: the agent reflects and learns from those experiences.

### Sub-Axes
* **Generation**:
    * **Complexity**: underlying difficulty of the task. Can scale depth and width of task. This often exists in the context of LLMs as more required tool calls or more objectives needing to be met
    * **Dynamics**: selecting the difficulty of the tasks the agent does, what it sees and what it can do. Crucially, you can give the agent tasks to focus on particular skills it is lacking. Too easy or too hard, then learning is far less efficient. Must exist in the zone-of-proximal development. 
    * **Diversity**: making sure tasks encourage wide range of skills to prevent overfitting
* **Execution**: 
    * **Interactivity**: how much can the agent control? Range from static datasets up to full controllable environments
    * **Realism**: how reprentative is the environment?
* **Feedback**:
    * **Density**: how often the agent gets feedback
    * **Granularity**: type of feedback received. Is the agent scored on multiple goals?
    * **Automation**: how much can we automated and generate this feedback quickly?
    * **Ojectivity scaling**: how bias is our feedback? ranges from verifiable domains to non-vefiable