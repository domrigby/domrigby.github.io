## CHIRP:CHange Induced Regret Policy Metrics for Lifelong RL
[Arxiv link](https://arxiv.org/abs/2409.03577)

### Key Points
- Proposed a metric called $SOPR$ (Scaled Optimal Policy Regret) which quantified how much expected reward is lost 
when the environment changes. This is however infeasible to compute in anything other than simple environments.
- Proposed CHIRP (CHange Induced Regret Policy) requirements for a computable metric to replace SOPR.
- The above is very difficult to compute, therefore another metric was proposed for measuring similarities between 
different Markov Decision Processes: $W_{1}-MDP$
- This metric is for both discrete and continuous action spaces and is cheap to calculate.

### Key Methods
- CHIRP Policy Reuse (CPR): used for multi-task training. It was used to cluster different tasks and then to use the same policy on similar tasks.