## Enhanced POET: Open-Ended Reinforcement Learning through Unbounded Invention of Learning Challenges and their Solutions  
Date: 9th June 2025  
[arXiv Link](https://arxiv.org/abs/2003.08536)

### Key Points:
- Introduces Enhanced POET, improving on the original Paired Open-Ended Trailblazer to sustain never-ending generation 
of new, interesting and sufficiently challenging environments
- Open-endedness has two benefits:
  1. Never stops learning. It can learn far beyond the original task
  2. Can break down more complex tasks into smaller steps.
- Disadvantage: it is a bit random. You don't know if it will achieve your desired goal. It can go off on tangents.

### Key Methods:
- Population of agent-envrionment pairs. New environments are created by mutating the ones in which the agent has 
performed sufficiently. Mutating is done on the embedding vector of the environment.
- New environments are then assessed for difficulty (run their parent's agent on it) and ones which are in zone of proximal development stay.
- Inter-environment transfer: every N steps we test all agents on all the other environments. A fine-tuning optimisation step is done first.
- Domain-general novelty metric: quantifies “meaningful novelty” of new environments by measuring how all agents (active and archived) perform, enabling open-ended discovery.  
- Goal-switching heuristic: an efficient rule to decide when an agent should transfer from one environment to another, scaling open-ended search.  
- CPPN-based encoding: uses compositional pattern-producing networks to generate environments. 
- Universal progress measure (ANNECS): tracks the accumulated count of novel environments that are actually solved, ensuring innovation is quantified.