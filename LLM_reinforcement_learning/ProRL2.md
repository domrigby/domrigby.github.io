# ProRL V2 - Prolonged Training Validates RL Scaling Laws

Date read: 13th August 2025

[Blog link](https://hijkzzz.notion.site/prorl-v2)

## Key Points
* Extends [ProRl](LLM_reinforcement_learning/ProlongedRL.md) which address stability and diminishing returns in RL.
* The aim is to run RL for indefinite number of steps and still see improvements over time (they achieved 3000 steps of improvement)
* **Improvements:**
  * Stability: KL-regularised trust regions
  * Entropy collapse:
    * Clip higher GRPO: the clip parameter is higher for the upper clip. This encourages increasing probability of rare responses.
    * Reinforce++ baselines (averages over rewards of set of rollouts from that prompt to get a baseline reward to compare against).
    * Dynamic sampling: filters out responses with 100% success or failure (bit like curriculums)
    * **Policy resets: KL-penalty against baseline policy which is rarely reset**. This maintains stability by comparing to a stable baseline which is not changing. It is updated every 100-500 steps.
  * Verbosity: cosine length penalty (**only on someitimes (~15% of the time)**)