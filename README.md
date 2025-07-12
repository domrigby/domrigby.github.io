<h1 align="center">Paper Diary</h1>

<p align="center">By Dom Rigby</p>

---

## 📌 Introduction

Welcome to my Paper Diary! Due to the seemingly never ending supply of interesting reinforcement learning papers which have come out in the last few years, I began
to try and read at least one every day. I was however having the issue that after a month of two I could not remember for the life of me where I had read that interesting fact,
method or algorithm. I therefore began keeping a diary of the papers/blog posts I was reading. I recently decided I may as well start posting these incase anyone else find this
interesting or useful!

Entries are added on the go (often from my phone or iPad) and later refined on my laptop.

> **Note:** Layout and formatting are continuously improved when time permits.


## 🛠️ Method

####  Identification of Papers
   1. **X (Twitter)**: there is a huge AI community on twitter which post papers with discussion in the comments.
      * **TIP**: If others choose to use this I would highly recommend using the 'Not Interested' feature on posts, otherwise your feed will rapidly deteriorate and show less papers.
   2. **Reddit**: r/MachineLearning 
   3. **Conferences**: I recently attend ICLR and came back with a treasure trove of interesting reads.
   4. **Paper references**

#### Use of LLMs
   1. **LLMs are NOT used for the analysis of the papers**. They are however **used for checking**. I read the paper, write down what I think the key points are.
      I then ask o4-mini-high to do the same and double check if we disagree.
   2. Paper recommendations
   3. Formatting and helping with markdown.
   4. Quick analysis scripts.


---

## ⚙️ Website Workings

This website is a user-friendly entry point and summary of the repository. This hosts the top level themes and parts I thought were interesting.
All paper summaries are stored in **[this repository](https://github.com/domrigby/domrigby.github.io)**

## Fun Plots

Below is a plot of a t-SNE dimensionality reduction 

<iframe
  src="tsne_papers.html"
  width="100%"
  height="600"
  frameborder="0"
  title="t-SNE of Paper Descriptions">
</iframe>
---

## 🔍 Highlights & Lessons Learned

The following section includes:
   * **Interesting ideas**: any ideas I saw in papers which might be useful if someone is tackling a similar problem.
   * **Concise fundamentals**: try and explain the fundamentals of a topic in a few short bullet points!

### 1. Reinforcement Learning (RL)

1. **High‑Entropy Token Training**\
   Training only on high‑entropy (“forks in the road”) tokens yields significant performance gains in LLMs. ([80:20 Rule](LLM_reinforcement_learning/TokenEntropyRLVR.md)). Many tokens in language are determined
   by other words so provide little information in te RL process when they are chosen. E.g. "I went to the shop", "to" and "the" are determined by the other words
2. **Zone of Proximal Development**\
   Methods like [ProRL](LLM_reinforcement_learning/ProlongedRL.md) and [Absolute Zero Reasoner](LLM_reinforcement_learning/AbsoluteZeroReasoner.md) filter out consistently correct or incorrect prompts to focus learning in the optimal difficulty zone.
   This is discussed in detail in [section 2](#2-openendedness--autocurricula).
3. **It is possible to make Non‑Verifiable Reward Models**
   [Writing‑Zero](LLM_reinforcement_learning/WritingZeroNonVerifiableRewards.md) introduces LLM based prefserence training in non‑verifiable environments,
   then uses that model as a reward in competitive creative writing games.
4. **You can use generative AI to expand experience buffer**\
   [SynthER](non_LLM_reinforcement_learning/SyntheticExperienceReplay.md) trains a diffusion model to expand the replay buffer with synthetic experiences for mixed real‑fake training.
5. **You can learn to reason by simply playing games**\
   [Play to Generalise](LLM_reinforcement_learning/ReasoningThroughGames.md) demonstrates that game‑based move prediction enhances specific reasoning capabilities.
6. **GPU‑Accelerated Environments provide monumental speeds up**\
   Frameworks like [Kinetix](distribution_and_gpu_acceleration/KInetixGeneralRL.md) and [JaxMARL](marl/JaxMARL.md) allow you to run tens of thousands of environments in parallel, as well as minimise CPU-GPU overhead.
7. **Foundation Models roles in RL:**
   Foundation models have intuition about what humans find interesting. They are therefore capable of designing curriculums for RL or being involved in the policy improvement steps. 
   See more in the [open-endedness section of this blog](#4-openendedness--autocurricula). Summary of a few interesting methods:
      * Create environments of interest ([OMNI-EPIC](open_endedness_and_auto_curriculums/OpenEndednessUsingLLMS.md))
      * Writing code based policies and suggesting improvements after view results ([Foundation Model Self-Play](open_endedness_and_auto_curriculums/FoundationModelSelfPlay.md))

### 2. Open‑Endedness & Auto‑Curricula
1.  **Open-Endedness Requires Novel and Learnable Artefacts**\
   Open-ended is defined in [Open-Endedness is Key to ASI](open_endedness_and_auto_curriculums/OpenEndednessIsKeyToASI.md).
   A system is open-ended if it **continually creates novel and learnable artefacts**. This is dependent on the observer 
   and the time-horizon. E.g. a mouse can't learn chess and a computer will eventually plateau in performance.
2. **Procedural Level Generation** is used to create novel environments to learn in
   [POET](open_endedness_and_auto_curriculums/EnhancedPOETOpenEndedLearning.md) introduces new levels, checks they meet a minimum learnability criterion and then only adds the most novel.
3. **Prioritized Level Replay** is way to order those environments such that they are **learnable**. This creates an **auto-curriculum**.
   [Prioritized Level Replay](open_endedness_and_auto_curriculums/PrioritisedLevelReplay.md) suggest ranking levels by temporal‑difference error.
   Other methods include simply [filter out examples in which the agent completely fails or always succeeds](LLM_reinforcement_learning/AbsoluteZeroReasoner.md)
4. **Randomly generate a new level, or create a new one!**: this creates population or pool of environments for the agent to interact with
   [Auto-Curriculum Learning for Driving Scenarios](open_endedness_and_auto_curriculums/AutoCurriculumAutonomousDriving.md), [POET](open_endedness_and_auto_curriculums/EnhancedPOETOpenEndedLearning.md) and many others methods introduces the idea of random generator + editor as the basic building blocks for creating levels. One creates random new levels 
   and the other perturbs existing interesting levels.
5. **Foundation models can act as 'intelligent search operators** to create new learning opportunities based on what they have learned that humans would find interesting.
   This is suggest as a ['key method on the road to ASI'](open_endedness_and_auto_curriculums/OpenEndednessIsKeyToASI.md). and is explored for level generation in [OMNI-EPIC](open_endedness_and_auto_curriculums/OpenEndednessUsingLLMS.md)
   and for policy generation is [Foundation Model Self-Play](open_endedness_and_auto_curriculums/FoundationModelSelfPlay.md)
6. **Performance annealed exploration reward**:
   [Curriculum Learning and Population-based Self-Play](open_endedness_and_auto_curriculums/MultiAgentCurriculumSelfPlay.md) suggests using an exploration reward
   which is annealed according to agent performance. It therefore explores more when it is doing badly and exploits when it is doing well.

### 3. Pretraining & General Training Tips

1. **Heterogeneous Pretraining: think outside the box when it comes to data**\
   [Pi0.5](robotics/Pi0.5VLA.md) and [V‑JEPA](general_training/V-JEPA2.md) both use video data to train robotics models. This video still contains information of interst to robotics.
   Pre-training data can come from a wide range of sources!
2. **Reasoning with Next Token Prediction (RNTP)**: (allowing the model to reason about the next token during pre-training) \
   * [RL‑Pre‑Training](LLM_reinforcement_learning/RLPretraining.md) suggests using next token prediction for RL but only applies in fine-tuning.
   * [Jack Morris' blog post on scaling RL](LLM_reinforcement_learning/ScalingRLto10^26FLOPS.md) suggest that this might be way to squeeze the absolute maximum 
   out of our ['fossil fuel-like'](https://www.youtube.com/watch?v=YD-9NG1Ke5Y) internet data. Next token prediction is verifiable so should allow us to get further performance on this internet dataa. We just
   need to work out how to scale LLM RL (see blog post and summary for further details).
3. **When doing PPO/GRPO, make the upper bound clip larger ($|\epsilon_{clip, high} - 1|> |1 - \epsilon_{clip, low|}$)**\
   The upper clip bound being higher increases the probability of unlikely choices and increases exploration (as in [ProRL](LLM_reinforcement_learning/ProlongedRL.md) and [Play to Generalise](LLM_reinforcement_learning/ReasoningThroughGames.md)) improve exploration and stability.
4. **Dual‑Outcome Reasoning: knows what's bad is also useful!**\
   Generating both best and worst moves in game scenarios deepens model understanding of decision boundaries ([Play to Generalise](LLM_reinforcement_learning/ReasoningThroughGames.md))
5. **Always use a GPU based environment when possible**\
   Always host simulation environments on the GPU when possible. This allows you to run tens of thousands of environments in parallel ([JaxMARL](marl/JaxMARL.md), [Kinetix](distribution_and_gpu_acceleration/KInetixGeneralRL.md))
6. **Beware When Using Qwen for RL**\
   [RL with Spurious Rewards](LLM_reinforcement_learning/SpuriousRewardsRL.md) shows that random reward signals can still drive code production due to clipping effects.
7. **Telling the model how to think improves performance**\
   [FinCoT](finance_applications/FinCoT.md) improved performance by giving the reasoning model **strucutred chain-of-thought prompts. For finance problems, methods to solve certain types of problems are well known, or at least the important things to look for.
   These chain of thought patterns are generated using DeepResearch and then added to the prompt after the question as a suggestion of how to think.

### 4. Robotics & Control

1. **Action Chunk Prediction**\
   [Mimic One](robotics/MimicOneDexterousHand.md) predicts chunks of actions to enforce temporal consistency.
2. **Diffusion‑Based Policies**\
   Diffusion models generate continuous action fields for robot control ([Pi0.5](robotics/Pi0.5VLA.md), [Mimic One](robotics/MimicOneDexterousHand.md)).
3. **Frame Prediction for Planning**\
   [V‑JEPA](general_training/V-JEPA2.md) pretrains on millions of videos to predict missing frames, then fine‑tunes on robotic datasets for causal understanding and planning.
4. **Pre-Training is Possible in Robotics**
   * [V-JEPA](general_training/V-JEPA2.md) and [Pi0.5](robotics/Pi0.5VLA.md) both used huge amounts of internet video data to train world models to predict actions and effects.

### 5. Distribution

1. This [blog post by Jeremy Jordan](distribution_and_gpu_acceleration/TrainingOnThousandsOfGPUs.md) covers the basics of how to 
   train a network on thousands of GPUS. Some of the key methods spoke about were:
   * **Types of parallelism**:
     1. **Data parallelism**: each GPU has a copy of the model and a different batch of data. They then share gradients to do joint updates.
     2. **Model parallelism**: for large models. Model layers are split over many GPUs.
   * **Communication methods**: 
     1. Scatter: send different data to each GPU
     2. Broadcast: same data to all
     3. Reduce: combine all data on one GPU.
2. This [blog post on distributed PPO](distribution_and_gpu_acceleration/DistributedPPO.md) outlines some extra factors to think about:
   1. **Synchronous**: waits for all agents to calculate their respective gradients before doing a weights update.
   2. **Asynchronous**: doesn't wait.
   3. **Centralised**: single server does all gradient accumulation and weights updates.
   4. **Decentralised**: all share gradients (all-reduce) but have their own model.
3. [IMPALA](distribution_and_gpu_acceleration/IMPALA_DistributedRL.md) outlines a now common, distributed reinforcement learning method with multiple actors and a single centralised learner 
 which broadcasts weights update. This is mimicked in PyTorch in [TorchBeast](distribution_and_gpu_acceleration/TorchBeastDistributedPyTorch.md)
4. [Docker](distribution_and_gpu_acceleration/DockerInRL.md) can be used like a lightweight virtual machine for distributing actors or learners across large clusters.

### 6. Multi‑Agent Reinforcement Learning (MARL)

1. **Stabilise MARL by condition agents actions on the actions of other agents**\
   [JointPPO](marl/JointPPO.md) orders agents by decision importance, then uses a recurrent action‑conditioned network to generate actions sequentially
2. **GPU based environments are key to tackling to complexity of MARL**
   [JaxMARL](marl/JaxMARL.md) allows you to run the environment tens of thousands of times in parallel. This means the monumental search space can be explore
   a bit more thoroughly.
3. Population‑based methods prevent overfitting and foster diverse behaviors.
4. Agent selection via ELO‑weighted sampling encourages robustness and competitive balance.
   This is used in [Foundation Model Self Play](open_endedness_and_auto_curriculums/FoundationModelSelfPlay.md), [Multi-Agent Pommerman](open_endedness_and_auto_curriculums/MultiAgentCurriculumSelfPlay.md)

### 7. Self‑Improvement Strategies

1. **LLMs can do self-play for reasoning, as long as their grounded to reality**\
   [Absolute Zero Reasoner](LLM_reinforcement_learning/AbsoluteZeroReasoner.md) creates coding puzzles in a self-play method 
2. **Unsupervised Self‑Dialog Games**\
   VLMs play in‑domain “Guess Who” style games to self‑improve vision‑language reasoning. ([VLM Self‑Dialog Games](self_improvement/SelfDialogueGames.md))
3. **Adaptive Prompting & Team Agents**\
   [Agents of Change](LLM_reinforcement_learning/LLMsForStrategicPlanning.md) evolve prompts and orchestrate agent teams (analyst, coder, researcher) for strategic planning tasks.
4. **Self‑Adapting LLMs**\
   [SEAL](LLM_reinforcement_learning/SelfAdaptingLanguageModels.md) uses RL to generate synthetic edits and hyperparameters, enabling rapid adaptation to new tasks.


---

## ⚙️ Repository Structure

```text
├── LLM_reinforcement_learning/     # Papers on RL with language models
├── marl/                          # Multi‑agent RL resources
├── non_LLM_reinforcement_learning/ # RL methods outside LLM context
├── robotics/                      # Robotic learning and control papers
├── self_improvement/              # Self‑play and self‑dialog approaches
├── distribution_and_gpu_acceleration/ # GPU‑accelerated training methods
├── open_endedness_and_auto_curriculums/ # Curriculum learning and open‑endedness
└── README.md                      # This overview and highlights
```

---

# 📖 Full Diary

### May 2025

* 23rd: [Absolute Zero Reasoner](LLM_reinforcement_learning/AbsoluteZeroReasoner.md)
* 24th: [Pi0.5](robotics/Pi0.5VLA.md)
* 25th: [TD-MPC2](non_LLM_reinforcement_learning/TDMPC2.md)
* 26th: [JointPPo](marl/JointPPO.md)
* 29th: [Synthetic Experience Replay](non_LLM_reinforcement_learning/SyntheticExperienceReplay.md)
* 30th: [Reinforcing Multi-Turn Reasoning in LLM Agents via Turn-Level Credit Assignment](LLM_reinforcement_learning/MultiTurnCreditAssignmentLLMRL.md)

### June 2025
* 1st: [Ultimate Guide to Supervised Fine-Tuning](general_training/UltimateGuideToSFT.md)
* 2nd: [Learning Complex Dexterous Manipulation with Deep Reinforcement Learning and Demonstrations (Demo Augmented RL)](non_LLM_reinforcement_learning/DemoAugmentedRL.md)
* 3rd: [Spurious Rewards: Rethinking Training Signals in RLVR](LLM_reinforcement_learning/SpuriousRewardsRL.md)
* 4th: [ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Models](LLM_reinforcement_learning/ProlongedRL.md)
* 5th: [JaxMARL: Multi-Agent RL Environments and Algorithms in JAX](marl/JaxMARL.md)
* 8th: [Illusion of Thinking: Understanding Strengths and Limitations of Large Reasoning Models (LRMs)](LLM_reinforcement_learning/IllusionOfThinking.md)
* 9th: [CHIRPs: Change-Induced Regret Proxy Metrics for Lifelong Reinforcement Learning](non_LLM_reinforcement_learning/CHIRPLifeLongRL.md)
* 9th: [Enhanced POET: Open-Ended Reinforcement Learning through Unbounded Invention of Learning Challenges and their Solutions](open_endedness_and_auto_curriculums/EnhancedPOETOpenEndedLearning.md)
* 10th: [Reinforcement Pre-Training](LLM_reinforcement_learning/RLPretraining.md)
* 11th: [Reinforcement Learning Teachers of Test Time Scaling](LLM_reinforcement_learning/CreatingRLTeachers.md)
* 11th: [Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning](LLM_reinforcement_learning/TokenEntropyRLVR.md)
* 12th: [Writing-Zero: Bridge the Gap Between Non-verifiable Tasks and Verifiable Rewards](LLM_reinforcement_learning/WritingZeroNonVerifiableRewards.md)
* 16th: [Play to Generalize: Learning to Reason Through Game Play](LLM_reinforcement_learning/ReasoningThroughGames.md)
* 17th: [mimic-one: a Scalable Model Recipe for General Purpose Robot Dexterity](non_LLM_reinforcement_learning/DemoAugmentedRL.md)
* 18th: [Prioritised Level Replay](open_endedness_and_auto_curriculums/PrioritisedLevelReplay.md)
* 19th: [Self-Adapting Language Models](LLM_reinforcement_learning/SelfAdaptingLanguageModels.md)
* 20th: [Agents of Change: Self-Evolving LLM Agents for Strategic Planning](LLM_reinforcement_learning/LLMsForStrategicPlanning.md)
* 21st: [KINETIX: INVESTIGATING THE TRAINING OF GENERAL AGENTS THROUGH OPEN-ENDED PHYSICS-BASED CONTROL TASKS](distribution_and_gpu_acceleration/KInetixGeneralRL.md)
* 22nd: [Superintelligence From First Principles (blog post)](open_endedness_and_auto_curriculums/SuperintelligenceFromFirstPrinciples.md)
* 23rd: [Automatic Curriculum Learning for Driving Scenarios: Towards Robust and Efficient Reinforcement Learning](open_endedness_and_auto_curriculums/AutoCurriculumAutonomousDriving.md)
* 24th: [How Visual Representations Map to Language Feature Space in Multimodal LLMs](general_training/SharedRepresentationsInVLMs.md)
* 25th: [Multi-Agent Training for Pommerman: Curriculum Learning and Population-based Self-Play Approach](open_endedness_and_auto_curriculums/MultiAgentCurriculumSelfPlay.md)
* 26th: [Automatic Curriculum Design for Zero-Shot Human AI Coordination](open_endedness_and_auto_curriculums/AutoCurriculumForHumanAICoordination.md)
* 28th: [OMNI-EPIC: Open-Endedness Via Models of Human Notions of Interestingness With Environments Programmed In Code](open_endedness_and_auto_curriculums/OpenEndednessUsingLLMS.md)
* 30th: [Self-Supervised Video Models Enable Understanding, Prediction and Planning (V-JEPA)](general_training/V-JEPA2.md)

### July 2025
* 1st: [Open-Endedness is Essential for Artificial Superhuman Intelligence](open_endedness_and_auto_curriculums/SuperintelligenceFromFirstPrinciples.md)
* 2nd: [SPIRAL: Self-Play on Zero-Sum Games Incentivizes Reasoning via Multi-Agent Multi-Turn Reinforcement Learning](LLM_reinforcement_learning/SelfPlayZeroSumGames.md)
* 4th: [Training extremely large neural networks across thousands of GPUs by Jeremy Jordan](distribution_and_gpu_acceleration/TrainingOnThousandsOfGPUs.md)
* 5th: [IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures](distribution_and_gpu_acceleration/IMPALA_DistributedRL.md)
* 5th: [TorchBeast: A PyTorch Platform for Distributed RL](distribution_and_gpu_acceleration/TorchBeastDistributedPyTorch.md)
* 7th: [Distributed PPO Blog Post](distribution_and_gpu_acceleration/DistributedPPO.md)
* 8th: [Reinforcement Learning with Docker](distribution_and_gpu_acceleration/DockerInRL.md)
* 9th: [FinCoT: Grounding Chain-of-Thought in Expert Financial Reasoning](finance_applications/FinCoT.md)
* 10th: [Illuminating search spaces by mapping elites](open_endedness_and_auto_curriculums/MAP_Elites.md)
* 11th: [Foundation Model Self-Play: Open-Ended Strategy Innovation via Foundation Models](open_endedness_and_auto_curriculums/FoundationModelSelfPlay.md)
* 12th: [How to scale RL to 10^26 FLOPs blog by Jack Morris](LLM_reinforcement_learning/ScalingRLto10^26FLOPS.md)


&#x20;&#x20;

---

*Last updated: July 9, 2025*
