# PaperDiary
I have been keeping up to date with deep learning but trying to read at least one paper a day. 

I recently decided to keep track of this using this repository!

Note: I often update this on my phone or iPad when I am out and about. Formatting is then tuned on my laptop
in the followinf days.

## Interesting Methods and Lesson Learned
This section outlines in a sentence or two some of the most interesting methods and common themes.

### Reinforcement Learning
1. **LLM reasoning self-play**, rooted to reality by a coding engine ([Absolute Zero Reasoner](May%202025/23rdMayAbsoluteZeroReasoner.md)).
    - The dynamic was for the LLM to both design coding problems as well as solve them. Interesting to see how this 
      could be applied in other domains, with different environments rooting them to reality (e.g. robotics)
2. Unsupervised self-improvement via self-dialog games. VLMs can self-improve by playing a
Guess Who like game with in domain images. 
3. RL performance can be improved (especially in LLMs) by only training on **high entropy** tokens.
These are 'forks in the road' and affect performance the most. Other tokens are pre-decided by thess high entropy tokens.
Training on these alone delivers performance improvements. ([80:20 Rule](June%202025/11thJuneTokenEntropyRLVR.md))
4. ProRL and Alpha Zero both filter out prompts which the agent always got wrong or right.
You want to exist in the **zone of proximal development**.
5. Non-Verifiable Reward Models: Writing-Zero proposed an interesting method for training preference models in non-verifiable reward 
environments. This preference was then used a reward function in a two player creative writing game.
6. **MARL policies as conditional sequences**. [JointPPO](May%202025/28thMayJointPPO.md) proposed a two-step approach to MARL: 
    1) **Ordering network** orders agent on how important their decision is.
    2) **Recurrent action-conditioned** network (transformer in this case but could be LSTM) produces actions conditioned on previous agents action.
7. **Synthetically expand experience replay**: [SynthER](May%202025/29thMaySynthER.md) trains a diffusion model to expand the experience buffer. It then trains on mixed real-fake batches
8. **Reasoning via games**: [Play to Generalise](June%202025/16thJuneReasoningThroughGames.md) showed reasoning capabilities can deciding the best move in a game. This 
can induce specific reasoning capabilities dependent on the game.
9. SEAL used RL to produce self-edits and hyperparameters to tune itself. These self-edits were synthetic data aimed at baking in knowledge or adapting to a new task as quickly as possible.


### Training in General
1. **Heterogenous pre-training for robotics**: [Pi0.5](May%202025/24thMayPi0.5VLA.md) trained on video datasets in order to learn transferable skills for robotics.
2. [RL-Pre-Training](June%202025/10thJuneRLPretraining.md): allow model to reason during pre-training for every token. This can also be used for fine-tuning.
3. **Higher clip parameter in PPO/GRPO**: [ProRL](June%202025/4thJuneProlongedRL.md) showed this benefits exploration as it encourages increasing probablity of previously unlikely tokens.
4. [ProRL](June%202025/4thJuneProlongedRL.md) showed that GRPO could be improved by adding KL divergence term against a constantly updated reference policy. 
[Play to Generalise](June%202025/16thJuneReasoningThroughGames.md) however preferred to not use a KL divergence term increase exploration.
5. **Reasoning agent produce best and worst move**: [Play to Generalise](June%202025/16thJuneReasoningThroughGames.md) showed that performance could improve from 
simultaneously getting the model to output the best and worst move (increased game understanding)
6. **Always put the environment on the GPU if possible** [(JaxMARL)](June%202025/5thJuneJaxMARL.md)
7. **Be careful when using Qwen and Clipping for RL**: [RL with Spurious Rewards](June%202025/3rdJuneSpuriousRewardsRL.md) showed that Qwen
tends to improve when its used for RL even if the rewards are completely random. This is because it:
   1. Encourages it to write more code.
   2. GRPO clipping concentrates probability on high likelihood behaviours.

### Robotics
1. Pre-training can be effective in robotics like it is in LLMs [Pi0.5](May%202025/24thMayPi0.5VLA.md)
2. [Mimic One](June%202025/17thJuneMimicOneDexterousHand.md) suggested predicting action chunks rather than single actions to encourage
temporal consistency.
3. [Mimic One](June%202025/17thJuneMimicOneDexterousHand.md) also used a U-Net based diffusion model as the policy.
[Pi0.5](May%202025/24thMayPi0.5VLA.md) also uses a diffusion model to create fields to control the robots.

### Open-Endedness and Auto Curriculums
1. Poet ranks levels by the agents win rate.
2. Prioritised Level Replay ranks previous levels on their temporal differencd error to rate how useful they are.