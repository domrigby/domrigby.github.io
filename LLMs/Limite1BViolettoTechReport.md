# Limite 1B - Violetto: Technical Report

**Date:** 10th - 17th October 2026

[PDF](https://paradigma.inc/research/limite-1b-violetto.pdf) | [Blog](https://paradigma.inc/blog/limite-1b-violetto/) | [Code](https://github.com/paradigma-inc/limite-violetto) | [Models](https://huggingface.co/paradigma-inc/models)

Related: [Scaling Up RL: Unlocking Diverse Reasoning in LLMs via Prolonged Training](../LLM_reinforcement_learning/ScalingUpRLProlongedTraining.md)

## Key Points
* **Trains mathematical reasoning model from scratch** which is designed to be used IRL, potentially with many instances in parallel.
* Proposes sample efficient method of training.
* **Interleave RL with SFT**: SFT stabilises the new capabilities introduces by RL.


## Model Architecture
* Three rounds of local attention and then global attention
* Capped logits via a sigmoid
* SwiGLU MLPs
* Grouped Query Attention with 10:2 Q:KV ratio.
* **47 layers: they prefer depth over width for reasoning**
* RoPE positional encoding
* **Regularly augments value embedding with embedding of original token**
    * Does this every three layers
    * Utilises learned $$\alpha$$ value to control mixing
* **Sparse residual mixing**
    * half-way through and at the end they readd

## Pre-Training
* Occurs on 6 nodes of 8xH100s
* Utilise a modified optimiser (check paper)
* Batch size of 524k
* **Focused on high quality dataset to maximise value**:
    * Problem solution pairs
    * Synthetic knowledge generated in high quality formats
    * Filtered web -> rewritten by LLM ti be higher quality

## Souping
* They do a lot of model souping:
    1. Get best checkpoint
    2. Train 3x in parrallel on highest quality data but with different seeds and learning rates
    3. Soup results (average weights)
* Improved performance

## Post-Training
Post-training still required as model has never been exposed to its own rollouts. It needs to get to know itself. This included multiple RL stages with stabilising steps in the middle.

### Problem Generation
* **Targetted data selection uses the whole time**: focus on areas we struggle on.
* Utilise a mix of:
    1. **Procedural generation:** set of problems which can easily have their difficulty changed. These can be produced on mass with ease. This provides the **quantity**
    2. **LLM generated problems**: provides the **quality**. Looks at areas we struggle in and provides structural variety.
* All problems have answers which are verified in code
* **Favour problems with around 50% pass rate** and then **DAPO style sampling** (get rid of 0 and 100% success)

### RL Method
* High learning rates: 1-2e-5\
* Many off policy updates for sample efficiecny (7-19)
* Few actor updates (461)
* Interleave RL and SFT
* End with critic assisted credit assignment
* **Utilise DPPO style clipping**
    * This relies of **magnitude of change of probability rather than ratio**
    * This stops unlikely tokens from having their probability increase suppressed
* Utilise Dr. GRPO normalisation (no std norm and no length norm as they bias toward shorter answers)

1. **Knowledge distillation**
    * Utilise larger model and generate 5.5million reasoning traces on maths
    * Logits distilled on 32 top log probabilities
2. **Direct Policy Optimisation**
    * Utilises high quality maths problems
    * Generate preferred using 32B model, reject using 0.6B modelk
3. RL1
    * Train until unstable
    * LR = 1.5e-5
    * 19 off policy updates
    * 704 problems, 8 rollouts each
    * 65k max context length
4. RL2
    * Double mini-batch size
    * Add more data 
    * 9 off policy updates
5. Supervised Fine Tuning 1
    * Instability of last run suggest more RL would be useless
    * Double max context length
6. Model soup
7. SFT2
8. RL3
    * **Train and utilise critic for credit assignment**
    * Lambda = 0.5^(1/L)
    

## Critic Training
* Utilise actor network with action converter removed
* Replace it with a scalar output
* Train on old rollouts then with frozen critic
* Use terminal reward (lambda = gamma = 1)


## Distribution Method
* They choose to go for B complete episodes rather waiting for buffer to fill. This is because waiting for buffer to fill biases towards shorter, easier problems. They maximise parallelisation to ammortise this cost.


