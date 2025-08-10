# Deep Dive into Yann LeCun’s JEPA by Rohit Bandaru

[Blog post](https://rohitbandaru.github.io/blog/JEPA-Deep-Dive/)

## Key Points
* Discusses issues with LLMs: hallucinations, limited reasoning (post was pre-reasoning models) and lack of planning.
* Big issue with AI in general:
  * **Lacks common sense**. Humans can make assumptions about tasks without even trying them, models struggle to.
  * Inability to plan
* **How humans learn**: have innate knowledge at birth from years of evolution.
* Learning to think (pre reasoning models): 
  * Models can use layers, but always have to think for the same amount of time.
  * Can think during output... but habe to generate a token preventing pure information processing. 
* Presents modes of thinking:
  * System 1: instinctive thought
  * System 2: slower, calculate thought.
* Modality:
  * Text alone might not be enough, humans rely on linking text to prior knowledge learned vision, touch etc.
* Presents framework for super-intelligent AI
  * Configurator: controls all the other parts, weights goals
  * Perception: embeds sensing
  * World model: predicts future states
  * Cost module: measures discomfort
  * Intrinsic cost: discomfort of current state
  * Trainable cost: predicts future cost
  * Short-term memory: stores relevant information
  * Actor: chooses the actions
    * Actor has system 1 and system 2 modes
* **Energy-based models**:
  * Future state is likely impossible to predict, but we can predict plausible future states.
  * **Understanding this plausibility requires common sense**
  * We can't predict the state directly... but we can **predict a representation** (slight information loss).
  * We therefore aim to be able to predict the future latent representation rather than the state
  * Energy based models: score how compatible the rest of a sequence is. 
  * They take a random variable as input to learn to map this probability dist of all the future things which could happen