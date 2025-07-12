# How to scale RL to 10^26 FLOPs

[Blog post by Jack Morris](https://blog.jxmo.io/p/how-to-scale-rl-to-1026-flops)

## Key Points
* We are starting to approach the upper-bound of Web-scale data pre-training. How do we go beyond this?
* Reinforcement learning provides a potential path. It allows us the model to create novel reasoning paths on its own.
It currently however has a few issues:
  * Relies on **verifiable rewards**: having a well defined scoring function (e.g. Go in easy to score once the game is over)
  * **Massive compute issues**: current models only do a few hundred of few thousand RL steps
    * Sampling reasoning tokens is expensive
    * Rollouts **require verification** at the end which is expensive. This can mean compiling code, running unit tests for example. This can also mean
    that the bottleneck is on CPU!
* Way forward in his opinion: **RL based pre-training by allowing the model to _think_ during pre-training**. Calls this
**Reasoning with Next Token Prediction (RNTP)**
  * This allows us to get more out of the data. The next token prediction is verifiable, but RL can let the model learn to think beyond that.