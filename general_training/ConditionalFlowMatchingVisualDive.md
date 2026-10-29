# A Visual Dive into Conditional Flow Matching

**Date:** 29th October 2026

[Blog Post](https://iclr-blogposts.github.io/2025/blog/conditional-flow-matching/)

Related: [π0.5: a Vision-Language-Action Model with Open-World Generalization](../robotics/Pi0.5VLA.md)

## Generative Modelling
* Three bottlenecks of generative modelling:
    * Sample speed
    * Quality
    * Fully cover underlying distribution p
* Typical strategy: sample from simple base distribution and learn to map it to complex one. Sample x from p, and then learned function
* Learn to maximimise the likelihood that p and T are close in sense of KL divergence

## Conditional Flow Matching
* Goal: **find velocity field such that p flows through it and ends up as p_desired** (check article for intuitive graphics)
* We want to get a supervision signal for every intermediate between t = [0, 1], however there are infinite ways to flow p to p_desired. Therefore, we must learn to specify full vector field
* This is very hard as p_desired is unknown

* Core idea of CFM is to choose a conditioning variable z and a conditional probability p(x|t,z) such that global p(x|t) trasnform p_0 into p_desired and that the assosciate veclotiy field has analytical form. 