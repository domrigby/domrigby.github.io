# Introducing Instant Robot Learning

**Date:** 2nd November 2026

[Blog Post](https://machenta.com/blog/instant-robot-learning)

## Key Points
* Aims: 
    * Build robotic foundation model which **learns tasks from one demonstration**
    * Train this model entirely in simulation
    * Address issue that post-training VLAs to a new task is expensive and doesn't generalise well (don't respond well to new edge cases).

* Solution:
    * **Vision Demonstration Action model (VDA)**
        * VDAs are conditioned on demonstrations rather than text
        * Demonstrations provide a far richer representation of a task than text
        * Model has to **learn-to-learn quickly** rather than memorise tasks
        * Model has to learn abstract representation of task... will learn the similarities between many tasks

* Benefits of Instant Learning
    * Teach robot on job without expensive post-training
    * **sim2real is built in**: can observe robot parameters in the demonstration.