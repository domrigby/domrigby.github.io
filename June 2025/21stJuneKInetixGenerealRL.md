# KINETIX: INVESTIGATING THE TRAINING OF GENERAL AGENTS THROUGH OPEN-ENDED PHYSICS-BASED CONTROL TASKS

Date: 21st June 2025 (read in a sunny Hyde Park!)

[ArXiv link](https://arxiv.org/pdf/2410.23208)

# Key Points
- Aims to create a generalisable agent from creating vast amounts of online data on a generic task
- Generic task in this case was to a space of shapes, some connected by joints and motors and other with thrusters. They must get the green and blue shapes to touch without touching the red shape.
This is a surprisingly generic definition for a lot of games (e.g. pong, cheetah 2d, robotic hand).
- They generated vast amounts of data using their engine: **Jax2D**. This is a physics simulator which is completed GPU-accelerated. It can be run thousands of times in parallel.
- Levels were random generated... but followed an auto-curricula framework. They used a method called 'SFL' to check levels were solvable and learnable (not too hard, not too easy). This created a natural curriculum of levels.
- A transformer with no positional encodings was used for the model.