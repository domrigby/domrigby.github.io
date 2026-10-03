# RL Token: Bootstrapping Online RL with Vision-Language-Action Models

**Date:** 28th September 2026

[arXiv Link](https://arxiv.org/abs/2604.23073) | [PDF](https://arxiv.org/pdf/2604.23073)

Related: [π0.5 VLA](Pi0.5VLA.md)

## Key Points
* Motivation:
    * VLAs contain rich contextual information, generalisation and reasoning capabilities. They can produce action sequences which meet goals from natural language instructions.
    * However, they struggling with out-of-distribution teleoperation and struggle with precise tasks (e.g. dexterity).
    * It is difficult to get them to learn online as training the VLAs is compute intensive and sample inefficient
    * Aim of this paper is to propose a method which takes advantage
* Bit of background:
    * VLAs utilise action-chunking: create groups of actions rather than one-by-one.
    * Utilise expressive output distributions, e.g. diffusio 
    * Other methods tend to either fine-tune action head only or train a complimentary network, which produces a residual for the outputted action to help with precision.
    * Online RL methods too data inefficient

## Key Methods
1. **Train a VLA to output a rich RL-token**.
    * This is created by training an small transformer autoencoder on the VLAs embeddings on task relevant data.
    * The aim is to produce a vector which contains all the context the VLA has absorbed
    * **Tuned VLA is then frozen**
2. **Train small network actor-critic with RL Token and outputted action as observation**:
    * RL-token contains compact reasoning info.
    * Outputted action means that we can tune the already generated action, rather than having to learn action dynamics from scratch
        * Action is randomly masked out so stopped the policy just regurgitating the action.
    * Operates over action chunks
    * Policy and critic operate over **smaller action chunks**, to be more reactive
    * Trained with **TD3**
    * Light weight MLP actor and critic (256x256 MLPs)

