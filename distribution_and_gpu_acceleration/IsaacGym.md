# Isaac Gym: High Performance GPU-Based Physics Simulation For Robot Learning

Date read: 16th August 2025

[arXiv Link](https://arxiv.org/abs/2108.10470)

## Key Points
* IsaacGym introduces a set of models and training frameworks which **operates entirely on the GPU**
* There is zero CPU-GPU communications during training. This is opposed to usual RL training which runs the model on the GPU
and the environment on the CPU, introducing significant communication bottlenecks.
* Algorithms such as PPO store there memory buffers **on the GPU**.
* GPU environment also allow **huge parallelism**, potentially running tens of thousands of parallel environments.
* Takeaways:
  * Combined speedups means that training runs which would require clusters can now be done on a desktop and clusters can achieve previously
  unobtainable performance levels.
  * Data creation speed allows for domain randomisation and therefore generalisation and bridging the sim2real gap.
* Implementation details:
  * Written in C++ and CUDA but with Python interface.
  * Built around Nvidia PhyX GPU physics sims
  * States, rewards etc are stored in extremely large tensors. 
  * Initial states can be edited on the CPU and then sent to GPU for processing