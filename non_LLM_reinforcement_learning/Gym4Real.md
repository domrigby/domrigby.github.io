# Gym4ReaL: A Suite for Benchmarking Real-World Reinforcement Learning

Date read: 30th July 2025

[arXiv link](https://arxiv.org/abs/2507.00257)

## Key Points
* Set of 'realistic' environments across varied domains for developing RL algorithms for real world applications
(not massively realistic)
* Realism: environments unstationary (based on real life, past data) and partially observable
* Environments: tends to focus on scheduling problems.
  1. **DamEnv**: control waterflow out of a reservoir to avoid over/underfilling but meeting water demands
  2. **ElevatorEnv**: controls elevator (up, down, stop) to minimise time in which people have to wait to get to ground
  floor.
  3. **MicrogridEnv**: manage a battery (digital twin of real battery to match performance) to maximise profit from
  buying and selling electricity.
  4. **RoboFeeder: picking and planning**:
      * Picking: agent select where the robot arm will grasp by selecting (x, y) coordinate in an image.
      * Planning: agent selects order of items to pick first by selcting an image from a set
  5. **Trading env**: forex environment. Gets a market state and then decides whether to short, long or flat. 
  6. **WaterDistributionSystemEnv**: control a set of pumps which can fill water tanks, reservoirs etc. Agent must decide
    which valves to open or close (seems to have a very large action space of 2^P - 1...)