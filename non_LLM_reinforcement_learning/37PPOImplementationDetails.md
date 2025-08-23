# The 37 Implementation Details of Proximal Policy Optimization

Date read: 18th August

[Blog link](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)

## Key Points
* This blog post is often referenced in PPO libraries, e.g. [stable-baselines PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
* Reviews, test and compares different commits to the OpenAI spinning up PPO implementation. The authors then give their 37 top tips on PPO implementation: 13 on general details, 9 ATARI specific, 9 for continuous control, 5 for LSTMs and 1 for multi-discrete action spaces.

### Useful Implementation Details
A lot of the implementation details are stating what the repository does for clarity, rather than useful tips. Here are some of the more useful tips (many are obvious if you are familiar with PPO):
1. Unsurprisingly recommends **vectorised environments**.
   * Vectorised environments are synchronous, actor-learner frameworks.
   * Learn from fixed length rollouts rather than whole episode. You can bootstrap values at the end if the episode has not terminated.
   * Each set of rollouts collects M (rollout length) x N (number of parallel environments) experiences. There is trade off due to memory constraints on M and N
   * Why? Allows you to collect for data per second and therefore explore more of the search space. Wider search should equal more optimal solutions found.

2. **Use GAE**:
  * Bootstrap the values if you the environment is not terminated or truncated
  * Truncated environments are often given done flags and therefore PPO does not estimate their future value, this is not correct. Truncated states should have their future value estimated unless the timing is part of the observation.
  * Use TD(lambda) to estimate returns (lambda = 0 is pure bootstrap and lambda = 1 is pure Monte Carlo)

3. **Mini-Batch Update Tips**:
  * Shuffle the rollout data, but do not randomly sample it. You want to train on all the datapoints

4. **Advantage Normalisation**:
  * Norm at mini-batch level rather than batch level

5. **Clipped surrogate objective**:
  * Explains the PPO clipped surrogate as a stand in for trust regions for PPO
  * Can be done for **value too**, there is however little evidence to suggest this is advantageous

6. **Use an entropy bonus** to encourage exploration
7. **Clip global gradients**
8. **Suggested debugging values**:
  * Policy loss, value loss, entropy loss: shouldn't go up too much and it should identify when one term dominates
  * Clip fraction (what fraction of data is needing to be clipped): shouldn't be too high as this will effect gradient flow
  * KL-divergence estimate: suggests using (ratio - 1) - logratio.mean() as a less bias estimator

9. **Separate value and policy network**:
  * Whilst still feasible (i.e. feature extractor not too heavy) use separate policy and value feature extractors as this massively improves performance.

10. **State dependent and state indepedent standard deviation have similar performance for continuous control**
11. **PPO treats action components independently** (e.g. x and y velocity generated independently)
12. **Tanh squashing actions into allowed range far outperforms clipping**
13. **Moving average normalisation to normalise observations and rewards** (keep track of running mean and standard deviation to normalise the observation and reward space automatically.ss
