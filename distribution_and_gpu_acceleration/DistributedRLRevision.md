# Distributed RL Revision

**Date:** 30th October - 1st November 2026

Related: [IMPALA](IMPALA_DistributedRL.md) | [TorchBeast](TorchBeastDistributedPyTorch.md) | [Distributed PPO](DistributedPPO.md) | [DD-PPO](DD-PPO.md)

## Key Points
* Run through different parts of distributed RL lifecycle and cover each different parts and design decisions

## Breakdown Into Steps
* Env: sim / tool calls <-> Inference / rollouts -> Reward model / trajectory queue -> Distributed learners -> Weight sync

## Where Do We Run the RL?
* **Colocated**:
    * All 8 GPUs in nodes **switch between acting, learning and reward**
    * Switching parts must be synchronised
    * Beneficial if time saved by making each parts time saved is greater than the cost of switching

* **Separate**:
    * Certain GPUs are used for different role as actor or a learner. Means **we can make acting and learning simultaneous and asynchronous**
    * **Must be balanced**: different parts must take approximately same amount of time. 
    * Resources sit if idle if poorly balanced. E.g. if learners take 2x as long as actors, the actors will either sit idle of produce too much experience.

* **Hybrid**:
    * Some GPUs have role, others share a jobs. E.g. if acting takes 4x as long as loading its weights takes a long time, it may be pinned, whilst learners and reward model share.

## When Do We Run?
* **Sync**: everything runs in order, in lockstep. Simple and makes on-policyness easier, but can bottleneck.
* **Async**: actor and learner operate in parallel. Can lead to higher utilisation, but have to correct for off-policyness and balance system.

### Policy Staleness
* Must distinguish between:
    1. Behaviour policy: used to collect data
    2. Current policy: the one being optimised (likely been updated since behaviour was recorded)
    3. Reference policy: frozen version of the policy used for KL-divergence etc.

* Source of mismatch:
    * Temporal: weight updates occur inbetween
    * Numerical: different quantisations -> different logits

* Typical correction:
    * **Importance Sampling**: exp(dif of log probs of current and behaviour policy)
        * Used in PPO, IMPALA etc.
        * Choice whether to do at trajectory level or single obs level: variance vs bias

## Memory Constraints
* Network: weights, optimiser state, activations for all models
* KV-cache if using transformers
* Environment state if running on GPU
* Observations and trajectories, especially if visions

## Common Bottlenecks and their Symptoms
1. **Policy throughput**: learner is waiting for actor to create experiences. Increase rollouts capacity to fix, use lighter policy, quantise etc.
2. **Environmnet**: same symptom as above but profile inside actor. Environment is taking longer than policy forward pass. Fix by increasing capacity for env / increase efficiency, look to GPU based environments etc.
3. **Learner**: queue to learner expands, workers get throttled by rate limiters. Increase learn capacity.
4. **Comms / weights transfer**: long time spent on comms related activities. Switch from collocation to separate roles?
5. **OOMs**: use lighter models, tensor / pipeline parallelism.
6. **Stragglers**: long tail of actor time taken dsitribution is blocking other workers. You could switch to dynamic batching or stop waiting for all N batches to finish. Use a cut-off to stop crazy long stragglers, but be very careful otherwise you will bias towards short and easy answers.

## Maxmising Efficiency When Using CPU Based Environment
* Some environment have to be run on CPU (e.g. ones with OS calls)
* This means comms between GPU and CPU and dealing with long-tail CPU calls.
* Tools:
    * **Generate other trajectories whilst waiting for tool calls**
    * Continuous, asynchronous, per-trajectory execution. Basically don't wait for whole batch to finish on CPU before calling GPU.
    * **Queues for CPU and GPU: GPU queue batches returned tool calls and sends to GPU for next steps**
    * Architecture:
        * Agent-loops: maintain trajectory states, request tool calls.
        * Inference servers: batches batches requests for GPUs.
        * Tool workers: do tool calls upon request
    