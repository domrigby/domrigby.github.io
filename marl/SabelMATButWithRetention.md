# Sable: a Performant, Efficient and Scalable Sequence Model for MARL

Date read: 10th August 2025

[arXiv Link](https://arxiv.org/pdf/2410.01706)

## Key Points
* Attempts to recreate benefits of Multi-Agent Transformers but with retention rather than attention.
* Retention:
  * Operates on **all agents observations in sequential chunks**... this is a trade-off between parallelism and recurrence. It also allows you to process longer data sequences.
  * Encoder-decoder architecture: observations are encoded and then decoded into actions
  * Works like an RNN: h_t+1 = function of h_t
  * Creates Queries, Keys and Values like in normal attention. It queries the hidden state, decays this value and then adds the KV matrix multiplication
* Why retention?
  * Linear in memory rather than quadratic like attention.
  * This allows very large number of agents