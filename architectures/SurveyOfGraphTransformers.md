# A Survey of Graph Transformers: Architectures, Theories and Applications

Date read: 6th August 2025

[arXiv link](https://arxiv.org/abs/2502.16533)

## Key Points
* Reason for graph transformers: GNNs struggle with long term dependencies when message passing. Transformers direct connections
overcome this problem.
* Types of Graph Transformer:
  1. Multi-Level Graph Tokenisation: represent nodes, edges, sub-graphs etc. as tokens
  2. Structural Positional Encoding: inject the structure of the graph through positionally encoding the nodes. This can either be by concatenation or summing.
  3. Structure Aware Attention: influences attention mechanism to give indications on structure. Either by design or learned. E.g. mask all accept neighbours.
  4. Model Ensemble of GTs and GNNs: takes turns doing bits. E.g. GNN sub-graph encoder and then GT decoder
* Introduces GNNs:
  * They encode graphs by passing around information about their neighbouring nodes. Both nodes and edges can be the features.
  * Graphs are stores often as (V, E) -> (Vertices, Edges)
  * The find features of the nodes and their edges and then update the nodes.
* Graph transformers are similar: but do all the nodes in one go via self-attention between nodes and/or edges