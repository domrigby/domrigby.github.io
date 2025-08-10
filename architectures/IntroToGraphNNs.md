# A Gentle Introduction to Graph Neural Networks

Date read: 7th August 2025

[Blog post link](https://distill.pub/2021/gnn-intro/)

## Key Points
* Introduce representing data as graphs, message passing and then graph neural networks.
* Graphs: any data with relationships in it can be classified as a graph (e.g. images are graphs)
* Types of GNN tasks:
  * Node level: making prediction about nodes (e.g. image segmentation)
  * Edge level: inferring information about edges (relationships)
  * Graph level: predictions about graph as a whole (e.g. classification)
* Representing graphs:
  * Simple method is adjacency matrix... but this is not unique (many different adjacency matrices for one graphs).
  * **Adjacency lists are better** (e.g. [1,5], [7,12]). These can be converted into adjacency matrices by gather operations
  and save compute if the adjacency matrix is sparse
* Simplest GNN just does MLP on edges and nodes
* Message passing:
  * Gets information about neighbouring nodes
  * Takes their embeddings and either sums then (if same dim) or concatenates them
  * Pass through NN to create sub-graph embedding (embedding with wider context)
  * Issue: if the neighbours are one away then you can only see N steps away
* Decision: 
  * Must decide which order message passing is done (node-node, edge-node, node-edge, edge-edge)
  * Pooling must be permutation invariant (max, mean, sum all good)
* **Sampling for batches:**
  * Training requires batches of sequences of the same length.
  * This is done by breaking down graphs into a **constant size hypergraph**... i.e. a number of sub-graphs which preserves 
  graph structure.
* **Graph attention:**
  * Performs self-attention on graph nodes and/or edges