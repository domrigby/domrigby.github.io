# Training extremely large neural networks across thousands of GPUs.

Date: 4th July 2025

[Blog post by Jeremy Jordan](https://www.jeremyjordan.me/distributed-training/)

# Key Points
* Talks of how it is key to train on larger batches which is the key to the rest of the piece.
  * Why? Larger batches reduce noise in the gradient calculations and allow you to get to better minima more quickly.
  * Eventually improvements from increasing the batch size plateaus, but this is normally very high number in modern datasets
  * We need to be able to distribute to help increase batch size or memory will run out.
* Discusses methods to reduce GPU memory usage: gradient accumulation of smaller batches, not storing activations of forward pass but instead
recalculating them and CPU offloading.
* **Types of parallelism:
  1. **Data parallelism**: each GPU has a copy of the model and a different batch of data. They then share gradients to do joint updates.
  2. **Model parallelism**: for large models. Model layers are split over many GPUs.
* **Communication methods**: 
  1. Scatter: send different data to each GPU
  2. Broadcast: same data to all
  3. Reduce: combine all data on one GPU.
* E.g.: all-reduce gradients to one GPU for an update.
* Model parallelism: simple implementation leads to lots of GPU downtime whilst you wait for forward pass to finish and then backward one to begin. 
Methods exist to overcome come this like mini-batching