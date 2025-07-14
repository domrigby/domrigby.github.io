# INTELLECT-1 Technical Report

Date: 13th July 2025

[arXiv link](https://arxiv.org/html/2412.01152v1)

## Key Point
* Intellect-1 provides a distributed training framework for training across the entire globe.
* It therefore focuses on **communication efficiency** as it has very limited bandwidth
* Method notes:
    * Each worker does multiple weights updates before an all-reduce
    * It records the **changes in the weights** rather than the gradients
    * These are **int8** quantised to save bandwidth
      * Quantisation technique: quantises across $[\mu - 6\sigma, \mu + 6\sigma]$
      * Codebook is established by taking the average of each bin.
      * This is only during transmission. The updates are converted back to fp32 for the actual update
      * Quantisation is done in C++ as it is a computationally heavy operation
    * Weights updates are done on CPU to save GPU update time