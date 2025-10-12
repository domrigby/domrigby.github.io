`# Long-Horizon Perception Requires Re-Thinking Recurrence

Date read: 10th October 2025

[Blog link](https://x.com/mike64_t/status/1976397973841117527?s=61)

## Key Points
* Problem with transformers: 
	* As sequence length goes up, number of layers stays the same and therefore the job becomes harder.
	* Longer sequence length means the parameters per token decreases, it **trades expressiveness for parallelism**

* RNNs can be thought of as side-ways deep neural networks, but they are deep in time rather than deep in layers. Their **depth therefore scales with time**
	* E.g. post claims that LSTMs inspired ResNet with how to solve the vanishing gradients problem across layers with how they do it across time.
	* RNNs are like 'infinitely deep neural networks across time'
	* To achieve the same level of compute across sequence length in a transformer would require an infinite number of layers **and** you would need to do it for **every token** as compute per token is constant.

* Suggestion:
	* Author does not suggest processing every token sequentially, but perhaps processing blocks in a sequential manner.
	* Introduces a **'Frame Based Attention Model'**, I would suggest reading teh blog post for an explaination.
