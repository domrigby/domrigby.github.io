# How Far Can Off-Policy RL Reach With Stale Data on LLMs

Date read: 14th October 2025

[Blog link](https://m2po.notion.site/rl-stale-m2po)

## Key Points
* Aims:
	* Off-policy RL is super important as it makes scaling across data centres easy.
	* Off-policy RL -> can be asynchronous -> distributed rollouts can be collected independently on stale policies with training occurs elsewhere in the data centre.
	* On-policy RL requires a lot more syncs
	* Off-policy RL does however suffer from instabilities due to the distribution shift between data collection and the current policy being trained

* Introduces stale@k: data policy is being trained on was collected k training steps ago

* Observations:
	* For vanilla GRPO; staleness degrades performance
	* Clipping fraction massively increases due to staleness of data
	* Clipping introduces optimisation bias

* They perform an **experiment to remove the trust region from GRPO and train on both non-stale and stale data**
	* Results: stale data managed to keep up with on-policy data until it collapses due to high variance
	* This means off-policy data is as informative as on-policy data, but its variance makes it too unstable.
	* This is called **prosperity before collapse**

* Reason for this behaviour:
	* r-ratio is proportionate to entropy, meaning that high entropy tokens tend to masked out more often.
	* Issue is that high entropy tokens tend to be the most informative (they are forks in the road of reasoning)
	* Staleness makes this problem worse as tokens generated have even high r-ratios meaning that larger and larger numbers get clipped
	* Hence, the high performance before collapse (informative but too unstable)
	* There is a **trade-off between effectiveness and stability**

* How to fix:
	* They suggest **Second Moment Policy Optimisatio (M2PO)**
	* **Measures distribution mismatch across whole batch, rather than per token**
	* If only a few tokens display high variance, none will be masked out
	* Measures this by taking the second moment of log probabilities across the whole batch, and then only masking once this passes a threshold




