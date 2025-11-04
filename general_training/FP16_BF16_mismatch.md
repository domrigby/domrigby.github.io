# Defeating the Training-Inference Mismatch via FP16

Date read: 1st November 2025

[Paper link]()

## Key Points
* In RL, the training (optimised for calculating gradients) and the inference are often separate.
* There is often a slight mismatch between the inference and training policy which causes isntability in training.
* This is exagerated if one is in FP16 and the other in BF16:
	* FP16: Mantissa of 10 and exponent of 5. High precision but low range. Susceptible to over and underflow (values appear as max or as 0)
	* BF16: Mantissa of 7 and exponent of 8. Low precision but full FP32 range. Common for training neural networks.
* BF16 causes the instabilites, as even if both inference and training are in BF16, the low precision will cause instabilities.
* Paper **finds using FP16 for RL* fixes these issues. During fine-tuning, overflow should not be an issue as dynamic range is ound during training.