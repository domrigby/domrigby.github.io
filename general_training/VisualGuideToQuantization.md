# Visual Guide To Quantization

Date read: 7th September 2025

[Blog link](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization)

## Key Points
* Clearly visualises different types and methods for quantising LLMs (clue in name!)
* Quantisation:
  * Reducing the precision of a model numerical representation tp reduce its memory overhead.
* Why quantise?
  * LLMs required billions of parameters and therefore massive amounts of memory.
  * Storing these numbers as lower precision, smaller number of bits can reduce this overhead.
* Techniques:
  * Linear mapping:
    * Symmetric: scales all values by s and then used a signed integer (range is -max to +max)
    * Asymmetric: scales and then applies bias such that range is min to max (more efficient and precise)
  * Clipping and calibration:
    * Including outliers can massively reduce precision, as they increase range.
    * Methods often set a reasonable range (e.g. +-5std) and then clip the rest of the values
* Types:
  * Post Training Quantisation:
    * Weights are quantised **after** training
    * Activation are dynamically quantised (you don't know their range) during inference or a quantisation rate is set
    before inference on a pre-defined dataset.
  * Quantisation Aware Training
    * Quantises and dequantises during training such that the model can locate the best minima which accounts for its effects.
    * Often lowers FP32 accuracy (no quant) but increases accuracy in low precision models (e.g. int4)