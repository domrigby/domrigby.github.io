# How Visual Representations Map to Language Feature Space in Multimodal LLMs
[Arxiv link](https://arxiv.org/abs/2506.11976)

Date: 24th June 2025

## Key Points
- Aims to identify where the latent representations of words and images unify in VLMs
- They freeze a vision transformer encoder and freeze an LLM and then train a linear adapter to project between them
- Sparse Auto-Encoders (SAEs), a method for representing activation layers to humans, are used to analyse features
- Found that representations only unify in the mid-to-late layers
- Experiment was done using images and captions to identify when they shared the same representations