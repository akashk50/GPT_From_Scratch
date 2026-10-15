# Transformer Decoder from Scratch

A compact PyTorch implementation of a decoder-style Transformer, written to make the main architecture components easy to inspect.

## Implemented components

- Token embeddings and learned positional embeddings
- Causally masked self-attention with multiple heads
- Feed-forward blocks with ReLU and dropout
- Residual connections and layer normalization
- A final projection to vocabulary logits

The implementation is contained in `gpt.py`. Its `GPT` class accepts a batch of token IDs and returns next-token logits for each position.

## Dependencies

The source imports `torch` and `torchtyping`. A dependency manifest is not included.

## Scope

This repository contains the model architecture. It does not currently include a tokenizer, dataset, training loop, trained weights, or evaluation script.

It is a learning implementation, not a pretrained language model or a reproduction of a specific production GPT model. No training or benchmark results are claimed here. Runtime behavior has not been revalidated as part of this documentation update.

