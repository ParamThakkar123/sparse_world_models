# Phase 3 Gating Design

## Goal

Predict a binary per-object `changed` gate that decides which objects receive a residual state update.

## Interface

Implemented in:

- `models/sparse_gating.py`

Core module:

- `ObjectChangeGate`

Input:

- object-wise features with shape `(batch, num_objects, feature_dim)`

Output:

- `logits`: raw change scores per object
- `probs`: sigmoid probabilities per object
- `gates`: differentiable binary gate samples per object

## Estimators

Supported forward modes:

- `sigmoid`: fully soft gate, useful for ablations
- `st`: straight-through Bernoulli gate using hard threshold in forward pass
- `gumbel`: relaxed Gumbel-sigmoid sample
- `gumbel_st`: Gumbel-sigmoid with straight-through hard sampling

## Recommended Default

For the first sparse/residual model:

- estimator: `gumbel_st`
- training: start with `temperature = 1.0`, anneal later if needed
- inference: use hard binary gates

Reason:

This gives a discrete object selection signal while preserving gradient flow during training.
It is the most practical default for comparing against the dense baseline without overcomplicating Phase 3.

## Planned Supervision

Use the existing dataset label:

- `object_change_mask`

Primary loss for the gate head should be binary classification per object.
The residual predictor can then be trained conditionally on the predicted or teacher-forced gate.
