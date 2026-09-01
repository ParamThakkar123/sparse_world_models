# Phase 3 Sparse/Residual Design

## Delta Head

Implemented in:

- `models/sparse_residual.py`

Core module:

- `ObjectDeltaHead`

Input:

- object-wise features with shape `(batch, num_objects, feature_dim)`

Output:

- per-object planar delta `(dx, dy, dtheta)` with shape `(batch, num_objects, 3)`

Design choice:

- regression output is always produced for every object
- supervision is applied only on objects labeled `changed`

Reason:

This avoids forcing the delta head to explain persistent objects while keeping the architecture simple.
The gate is responsible for deciding which objects should receive an update.

## Combined Sparse/Residual Head

Core module:

- `SparseResidualHead`

Forward output:

- gate logits/probabilities/samples
- raw per-object delta prediction
- gate-masked delta prediction

## Total Loss

Implemented as:

- `sparse_residual_loss(...)`

Loss form:

- `BCE(gate_logits, changed_mask) + L2(delta | changed objects only) + sparsity_weight * mean(gate_probs)`

Concretely:

- gate term: binary cross-entropy on `object_change_mask`
- delta term: masked L2 regression using only changed objects
- sparsity penalty: mean gate activation probability

## Recommended Default Weights

Initial settings:

- `gate_loss_weight = 1.0`
- `delta_loss_weight = 1.0`
- `sparsity_weight = 1e-2`

These should be treated as starting values, not frozen hyperparameters.

## Training Note

During early experiments, it is reasonable to:

- train the gate head first with teacher supervision
- then train the joint sparse/residual model

That staged setup reduces instability and makes debugging easier than training everything jointly from the start.
