"""Gate ablation -- separating *object-centric structure* from *modeling change*.

The headline comparison is sparse (per-object gate + residual) vs a monolithic dense
MLP, and ``param_matched_baseline`` removes the "sparse is just smaller" confound. One
confound remains, and it is the one a reviewer will reach for first: the sparse model is
both **object-centric** (per-object features, weights shared across objects) *and*
**change-modeling** (a discrete gate over a residual). The dense monolith is neither, so
the headline gap cannot say which ingredient earns the win.

This experiment interpolates the two designs with a capacity-matched ladder, so each rung
adds exactly one ingredient:

  1. ``dense``       -- monolithic MLP on raw state -> all absolute poses.  (neither)
  2. ``oc_absolute`` -- per-object shared MLP on the *same* features the sparse model
                        sees -> that object's absolute next pose.           (object-centric)
  3. ``oc_residual`` -- same, but predicts a pose *delta* always applied.   (+ residual)
  4. ``sparse``      -- delta applied only where a learned gate fires.      (+ change gate)
  5. ``no_op``       -- copy the current pose.                             (reference)

Reading the ladder: 1->2 is the value of object-centric featurization and weight sharing,
2->3 the value of the residual parameterization, and **3->4 the value of the change gate
itself** -- the paper's actual claim, isolated.

Rungs 2 and 3 are trained here; ``sparse``/``dense`` reuse the canonical checkpoints. The
two new models are width-matched to the sparse model's *total* parameter count (gate +
delta head) so no rung wins on capacity -- ``--width-mode identical`` instead gives them a
delta head architecturally identical to the sparse model's, leaving sparse ahead on
parameters by roughly the size of its gate. Metric definitions are imported from
``compare_phase4_models`` so every number is computed exactly as in the headline table.

**W2 extension.** ``--extended`` adds four rungs that answer the two objections the
original ladder cannot: ``gnn`` and ``set_transformer`` (permutation-equivariant models
that *can* see other objects, so "ungated" is no longer confounded with "no interaction
modelling"), and ``dense_l1`` and ``soft_gate`` (sparsity by L1 penalty and by a continuous
sigmoid, so "gated" is no longer confounded with "discrete"). See ``EXTENDED_RUNGS`` below.

**Splits.** The default ``--split-template`` points at the original directories, which leak
(25% of source episodes have chunks in both train and test -- see
``experiments/build_clean_splits.py``). Any *new* comparison must use the clean splits AND
the clean baseline checkpoints, or the pre-trained sparse/dense rungs are scored on data
they partly memorised while the newly-trained rungs are not::

    python -m experiments.gate_ablation --counts 3 5 8 --seed 0 --extended \\
      --split-template "data/transitions/splits_clean_{n}obj_s{seed}/scale_{n}obj_s{seed}_hard_{split}.npz" \\
      --sparse-template "models/checkpoints/sparse_clean_{n}obj_s{seed}.pt" \\
      --dense-template  "models/checkpoints/dense_clean_{n}obj_s{seed}.pt" \\
      --run-name gate_ablation_extended_clean
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from experiments import ExperimentLogger
from experiments.compare_phase4_models import (
    compute_mask_metrics,
    compute_pose_metrics,
    count_parameters,
    evaluate_dense,
    evaluate_noop,
    evaluate_sparse,
    load_dataset,
    load_dense_model,
    load_sparse_model,
    predicted_change_mask_from_pose_delta,
)
from experiments.train_sparse_model import build_object_features_by_mode
from models import DenseStatePredictor, POSE_DIM, SparseResidualHead, StateLayout, TransitionDataset
from models.relational import (
    InteractionNetworkPredictor,
    SetTransformerPredictor,
    match_hidden_dim,
)
from models.sparse_residual import ObjectDeltaHead

# Rungs are ordered from "neither ingredient" to "both", and drawn in this order.
MODEL_ORDER = (
    "dense", "dense_l1", "oc_absolute", "oc_residual",
    "gnn", "set_transformer", "soft_gate", "sparse", "trivial_velocity", "no_op",
)
MODEL_STYLE = {
    "dense": {"color": "#d95f02", "label": "dense monolith"},
    "dense_l1": {"color": "#b3541e", "label": "dense + L1 on deltas"},
    "oc_absolute": {"color": "#e7a15c", "label": "object-centric, absolute"},
    "oc_residual": {"color": "#8da0cb", "label": "object-centric, residual"},
    "gnn": {"color": "#5573b0", "label": "interaction network"},
    "set_transformer": {"color": "#3b5081", "label": "set transformer"},
    "soft_gate": {"color": "#66c2a5", "label": "soft gate (sigmoid, no ST)"},
    "sparse": {"color": "#1b9e77", "label": "sparse (hard gate + residual)"},
    "trivial_velocity": {"color": "#999999", "label": "trivial: already moving"},
    "no_op": {"color": "#7570b3", "label": "no-op"},
}
TRAINED_HERE = ("oc_absolute", "oc_residual", "gnn", "set_transformer", "dense_l1", "soft_gate")

# The ladder as originally published, kept as the default so reruns of the existing
# experiment reproduce byte-identical rung sets.
BASE_RUNGS = ("dense", "oc_absolute", "oc_residual", "sparse", "no_op")

# An object is "at rest" below this planar speed; onset metrics are computed on those
# rows only. See experiments/momentum_shortcut.py.
REST_SPEED = 2.55e-05

# W2 additions. Two families, each aimed at a distinct reviewer objection:
#
#   * ``gnn`` / ``set_transformer`` answer "your ungated rungs process objects
#     independently, so you have confounded 'ungated' with 'no interaction modelling' --
#     a real relational model would not need a gate". Both are permutation-equivariant,
#     share weights across objects, and predict an always-applied residual, so the only
#     ingredient they add over ``oc_residual`` is inter-object communication.
#   * ``dense_l1`` / ``soft_gate`` answer "why a discrete gate at all -- just penalise the
#     deltas". ``dense_l1`` is L1-regularised sparsity with no gate anywhere; ``soft_gate``
#     keeps the sparse architecture but multiplies by a continuous sigmoid rather than a
#     Gumbel straight-through sample, so it isolates *discreteness* from *gating*.
#
# The claim these are meant to test is the sharp one: hard, discrete, per-object gating
# beats both equivariant-but-ungated and soft-sparse alternatives.
EXTENDED_RUNGS = MODEL_ORDER


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gate ablation: object-centric vs change-modeling.")
    parser.add_argument("--counts", type=int, nargs="+", default=[3, 5, 8])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--rungs",
        type=str,
        nargs="+",
        default=list(BASE_RUNGS),
        choices=list(MODEL_ORDER),
        help="Which ladder rungs to run. Defaults to the originally published five.",
    )
    parser.add_argument(
        "--extended",
        action="store_true",
        help="Shorthand for --rungs with every rung including the W2 baselines.",
    )
    parser.add_argument(
        "--split-template",
        type=str,
        default="data/transitions/splits_{n}obj_s{seed}/scale_{n}obj_s{seed}_hard_{split}.npz",
        help=(
            "Where to read splits. Point this at the episode-disjoint clean splits "
            "(data/transitions/splits_clean_{n}obj_s{seed}/scale_{n}obj_s{seed}_hard_{split}.npz) "
            "for any new comparison -- see experiments/build_clean_splits.py."
        ),
    )
    parser.add_argument("--l1-weight", type=float, default=1e-3,
                        help="Weight on the L1 delta penalty for the dense_l1 rung.")
    parser.add_argument("--sparsity-weight", type=float, default=0.2,
                        help="Sparsity penalty for the soft_gate rung; matches the canonical sparse runs.")
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--attention-blocks", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=15, help="Matches the sparse prediction runs.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-layers", type=int, default=2, help="Depth of the per-object MLP.")
    parser.add_argument(
        "--width-mode",
        type=str,
        default="matched",
        choices=["matched", "identical"],
        help=(
            "'matched' sizes the ablation MLP to the sparse model's *total* parameter count; "
            "'identical' uses the sparse delta head's exact width (128), leaving sparse larger."
        ),
    )
    parser.add_argument("--hidden-dim", type=int, default=128, help="Width used by --width-mode identical.")
    parser.add_argument("--feature-mode", type=str, default="global", choices=["global", "invariant", "contact"])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--run-name", type=str, default="gate_ablation")
    parser.add_argument("--sparse-template", type=str, default="models/checkpoints/sparse_{n}obj_s{seed}.pt")
    parser.add_argument("--dense-template", type=str, default="models/checkpoints/dense_{n}obj_s{seed}.pt")
    return parser.parse_args()


def split_path(count: int, seed: int, split: str, template: str | None = None) -> Path:
    if template is None:
        template = "data/transitions/splits_{n}obj_s{seed}/scale_{n}obj_s{seed}_hard_{split}.npz"
    return Path(template.format(n=count, seed=seed, split=split))


class ObjectCentricPredictor(nn.Module):
    """Per-object shared MLP with no change gate -- rungs 2 and 3 of the ladder.

    Consumes the same per-object features as the sparse model and shares weights across
    objects (so it is object-centric and count-agnostic exactly as the sparse model is),
    but predicts *every* object at *every* step. ``mode='absolute'`` regresses the next
    pose directly; ``mode='residual'`` regresses a delta that is always applied, which is
    the sparse model with its gate pinned open.
    """

    def __init__(self, object_feature_dim: int, hidden_dim: int, num_layers: int, mode: str):
        super().__init__()
        if mode not in {"absolute", "residual"}:
            raise ValueError(f"mode must be 'absolute' or 'residual', got {mode!r}.")
        self.mode = mode
        self.mlp = ObjectDeltaHead(
            object_feature_dim=object_feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=POSE_DIM,
        )

    def forward(self, object_features: torch.Tensor, current_pose: torch.Tensor) -> torch.Tensor:
        """Returns the predicted next pose, shape ``(batch, num_objects, POSE_DIM)``."""
        out = self.mlp(object_features)
        return out if self.mode == "absolute" else current_pose + out


def build_equivariant_rung(
    rung: str, feature_dim: int, hidden_dim: int, args: argparse.Namespace
) -> nn.Module:
    """Construct one of the rungs sharing the ``(features, current_pose) -> next_pose`` API."""
    if rung == "oc_absolute":
        return ObjectCentricPredictor(feature_dim, hidden_dim, args.num_layers, mode="absolute")
    if rung == "oc_residual":
        return ObjectCentricPredictor(feature_dim, hidden_dim, args.num_layers, mode="residual")
    if rung == "gnn":
        return InteractionNetworkPredictor(
            object_feature_dim=feature_dim, hidden_dim=hidden_dim,
            message_dim=hidden_dim, num_layers=args.num_layers, mode="residual",
        )
    if rung == "set_transformer":
        return SetTransformerPredictor(
            object_feature_dim=feature_dim, hidden_dim=hidden_dim,
            num_heads=args.attention_heads, num_blocks=args.attention_blocks, mode="residual",
        )
    raise ValueError(f"'{rung}' is not an equivariant rung.")


def resolve_rung_width(
    rung: str, args: argparse.Namespace, feature_dim: int, device: torch.device
) -> tuple[int, int]:
    """Width for a rung, and the sparse parameter target it is matched against.

    Under ``--width-mode matched`` every rung is sized to the sparse model's *total*
    parameter count, so the relational baselines cannot be dismissed as underpowered --
    nor credited for extra capacity.
    """
    target = sparse_param_count(feature_dim, device)
    if args.width_mode == "identical" or rung in {"oc_absolute", "oc_residual"}:
        return resolve_hidden_dim(args, feature_dim, device)

    if rung == "gnn":
        width, _ = match_hidden_dim(
            lambda hidden_dim: InteractionNetworkPredictor(
                object_feature_dim=feature_dim, hidden_dim=hidden_dim,
                message_dim=hidden_dim, num_layers=args.num_layers, mode="residual",
            ),
            target, range(4, 129),
        )
        return width, target
    if rung == "set_transformer":
        # Widths must stay divisible by the head count; match_hidden_dim skips the rest.
        width, _ = match_hidden_dim(
            lambda hidden_dim: SetTransformerPredictor(
                object_feature_dim=feature_dim, hidden_dim=hidden_dim,
                num_heads=args.attention_heads, num_blocks=args.attention_blocks, mode="residual",
            ),
            target, range(args.attention_heads, 129, args.attention_heads),
        )
        return width, target
    raise ValueError(f"No width rule for rung '{rung}'.")


def sparse_param_count(feature_dim: int, device: torch.device) -> int:
    """Total parameters of the canonical sparse head (gate + delta) at this feature dim."""
    reference = SparseResidualHead(
        object_feature_dim=feature_dim,
        gate_hidden_dim=128,
        gate_num_layers=2,
        delta_hidden_dim=128,
        delta_num_layers=2,
    ).to(device)
    return count_parameters(reference)


def resolve_hidden_dim(args: argparse.Namespace, feature_dim: int, device: torch.device) -> tuple[int, int]:
    """Pick the ablation MLP width, returning ``(hidden_dim, sparse_param_target)``.

    Under ``--width-mode matched`` we scan widths for the one whose parameter count lands
    closest to the sparse model's total, so the ladder is capacity-controlled end to end.
    """
    target = sparse_param_count(feature_dim, device)
    if args.width_mode == "identical":
        return args.hidden_dim, target

    best_width, best_gap = args.hidden_dim, None
    for width in range(8, 1025):
        probe = ObjectDeltaHead(
            object_feature_dim=feature_dim,
            hidden_dim=width,
            num_layers=args.num_layers,
            output_dim=POSE_DIM,
        )
        gap = abs(count_parameters(probe) - target)
        if best_gap is None or gap < best_gap:
            best_width, best_gap = width, gap
    return best_width, target


def run_epoch(model, loader, device, optimizer, feature_mode: str, train: bool) -> float:
    """One pass; returns mean next-pose MSE over *all* objects (the ungated objective)."""
    model.train(train)
    total_loss, total_count = 0.0, 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            num_objects = int(batch["object_change_mask"].shape[1])
            current_pose = batch["current_object_pose"].reshape(-1, num_objects, POSE_DIM)
            target_pose = batch["next_object_pose"].reshape(-1, num_objects, POSE_DIM)
            features = build_object_features_by_mode(batch["state"], batch["action"], feature_mode)
            predicted = model(features, current_pose)
            loss = nn.functional.mse_loss(predicted, target_pose)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            batch_size = batch["state"].shape[0]
            total_loss += float(loss.item()) * batch_size
            total_count += batch_size
    return total_loss / max(total_count, 1)


def run_dense_epoch(model, loader, device, optimizer, l1_weight: float, train: bool) -> float:
    """One pass for the ``dense_l1`` rung: pose MSE plus an L1 penalty on the implied delta.

    This is the "why not just regularise the deltas?" alternative to a gate. The penalty is
    applied to ``predicted_next_pose - current_pose`` -- the displacement the monolith
    implies -- which is the only place sparsity can be expressed without an explicit gate.
    """
    model.train(train)
    total_loss, total_count = 0.0, 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            num_objects = int(batch["object_change_mask"].shape[1])
            current_pose = batch["current_object_pose"].reshape(-1, num_objects, POSE_DIM)
            target_pose = batch["next_object_pose"].reshape(-1, num_objects, POSE_DIM)
            predicted = model(batch["state"], batch["action"]).reshape(-1, num_objects, POSE_DIM)
            mse = nn.functional.mse_loss(predicted, target_pose)
            l1 = (predicted - current_pose).abs().mean()
            loss = mse + l1_weight * l1
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            batch_size = batch["state"].shape[0]
            # Report the plain MSE so model selection is not biased by the penalty weight.
            total_loss += float(mse.item()) * batch_size
            total_count += batch_size
    return total_loss / max(total_count, 1)


def train_dense_l1(
    args: argparse.Namespace, count: int, device: torch.device, checkpoint_dir: Path
) -> tuple[DenseStatePredictor, dict]:
    torch.manual_seed(args.seed)
    train_dataset = TransitionDataset(split_path(count, args.seed, "train", args.split_template))
    val_dataset = TransitionDataset(split_path(count, args.seed, "val", args.split_template))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    sample = train_dataset[0]
    state_dim = int(sample["state"].shape[0])
    action_dim = int(sample["action"].shape[0])
    target_dim = train_dataset.num_objects * POSE_DIM
    # Matched to the canonical dense baseline so the ONLY difference is the L1 term.
    model = DenseStatePredictor(
        state_dim=state_dim, action_dim=action_dim, output_dim=target_dim,
        hidden_dim=256, num_layers=3, dropout=0.0,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val, best_state, best_epoch = float("inf"), None, -1
    for epoch in range(args.epochs):
        run_dense_epoch(model, train_loader, device, optimizer, args.l1_weight, train=True)
        val_loss = run_dense_epoch(model, val_loader, device, optimizer, args.l1_weight, train=False)
        if val_loss < best_val:
            best_val, best_epoch = val_loss, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError(f"dense_l1 training produced no validation metrics at {count} objects.")
    model.load_state_dict(best_state)
    model.eval()

    config = {
        "mode": "dense_l1", "l1_weight": args.l1_weight,
        "state_dim": state_dim, "action_dim": action_dim, "target_dim": target_dim,
        "hidden_dim": 256, "num_layers": 3, "dropout": 0.0,
        "num_parameters": count_parameters(model),
        "best_val_mse": best_val, "best_epoch": best_epoch, "epochs": args.epochs,
        "num_train_samples": len(train_dataset),
    }
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "config": config},
               checkpoint_dir / f"dense_l1_{count}obj_s{args.seed}.pt")
    return model, config


def train_soft_gate(
    args: argparse.Namespace, count: int, device: torch.device, checkpoint_dir: Path
) -> tuple[SparseResidualHead, dict]:
    """Train the sparse architecture with a continuous sigmoid gate (no Gumbel, no ST).

    Shells out to ``train_sparse_model`` so this rung and the canonical sparse checkpoint
    come from the same training code with the same hyperparameters -- the only difference
    is ``--estimator sigmoid``, which isolates *discreteness* from *gating*.
    """
    name = f"soft_gate_{count}obj_s{args.seed}"
    checkpoint = Path("models/checkpoints") / f"{name}.pt"
    command = [
        sys.executable, "-m", "experiments.train_sparse_model",
        "--train", str(split_path(count, args.seed, "train", args.split_template)),
        "--val", str(split_path(count, args.seed, "val", args.split_template)),
        "--run-name", name,
        "--estimator", "sigmoid",
        "--feature-mode", args.feature_mode,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--sparsity-weight", str(args.sparsity_weight),
        "--auto-balance-bce",
        "--seed", str(args.seed),
        "--device", args.device,
        "--checkpoint-dir", str(checkpoint.parent),
    ]
    print(f"[gate_ablation] training {name} (sigmoid gate)", flush=True)
    subprocess.run(command, check=True)
    model, config = load_sparse_model(checkpoint, device)
    config = dict(config)
    config["mode"] = "soft_gate"
    config["num_parameters"] = count_parameters(model)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "config": config},
               checkpoint_dir / f"{name}.pt")
    return model, config


def train_variant(
    args: argparse.Namespace,
    count: int,
    rung: str,
    device: torch.device,
    checkpoint_dir: Path,
) -> tuple[nn.Module, dict]:
    torch.manual_seed(args.seed)
    train_dataset = TransitionDataset(split_path(count, args.seed, "train", args.split_template))
    val_dataset = TransitionDataset(split_path(count, args.seed, "val", args.split_template))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    sample = train_dataset[0]
    feature_dim = build_object_features_by_mode(
        sample["state"].unsqueeze(0), sample["action"].unsqueeze(0), args.feature_mode
    ).shape[-1]
    hidden_dim, sparse_target = resolve_rung_width(rung, args, feature_dim, device)

    model = build_equivariant_rung(rung, feature_dim, hidden_dim, args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val, best_state, best_epoch = float("inf"), None, -1
    for epoch in range(args.epochs):
        run_epoch(model, train_loader, device, optimizer, args.feature_mode, train=True)
        val_loss = run_epoch(model, val_loader, device, optimizer, args.feature_mode, train=False)
        if val_loss < best_val:
            best_val, best_epoch = val_loss, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError(f"Training produced no validation metrics for {rung} at {count} objects.")
    model.load_state_dict(best_state)
    model.eval()

    config = {
        "mode": rung,
        "feature_mode": args.feature_mode,
        "feature_dim": feature_dim,
        "hidden_dim": hidden_dim,
        "num_layers": args.num_layers,
        "num_parameters": count_parameters(model),
        "sparse_param_target": sparse_target,
        "param_ratio_vs_sparse": count_parameters(model) / max(sparse_target, 1),
        "best_val_mse": best_val,
        "best_epoch": best_epoch,
        "epochs": args.epochs,
        "num_train_samples": len(train_dataset),
    }
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state_dict": model.state_dict(), "config": config},
        checkpoint_dir / f"{rung}_{count}obj_s{args.seed}.pt",
    )
    return model, config


def evaluate_object_centric(
    model: nn.Module,
    dataset: dict[str, np.ndarray],
    device: torch.device,
    feature_mode: str,
    batch_size: int = 256,
) -> dict:
    """Score an ungated per-object model with the headline metric definitions.

    The predicted change mask is thresholded from the predicted displacement exactly as
    for the dense baseline, so change-detection numbers are comparable across the ladder.
    """
    num_objects = int(dataset["num_objects"])
    layout = StateLayout(num_objects=num_objects)
    state = torch.from_numpy(dataset["state"]).to(device)
    action = torch.from_numpy(dataset["action"]).to(device)

    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, state.shape[0], batch_size):
            stop = min(start + batch_size, state.shape[0])
            features = build_object_features_by_mode(state[start:stop], action[start:stop], feature_mode)
            current_pose = state[start:stop, layout.object_pose_slice].reshape(-1, num_objects, POSE_DIM)
            chunks.append(model(features, current_pose).cpu().numpy())

    pred_next_pose = np.concatenate(chunks, axis=0)
    pred_mask = predicted_change_mask_from_pose_delta(pred_next_pose - dataset["current_pose"])
    return {
        "pred_next_pose": pred_next_pose,
        "pred_mask": pred_mask,
        "pose_metrics": compute_pose_metrics(
            pred_next_pose, dataset["current_pose"], dataset["next_pose"], dataset["target_mask"]
        ),
        "mask_metrics": compute_mask_metrics(pred_mask, dataset["target_mask"]),
    }


def at_rest_mask(dataset: dict[str, np.ndarray]) -> np.ndarray:
    """Objects whose planar speed is below REST_SPEED, i.e. not already in motion."""
    num_objects = int(dataset["num_objects"])
    layout = StateLayout(num_objects=num_objects)
    velocity = dataset["state"][:, layout.object_velocity_slice].reshape(-1, num_objects, 6)
    return np.linalg.norm(velocity[:, :, 3:5], axis=2) <= REST_SPEED


def collect_row(count: int, model_name: str, params: int, evaluation: dict,
                dataset: dict[str, np.ndarray] | None = None) -> dict:
    onset_f1 = float("nan")
    if dataset is not None:
        # Onset F1 is the metric that actually measures prediction: restricted to objects at
        # rest, where the momentum cue that dominates the headline F1 carries no signal.
        rest = at_rest_mask(dataset)
        if rest.any():
            onset_f1 = compute_mask_metrics(
                evaluation["pred_mask"][rest], dataset["target_mask"][rest]
            )["f1"]
    return {
        "object_count": count,
        "model": model_name,
        "num_parameters": params,
        "f1": evaluation["mask_metrics"]["f1"],
        "onset_f1": onset_f1,
        "precision": evaluation["mask_metrics"]["precision"],
        "recall": evaluation["mask_metrics"]["recall"],
        "overall_per_object_l2": evaluation["pose_metrics"]["overall_per_object_l2"],
        "changed_object_l2": evaluation["pose_metrics"]["changed_object_l2"],
        "unchanged_object_l2": evaluation["pose_metrics"]["unchanged_object_l2"],
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    counts = list(args.counts)
    rungs = list(EXTENDED_RUNGS) if args.extended else list(args.rungs)

    logger = ExperimentLogger(run_name=args.run_name)
    output_dir = logger.run_dir
    checkpoint_dir = output_dir / "checkpoints"
    logger.log_config(
        {
            "task": "gate_ablation",
            "counts": counts,
            "seed": args.seed,
            "epochs": args.epochs,
            "width_mode": args.width_mode,
            "num_layers": args.num_layers,
            "feature_mode": args.feature_mode,
            "device": args.device,
            "split_template": args.split_template,
            "l1_weight": args.l1_weight,
            "ladder": rungs,
        }
    )

    rows: list[dict] = []
    variant_configs: dict[str, dict] = {}
    equivariant_rungs = ("oc_absolute", "oc_residual", "gnn", "set_transformer")
    for count in counts:
        dataset = load_dataset(split_path(count, args.seed, "test", args.split_template))

        for rung in equivariant_rungs:
            if rung not in rungs:
                continue
            print(f"[gate_ablation] training {rung} on {count} objects", flush=True)
            model, config = train_variant(args, count, rung, device, checkpoint_dir)
            variant_configs[f"{rung}_{count}obj"] = config
            evaluation = evaluate_object_centric(model, dataset, device, args.feature_mode)
            rows.append(collect_row(count, rung, config["num_parameters"], evaluation, dataset))

        if "dense_l1" in rungs:
            print(f"[gate_ablation] training dense_l1 on {count} objects", flush=True)
            model, config = train_dense_l1(args, count, device, checkpoint_dir)
            variant_configs[f"dense_l1_{count}obj"] = config
            rows.append(collect_row(count, "dense_l1", config["num_parameters"],
                                    evaluate_dense(model, dataset, device, 256), dataset))

        if "soft_gate" in rungs:
            model, config = train_soft_gate(args, count, device, checkpoint_dir)
            variant_configs[f"soft_gate_{count}obj"] = config
            # evaluate_sparse multiplies by out.gate.gates, which for the sigmoid estimator
            # is the continuous probability -- so this scores the genuinely soft model, not
            # a thresholded stand-in. Detection metrics still threshold at 0.5.
            rows.append(collect_row(count, "soft_gate", config["num_parameters"],
                                    evaluate_sparse(model, config, dataset, device, 256), dataset))

        if "sparse" in rungs:
            sparse_model, sparse_cfg = load_sparse_model(
                Path(args.sparse_template.format(n=count, seed=args.seed)), device
            )
            rows.append(collect_row(count, "sparse", count_parameters(sparse_model),
                                    evaluate_sparse(sparse_model, sparse_cfg, dataset, device, 256), dataset))
        if "dense" in rungs:
            dense_model, _ = load_dense_model(
                Path(args.dense_template.format(n=count, seed=args.seed)), device
            )
            rows.append(collect_row(count, "dense", count_parameters(dense_model),
                                    evaluate_dense(dense_model, dataset, device, 256), dataset))
        if "no_op" in rungs:
            rows.append(collect_row(count, "no_op", 0, evaluate_noop(dataset), dataset))
        if "trivial_velocity" in rungs:
            # The baseline that beats every learned model on the headline metric.
            rest = at_rest_mask(dataset)
            trivial = (~rest).astype(np.float32)
            evaluation = {
                "pred_mask": trivial,
                "mask_metrics": compute_mask_metrics(trivial, dataset["target_mask"]),
                "pose_metrics": evaluate_noop(dataset)["pose_metrics"],
            }
            rows.append(collect_row(count, "trivial_velocity", 0, evaluation, dataset))

    rows.sort(key=lambda row: (row["object_count"], MODEL_ORDER.index(row["model"])))
    write_csv(rows, output_dir / "gate_ablation_results.csv")
    write_markdown(rows, output_dir / "gate_ablation.md")
    present = [rung for rung in MODEL_ORDER if any(row["model"] == rung for row in rows)]
    figure_path = plot_results(counts, rows, output_dir / "gate_ablation.png", present)

    # The claim under test: the change gate, not object-centricity alone, is what earns the
    # win -- so sparse must beat the ungated object-centric rungs, not merely the monolith.
    gate_earns_win = {}
    for count in counts:
        by_model = {row["model"]: row for row in rows if row["object_count"] == count}

        def beats(winner: str, loser: str, metric: str, lower_is_better: bool = True):
            """Comparison guarded against absent rungs, so a subset run still summarises."""
            if winner not in by_model or loser not in by_model:
                return None
            if lower_is_better:
                return bool(by_model[winner][metric] < by_model[loser][metric])
            return bool(by_model[winner][metric] > by_model[loser][metric])

        entry = {
            "sparse_beats_oc_residual_l2": beats("sparse", "oc_residual", "overall_per_object_l2"),
            "sparse_beats_oc_residual_f1": beats("sparse", "oc_residual", "f1", lower_is_better=False),
            "sparse_beats_oc_absolute_l2": beats("sparse", "oc_absolute", "overall_per_object_l2"),
            "oc_residual_beats_dense_l2": beats("oc_residual", "dense", "overall_per_object_l2"),
            # W2: the sharper claims. A hard discrete gate should beat relational models
            # that have no gate, and beat soft-sparsity alternatives that have no discreteness.
            "sparse_beats_gnn_l2": beats("sparse", "gnn", "overall_per_object_l2"),
            "sparse_beats_gnn_f1": beats("sparse", "gnn", "f1", lower_is_better=False),
            "sparse_beats_set_transformer_l2": beats("sparse", "set_transformer", "overall_per_object_l2"),
            "sparse_beats_set_transformer_f1": beats("sparse", "set_transformer", "f1", lower_is_better=False),
            "sparse_beats_dense_l1_l2": beats("sparse", "dense_l1", "overall_per_object_l2"),
            "sparse_beats_soft_gate_l2": beats("sparse", "soft_gate", "overall_per_object_l2"),
            "sparse_beats_soft_gate_f1": beats("sparse", "soft_gate", "f1", lower_is_better=False),
        }
        if "sparse" in by_model and "oc_residual" in by_model:
            entry["l2_ratio_oc_residual_over_sparse"] = (
                by_model["oc_residual"]["overall_per_object_l2"]
                / max(by_model["sparse"]["overall_per_object_l2"], 1e-9)
            )
            entry["f1_gap_sparse_minus_oc_residual"] = (
                by_model["sparse"]["f1"] - by_model["oc_residual"]["f1"]
            )
        gate_earns_win[str(count)] = entry

    def holds_everywhere(*keys: str) -> bool | None:
        values = [gate_earns_win[str(c)].get(k) for c in counts for k in keys]
        if any(v is None for v in values):
            return None
        return bool(all(values))

    summary = {
        "counts": counts,
        "seed": args.seed,
        "width_mode": args.width_mode,
        "rungs": rungs,
        "split_template": args.split_template,
        "results_csv": str(output_dir / "gate_ablation_results.csv"),
        "results_md": str(output_dir / "gate_ablation.md"),
        "figure": figure_path,
        "variant_configs": variant_configs,
        "gate_earns_win": gate_earns_win,
        "sparse_beats_ungated_object_centric_everywhere": holds_everywhere(
            "sparse_beats_oc_residual_l2", "sparse_beats_oc_residual_f1"
        ),
        "sparse_beats_relational_everywhere": holds_everywhere(
            "sparse_beats_gnn_l2", "sparse_beats_gnn_f1",
            "sparse_beats_set_transformer_l2", "sparse_beats_set_transformer_f1",
        ),
        "hard_gate_beats_soft_sparsity_everywhere": holds_everywhere(
            "sparse_beats_dense_l1_l2", "sparse_beats_soft_gate_l2", "sparse_beats_soft_gate_f1"
        ),
    }
    logger.log_summary(summary)
    print(json.dumps(summary, indent=2))


def write_csv(rows: list[dict], path: Path) -> None:
    columns = [
        "object_count", "model", "num_parameters", "f1", "onset_f1", "precision", "recall",
        "overall_per_object_l2", "changed_object_l2", "unchanged_object_l2",
    ]
    lines = [",".join(columns)]
    for row in rows:
        lines.append(
            ",".join(
                f"{row[c]:.6f}" if isinstance(row[c], float) else str(row[c]) for c in columns
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_markdown(rows: list[dict], path: Path) -> None:
    lines = [
        "# Gate ablation: object-centric structure vs modeling change",
        "",
        "Capacity-matched ladder, seed 0, held-out test split. Each rung adds one ingredient:",
        "`dense` (neither) -> `oc_absolute` (object-centric) -> `oc_residual` (+ residual)",
        "-> `sparse` (+ change gate). The `oc_*` rungs share the sparse model's features and",
        "weight sharing but predict every object every step.",
        "",
        "| N | model | params | F1 | **onset F1** | precision | recall | overall L2 | changed L2 | unchanged L2 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['object_count']} | {row['model']} | {row['num_parameters']} | "
            f"{row['f1']:.3f} | {row['onset_f1']:.3f} | {row['precision']:.3f} | {row['recall']:.3f} | "
            f"{row['overall_per_object_l2']:.4f} | {row['changed_object_l2']:.4f} | "
            f"{row['unchanged_object_l2']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_results(counts: list[int], rows: list[dict], path: Path, present: list[str]) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    width = 0.8 / max(len(present), 1)
    x = np.arange(len(counts))
    centre = (len(present) - 1) / 2.0
    panels = (
        (axes[0], "overall_per_object_l2", "Overall per-object $L_2$ (lower better)"),
        (axes[1], "f1", "Change-detection F1 (higher better)"),
    )
    for ax, metric, title in panels:
        for offset, model_name in enumerate(present):
            values = [
                next(r[metric] for r in rows if r["object_count"] == c and r["model"] == model_name)
                for c in counts
            ]
            ax.bar(x + (offset - centre) * width, values, width,
                   label=MODEL_STYLE[model_name]["label"], color=MODEL_STYLE[model_name]["color"])
        ax.set_xticks(x, [f"{c} obj" for c in counts])
        ax.set_title(title)
        ax.grid(alpha=0.2, axis="y")
    axes[0].legend(fontsize=8)
    fig.suptitle("Is the win object-centricity, the residual, or the change gate?")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return str(path)


if __name__ == "__main__":
    main()
