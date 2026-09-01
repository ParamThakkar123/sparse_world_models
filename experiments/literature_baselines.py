"""Published object-centric dynamics models on the change-detection task.

Every baseline this project has compared against so far is one we wrote. That is enough to
attribute our model's win to an ingredient; it is not enough to say anything about the
field, and it leaves open the reviewer's reasonable objection that our relational rungs are
weak stand-ins and a properly-built published model would not show the degeneracy we report.

This script runs five published models -- implemented in :mod:`models.literature_baselines`
from their papers rather than adapted from our ladder -- on exactly the same splits, the
same features and a matched parameter budget, alongside the trivial velocity rule and the
no-op reference:

    GNS / DPI-Net    encode-process-decode graph net, 3 message-passing steps, edge features
    C-SWM            contrastive structured world model, energy objective, no decoder
    SlotFormer       temporal transformer over object tokens with a history window
    PETS             ensemble of heteroscedastic Gaussians trained by NLL
    NPS              sparse learned rules, one applied per object per step

Each is trained with its **native objective**, not forced onto ours: C-SWM by the hinge
contrastive energy, PETS by Gaussian NLL on bootstrapped batches, the other three by
next-pose MSE, which is what their papers use for state-space dynamics.

Three questions this answers
----------------------------
**1. Does the ungated degeneracy survive published architectures?** Our finding is that every
model without an explicit gate predicts "everything changed" (recall exactly 1.000), and
that relational structure does not cure it. If GNS with three rounds of message passing and
proper geometric edge features still flags everything, that claim stops being about our
ladder and starts being about ungated regression as a class.

**2. Does sparsity of *mechanism* substitute for a gate?** NPS is the closest published
relative of our claim -- it is also a sparse-mechanism model -- but its sparsity is over
*which rule transforms an object*, not over *whether an object is transformed at all*. If
NPS reproduces our behaviour, our attribution to the change gate is wrong. This is the
sharpest available test of it and the reason NPS is in the set.

**3. Is there a gate-free way to read change out of a standard model?** PETS gives one:
ensemble disagreement. If per-object epistemic variance separates changed from unchanged
objects, then "which objects will move" is recoverable from an ordinary probabilistic
dynamics model, no gate required -- a genuine threat to the attribution. It is reported as
an AUC so it can be read even when the raw scale is not comparable to a gate probability.

**Pre-registered predictions**, recorded before running (project convention):

  a. Every ungated baseline -- GNS, SlotFormer, NPS, PETS, and C-SWM's probe readout --
     produces recall at or very near 1.000 on the motion benchmark, reproducing the
     degeneracy across published architectures.
  b. NPS does **not** reproduce the gate's behaviour, because rule sparsity and change
     sparsity are different axes.
  c. PETS disagreement AUC is above chance (0.5) but well below the learned gate's, because
     the ensemble disagrees most where the delta is *large*, which correlates with change
     but is not the same thing.

Usage::

    python -m experiments.literature_baselines --counts 3 5 8 --seeds 0 1 2 \\
        --split-template "data/transitions/splits_clean_{n}obj_s{seed}/scale_{n}obj_s{seed}_hard_{split}.npz"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from experiments import ExperimentLogger
from experiments.compare_phase4_models import (
    compute_mask_metrics,
    compute_pose_metrics,
    count_parameters,
    evaluate_noop,
    load_dataset,
    predicted_change_mask_from_pose_delta,
)
from experiments.gate_ablation import at_rest_mask, evaluate_object_centric
from experiments.train_sparse_model import build_object_features_by_mode
from models import POSE_DIM, SparseResidualHead, StateLayout, TransitionDataset
from models.literature_baselines import (
    ContrastiveStructuredWorldModel,
    GraphNetworkSimulator,
    NeuralProductionSystem,
    ProbabilisticEnsemble,
    SlotFormerDynamics,
)
from models.relational import match_hidden_dim

BASELINES = ("gns", "cswm", "slotformer", "pets", "nps")

BASELINE_CITATION = {
    "gns": "Sanchez-Gonzalez et al. 2020 / Li et al. 2019",
    "cswm": "Kipf, van der Pol & Welling, ICLR 2020",
    "slotformer": "Wu et al., ICLR 2023",
    "pets": "Chua et al., NeurIPS 2018",
    "nps": "Goyal et al., NeurIPS 2021",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Published baselines on change detection.")
    parser.add_argument("--counts", type=int, nargs="+", default=[3, 5, 8])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--baselines", nargs="+", default=list(BASELINES), choices=list(BASELINES))
    parser.add_argument(
        "--split-template",
        type=str,
        default="data/transitions/splits_clean_{n}obj_s{seed}/scale_{n}obj_s{seed}_hard_{split}.npz",
    )
    parser.add_argument("--feature-mode", type=str, default="global",
                        choices=["global", "invariant", "contact"])
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--num-rules", type=int, default=4)
    parser.add_argument("--message-passing-steps", type=int, default=3)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--attention-blocks", type=int, default=2)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--run-name", type=str, default="literature_baselines")
    parser.add_argument(
        "--capacity-match",
        action="store_true",
        default=True,
        help=(
            "Size every baseline to the sparse model's total parameter count. On by default: "
            "without it a reviewer cannot tell whether a baseline lost on architecture or on "
            "budget, and the published models are mostly larger than ours by default."
        ),
    )
    parser.add_argument("--no-capacity-match", dest="capacity_match", action="store_false")
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=None,
        help=(
            "If set, save each trained baseline to <dir>/<tag>_<baseline>_<N>obj_s<seed>.pt "
            "with the config needed to rebuild it. Required to plan through these models "
            "later -- experiments/planning_mpc.py loads them from here."
        ),
    )
    parser.add_argument("--checkpoint-tag", type=str, default="lit")
    return parser.parse_args()


def save_baseline(
    model: nn.Module, name: str, count: int, seed: int, feature_dim: int, args: argparse.Namespace
) -> None:
    """Persist a trained baseline together with everything needed to reconstruct it."""
    if args.checkpoint_dir is None:
        return
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = args.checkpoint_dir / f"{args.checkpoint_tag}_{name}_{count}obj_s{seed}.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "baseline": name,
                "object_feature_dim": feature_dim,
                "feature_mode": args.feature_mode,
                "num_layers": args.num_layers,
                "ensemble_size": args.ensemble_size,
                "num_rules": args.num_rules,
                "message_passing_steps": args.message_passing_steps,
                "attention_heads": args.attention_heads,
                "attention_blocks": args.attention_blocks,
                "capacity_match": args.capacity_match,
                "num_objects": count,
                "seed": seed,
            },
        },
        path,
    )


def load_baseline(path: Path, device: torch.device) -> tuple[nn.Module, dict]:
    """Rebuild a saved baseline. Mirrors :func:`build_baseline`'s sizing exactly."""
    payload = torch.load(path, map_location=device, weights_only=False)
    config = payload["config"]
    rebuild_args = argparse.Namespace(
        capacity_match=config["capacity_match"], num_layers=config["num_layers"],
        ensemble_size=config["ensemble_size"], num_rules=config["num_rules"],
        message_passing_steps=config["message_passing_steps"],
        attention_heads=config["attention_heads"], attention_blocks=config["attention_blocks"],
    )
    model = build_baseline(config["baseline"], config["object_feature_dim"], rebuild_args, device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, config


def split_path(template: str, count: int, seed: int, split: str) -> Path:
    return Path(template.format(n=count, seed=seed, split=split))


def sparse_param_target(feature_dim: int, device: torch.device) -> int:
    """Total parameters of the canonical sparse head, the budget every baseline is matched to."""
    reference = SparseResidualHead(
        object_feature_dim=feature_dim,
        gate_hidden_dim=128, gate_num_layers=2,
        delta_hidden_dim=128, delta_num_layers=2,
    ).to(device)
    return count_parameters(reference)


def build_baseline(
    name: str, feature_dim: int, args: argparse.Namespace, device: torch.device
) -> nn.Module:
    """Construct one baseline, capacity-matched to the sparse model when requested."""
    target = sparse_param_target(feature_dim, device)

    def sized(factory, candidates) -> int:
        if not args.capacity_match:
            return 64
        width, _ = match_hidden_dim(factory, target, candidates)
        return width

    if name == "gns":
        width = sized(
            lambda hidden_dim: GraphNetworkSimulator(
                feature_dim, hidden_dim, args.message_passing_steps, args.num_layers
            ),
            range(4, 129),
        )
        return GraphNetworkSimulator(
            feature_dim, width, args.message_passing_steps, args.num_layers
        ).to(device)
    if name == "cswm":
        width = sized(
            lambda hidden_dim: ContrastiveStructuredWorldModel(
                feature_dim, hidden_dim=hidden_dim, num_layers=args.num_layers
            ),
            range(4, 257),
        )
        return ContrastiveStructuredWorldModel(
            feature_dim, hidden_dim=width, num_layers=args.num_layers
        ).to(device)
    if name == "slotformer":
        width = sized(
            lambda hidden_dim: SlotFormerDynamics(
                feature_dim, hidden_dim, args.attention_heads, args.attention_blocks
            ),
            range(args.attention_heads, 129, args.attention_heads),
        )
        return SlotFormerDynamics(
            feature_dim, width, args.attention_heads, args.attention_blocks
        ).to(device)
    if name == "pets":
        # The ensemble's budget is its TOTAL across members, so each member is ~1/E the size.
        # Matching per-member instead would hand PETS five times the parameters and make any
        # win uninterpretable.
        width = sized(
            lambda hidden_dim: ProbabilisticEnsemble(
                feature_dim, hidden_dim, args.num_layers, args.ensemble_size
            ),
            range(4, 129),
        )
        return ProbabilisticEnsemble(
            feature_dim, width, args.num_layers, args.ensemble_size
        ).to(device)
    if name == "nps":
        width = sized(
            lambda hidden_dim: NeuralProductionSystem(
                feature_dim, hidden_dim, args.num_rules, args.num_layers
            ),
            range(4, 129),
        )
        return NeuralProductionSystem(
            feature_dim, width, args.num_rules, args.num_layers
        ).to(device)
    raise ValueError(f"Unknown baseline '{name}'.")


def unpack(
    batch: dict[str, torch.Tensor], device: torch.device, feature_mode: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(object_features, current_pose, target_pose)`` from a TransitionDataset batch.

    Object count comes from ``object_change_mask`` rather than being derived from the state
    width, matching ``gate_ablation.run_epoch`` so both experiments read the data the same
    way.
    """
    batch = {key: value.to(device) for key, value in batch.items()}
    num_objects = int(batch["object_change_mask"].shape[1])
    features = build_object_features_by_mode(batch["state"], batch["action"], feature_mode)
    current_pose = batch["current_object_pose"].reshape(-1, num_objects, POSE_DIM)
    target_pose = batch["next_object_pose"].reshape(-1, num_objects, POSE_DIM)
    return features, current_pose, target_pose


def train_baseline(
    name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> nn.Module:
    """Train with the baseline's NATIVE objective; select on its own validation criterion."""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_state, best_metric = None, float("inf")

    def objective(batch: dict[str, torch.Tensor]) -> torch.Tensor:
        features, current_pose, target_pose = unpack(batch, device, args.feature_mode)
        if name == "cswm":
            next_features = build_object_features_by_mode(
                batch["next_state"].to(device), batch["action"].to(device), args.feature_mode
            )
            return model.contrastive_loss(features, batch["action"].to(device), next_features)
        if name == "pets":
            return model.nll(features, target_pose - current_pose)
        return torch.mean((model(features, current_pose) - target_pose) ** 2)

    for _ in range(args.epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = objective(batch)
            loss.backward()
            optimizer.step()

        model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                size = batch["state"].shape[0]
                total += float(objective(batch)) * size
                count += size
        mean = total / max(count, 1)
        if mean < best_metric:
            best_metric = mean
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def cswm_ranking(
    model: ContrastiveStructuredWorldModel,
    dataset: dict[str, np.ndarray],
    next_state_array: np.ndarray,
    device: torch.device,
    feature_mode: str,
    batch_size: int = 128,
) -> dict[str, float]:
    """C-SWM's own evaluation: Hits@1 and MRR against in-batch candidates.

    Reported alongside the pose metrics because the pose numbers come from a probe trained on
    frozen latents, not from C-SWM itself -- scoring it only on pose would misrepresent a
    model that deliberately has no decoder.
    """
    state = torch.from_numpy(dataset["state"]).to(device)
    action = torch.from_numpy(dataset["action"]).to(device)
    # ``load_dataset`` drops ``next_state`` (it keeps only the pose slice), but C-SWM's
    # ranking metric needs the successor's FULL state to encode it, so the caller reads it
    # straight from the npz and passes it in.
    next_state = torch.from_numpy(next_state_array).to(device)
    hits, mrr, batches = 0.0, 0.0, 0
    for start in range(0, state.shape[0], batch_size):
        stop = min(start + batch_size, state.shape[0])
        if stop - start < 2:  # ranking is undefined against a single candidate
            continue
        features = build_object_features_by_mode(state[start:stop], action[start:stop], feature_mode)
        next_features = build_object_features_by_mode(
            next_state[start:stop], action[start:stop], feature_mode
        )
        metrics = model.ranking_metrics(features, action[start:stop], next_features)
        hits += metrics["hits_at_1"]
        mrr += metrics["mrr"]
        batches += 1
    if batches == 0:
        return {"cswm_hits_at_1": float("nan"), "cswm_mrr": float("nan")}
    return {"cswm_hits_at_1": hits / batches, "cswm_mrr": mrr / batches}


def fit_cswm_probe(
    model: ContrastiveStructuredWorldModel,
    train_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    """Fit the pose decoder on FROZEN latents, after contrastive training.

    Only ``model.decoder`` receives gradients, and ``decode`` detaches its input, so this
    cannot turn C-SWM into an autoencoder. It exists so C-SWM can appear on the same pose
    axis as the other baselines, and every number it produces is labelled a probe.
    """
    optimizer = torch.optim.Adam(model.decoder.parameters(), lr=args.lr)
    model.train()
    for _ in range(args.epochs):
        for batch in train_loader:
            features, current_pose, target_pose = unpack(batch, device, args.feature_mode)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                latent = model.encode(features)
                predicted = latent + model.transition(
                    latent, features.new_zeros(features.shape[0], 2)
                )
            loss = torch.mean((current_pose + model.decode(predicted) - target_pose) ** 2)
            loss.backward()
            optimizer.step()


def disagreement_auc(
    model: ProbabilisticEnsemble,
    dataset: dict[str, np.ndarray],
    device: torch.device,
    feature_mode: str,
) -> float:
    """AUC of PETS ensemble disagreement as a change detector -- the gate-free alternative.

    Computed as the Mann-Whitney U statistic (rank-based, so it needs no threshold): the
    probability that a randomly chosen changed object has higher disagreement than a randomly
    chosen unchanged one. 0.5 is chance.
    """
    state = torch.from_numpy(dataset["state"]).to(device)
    action = torch.from_numpy(dataset["action"]).to(device)
    scores: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, state.shape[0], 256):
            stop = min(start + 256, state.shape[0])
            features = build_object_features_by_mode(state[start:stop], action[start:stop], feature_mode)
            scores.append(model.epistemic_disagreement(features).cpu().numpy())
    flat = np.concatenate(scores, axis=0).reshape(-1)
    labels = dataset["target_mask"].reshape(-1).astype(bool)
    if labels.all() or not labels.any():
        return float("nan")
    order = np.argsort(flat, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, flat.size + 1)
    positive, negative = labels.sum(), (~labels).sum()
    return float((ranks[labels].sum() - positive * (positive + 1) / 2.0) / (positive * negative))


def trivial_velocity_row(dataset: dict[str, np.ndarray]) -> dict:
    """The one-line baseline, scored with the same metric code as everything else."""
    rest = at_rest_mask(dataset)
    prediction = (~rest).astype(np.float32)
    # A velocity rule makes no pose prediction, so its pose columns are the no-op's: it says
    # which objects move, not where they go. Reported as NaN rather than silently zero.
    return {
        "pred_mask": prediction,
        "mask_metrics": compute_mask_metrics(prediction, dataset["target_mask"]),
        "onset_f1": (
            compute_mask_metrics(prediction[rest], dataset["target_mask"][rest])["f1"]
            if rest.any() else float("nan")
        ),
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    logger = ExperimentLogger(run_name=args.run_name)
    # Paths are not JSON-serialisable, and log_config runs before any work, so leaving them
    # in kills the run at startup with nothing done.
    logger.log_config({k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()})

    rows: list[dict] = []
    skipped: list[str] = []
    for count in args.counts:
        for seed in args.seeds:
            paths = {
                split: split_path(args.split_template, count, seed, split)
                for split in ("train", "val", "test")
            }
            if not all(path.exists() for path in paths.values()):
                skipped.append(str(paths["train"]))
                continue

            train_dataset = TransitionDataset(str(paths["train"]))
            val_dataset = TransitionDataset(str(paths["val"]))
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
            test = load_dataset(paths["test"])
            # C-SWM ranks the true successor against in-batch candidates, which needs the
            # successor's full state; load_dataset keeps only its pose slice.
            test_next_state = np.load(paths["test"])["next_state"]

            probe_state = torch.from_numpy(test["state"][:1]).to(device)
            probe_action = torch.from_numpy(test["action"][:1]).to(device)
            feature_dim = int(
                build_object_features_by_mode(probe_state, probe_action, args.feature_mode).shape[-1]
            )

            # References, computed once per cell.
            trivial = trivial_velocity_row(test)
            rows.append({
                "object_count": count, "seed": seed, "model": "trivial_velocity",
                "num_parameters": 0,
                "f1": trivial["mask_metrics"]["f1"], "onset_f1": trivial["onset_f1"],
                "precision": trivial["mask_metrics"]["precision"],
                "recall": trivial["mask_metrics"]["recall"],
                "overall_per_object_l2": float("nan"), "changed_object_l2": float("nan"),
                "unchanged_object_l2": float("nan"), "disagreement_auc": float("nan"),
                "cswm_hits_at_1": float("nan"), "cswm_mrr": float("nan"),
            })
            noop = evaluate_noop(test)
            rest = at_rest_mask(test)
            rows.append({
                "object_count": count, "seed": seed, "model": "no_op", "num_parameters": 0,
                "f1": noop["mask_metrics"]["f1"],
                "onset_f1": (
                    compute_mask_metrics(noop["pred_mask"][rest], test["target_mask"][rest])["f1"]
                    if rest.any() else float("nan")
                ),
                "precision": noop["mask_metrics"]["precision"],
                "recall": noop["mask_metrics"]["recall"],
                "overall_per_object_l2": noop["pose_metrics"]["overall_per_object_l2"],
                "changed_object_l2": noop["pose_metrics"]["changed_object_l2"],
                "unchanged_object_l2": noop["pose_metrics"]["unchanged_object_l2"],
                "disagreement_auc": float("nan"),
                "cswm_hits_at_1": float("nan"), "cswm_mrr": float("nan"),
            })

            for name in args.baselines:
                torch.manual_seed(seed)
                model = build_baseline(name, feature_dim, args, device)
                model = train_baseline(name, model, train_loader, val_loader, args, device)
                if name == "cswm":
                    fit_cswm_probe(model, train_loader, args, device)
                model.eval()
                save_baseline(model, name, count, seed, feature_dim, args)

                evaluation = evaluate_object_centric(model, test, device, args.feature_mode)
                onset_f1 = float("nan")
                if rest.any():
                    onset_f1 = compute_mask_metrics(
                        evaluation["pred_mask"][rest], test["target_mask"][rest]
                    )["f1"]
                row = {
                    "object_count": count, "seed": seed, "model": name,
                    "num_parameters": count_parameters(model),
                    "f1": evaluation["mask_metrics"]["f1"], "onset_f1": onset_f1,
                    "precision": evaluation["mask_metrics"]["precision"],
                    "recall": evaluation["mask_metrics"]["recall"],
                    "overall_per_object_l2": evaluation["pose_metrics"]["overall_per_object_l2"],
                    "changed_object_l2": evaluation["pose_metrics"]["changed_object_l2"],
                    "unchanged_object_l2": evaluation["pose_metrics"]["unchanged_object_l2"],
                    "disagreement_auc": float("nan"),
                    "cswm_hits_at_1": float("nan"), "cswm_mrr": float("nan"),
                }
                if name == "pets":
                    row["disagreement_auc"] = disagreement_auc(model, test, device, args.feature_mode)
                if name == "cswm":
                    row.update(
                        cswm_ranking(model, test, test_next_state, device, args.feature_mode)
                    )
                rows.append(row)
                print(
                    f"N={count} s={seed} {name:11s} F1={row['f1']:.4f} onset={row['onset_f1']:.4f} "
                    f"recall={row['recall']:.4f} unchL2={row['unchanged_object_l2']:.4f} "
                    f"params={row['num_parameters']}",
                    flush=True,
                )

    write_outputs(rows, logger.run_dir)
    summary = build_summary(rows)
    summary["skipped"] = skipped
    logger.log_summary(summary)
    print(json.dumps(summary, indent=2))


COLUMNS = [
    "object_count", "seed", "model", "num_parameters", "f1", "onset_f1", "precision", "recall",
    "overall_per_object_l2", "changed_object_l2", "unchanged_object_l2",
    "disagreement_auc", "cswm_hits_at_1", "cswm_mrr",
]


def build_summary(rows: list[dict]) -> dict:
    def mean(model: str, metric: str, count: int | None = None) -> float | None:
        values = [
            row[metric] for row in rows
            if row["model"] == model and (count is None or row["object_count"] == count)
            and row[metric] == row[metric]
        ]
        return float(np.mean(values)) if values else None

    models = sorted({row["model"] for row in rows})
    counts = sorted({row["object_count"] for row in rows})
    per_count = {
        f"{count}obj": {
            model: {metric: mean(model, metric, count)
                    for metric in ("f1", "onset_f1", "recall", "unchanged_object_l2")}
            for model in models
        }
        for count in counts
    }

    # The pre-registered checks from the module docstring.
    ungated = [m for m in ("gns", "slotformer", "nps", "pets", "cswm") if m in models]
    recalls = {model: mean(model, "recall") for model in ungated}
    trivial_f1 = mean("trivial_velocity", "f1")
    return {
        "per_count": per_count,
        "recall_by_ungated_baseline": recalls,
        "every_ungated_baseline_is_degenerate": (
            bool(all(value is not None and value > 0.99 for value in recalls.values()))
            if recalls else None
        ),
        "trivial_rule_f1": trivial_f1,
        "baselines_beaten_by_trivial_rule": [
            model for model in ungated
            if trivial_f1 is not None and (mean(model, "f1") or 0.0) < trivial_f1
        ],
        "pets_disagreement_auc": mean("pets", "disagreement_auc"),
        "cswm_native": {"hits_at_1": mean("cswm", "cswm_hits_at_1"), "mrr": mean("cswm", "cswm_mrr")},
        "citations": BASELINE_CITATION,
    }


def write_outputs(rows: list[dict], output_dir: Path) -> None:
    lines = [",".join(COLUMNS)]
    for row in rows:
        lines.append(",".join(
            f"{row[column]:.6f}" if isinstance(row[column], float) else str(row[column])
            for column in COLUMNS
        ))
    (output_dir / "literature_baselines.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    md = [
        "# Published object-centric dynamics models on change detection",
        "",
        "All models trained with their NATIVE objective on identical splits and features, and",
        "capacity-matched to the sparse model's total parameter count. `recall` is the column",
        "that carries the degeneracy claim: a model that flags every object has recall exactly",
        "1.000. C-SWM's pose columns come from a probe trained on frozen latents (it has no",
        "decoder by design); its native ranking metrics are reported separately.",
        "",
        "| model | citation | F1 | onset F1 | recall | unchanged L2 | params |",
        "|---|---|---|---|---|---|---|",
    ]
    for model in sorted({row["model"] for row in rows}):
        matching = [row for row in rows if row["model"] == model]

        def column(metric: str) -> str:
            values = [row[metric] for row in matching if row[metric] == row[metric]]
            return f"{np.mean(values):.4f}" if values else "--"

        params = int(np.mean([row["num_parameters"] for row in matching]))
        md.append(
            f"| {model} | {BASELINE_CITATION.get(model, '--')} | {column('f1')} | "
            f"{column('onset_f1')} | {column('recall')} | {column('unchanged_object_l2')} | "
            f"{params} |"
        )
    (output_dir / "literature_baselines.md").write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
