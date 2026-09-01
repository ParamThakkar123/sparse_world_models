"""Generate publication figures for IEEE_Conference_Template/conference_101719.tex.

Produces PNGs under ``IEEE_Conference_Template/figures/``:
  * ``qualitative.png`` — predicted vs. ground-truth per-object displacement on example
    scenes (current position -> next position arrows), sparse vs. dense vs. ground truth.
  * ``headline.png``    — overall per-object L2 and change-detection F1 across object
    counts (sparse vs. dense vs. no-op), read from paper_tables/main_results.csv.

Deterministic and self-contained; reuses the trained checkpoints and the held-out test
split. Colours are colourblind-safe and the layout is print-friendly.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.compare_phase4_models import load_dense_model, load_sparse_model
from experiments.train_sparse_model import build_object_features_by_mode, resolve_gate_estimator
from models import POSE_DIM, StateLayout, infer_num_objects_from_state_dim

# Colourblind-safe (Okabe-Ito).
C_GT = "#000000"
C_SPARSE = "#009E73"
C_DENSE = "#D55E00"
C_NOOP = "#0072B2"


def predict_sparse(model, config, state, action, layout, num_objects):
    estimator = str(config.get("eval_estimator", resolve_gate_estimator(False, str(config["estimator"]))))
    temperature = float(config["temperature"])
    feature_mode = str(config.get("feature_mode", "global"))
    feats = build_object_features_by_mode(state, action, feature_mode)
    with torch.no_grad():
        out = model(feats, estimator=estimator, temperature=temperature, hard=True)
    current = state[:, layout.object_pose_slice].reshape(-1, num_objects, POSE_DIM)
    return (current + out.masked_delta).cpu().numpy()


def predict_dense(model, state, action, num_objects):
    with torch.no_grad():
        pred = model(state, action)
    return pred.reshape(-1, num_objects, POSE_DIM).cpu().numpy()


def make_qualitative(args, out_path: Path) -> None:
    raw = np.load(args.data)
    state = torch.from_numpy(raw["s_t"].astype(np.float32))
    action = torch.from_numpy(raw["a_t"].astype(np.float32))
    num_objects = infer_num_objects_from_state_dim(int(state.shape[1]))
    layout = StateLayout(num_objects=num_objects)
    current = state[:, layout.object_pose_slice].reshape(-1, num_objects, POSE_DIM).numpy()
    gt_next = raw["s_t1"][:, layout.object_pose_slice].reshape(-1, num_objects, POSE_DIM).astype(np.float32)
    gt_delta = raw["object_delta"].astype(np.float32)

    sparse_model, sparse_cfg = load_sparse_model(args.sparse_checkpoint, torch.device("cpu"))
    dense_model, _ = load_dense_model(args.dense_checkpoint, torch.device("cpu"))
    sparse_next = predict_sparse(sparse_model, sparse_cfg, state, action, layout, num_objects)
    dense_next = predict_dense(dense_model, state, action, num_objects)

    max_delta = np.linalg.norm(gt_delta[:, :, :2], axis=2).max(axis=1)
    examples = np.argsort(max_delta)[::-1][: args.num_examples]
    colors = [plt.get_cmap("tab10")(i % 10) for i in range(num_objects)]

    def panel(ax, title, start, end, tcolor):
        for i, c in enumerate(colors):
            ax.scatter(start[i, 0], start[i, 1], color=c, s=40, zorder=3, edgecolor="white", linewidth=0.6)
            d = end[i, :2] - start[i, :2]
            ax.arrow(start[i, 0], start[i, 1], d[0], d[1], color=c, width=0.002,
                     head_width=0.012, length_includes_head=True, alpha=0.9, zorder=2)
        ax.set_title(title, color=tcolor, fontsize=11)
        ax.set_xlim(-0.3, 0.3); ax.set_ylim(-0.3, 0.3)
        ax.set_aspect("equal"); ax.grid(alpha=0.2)
        ax.set_xticks([]); ax.set_yticks([])

    n = len(examples)
    fig, axes = plt.subplots(n, 3, figsize=(7.2, 2.5 * n), constrained_layout=True)
    axes = np.atleast_2d(axes)
    for r, idx in enumerate(examples):
        panel(axes[r, 0], "Ground truth", current[idx], gt_next[idx], C_GT)
        panel(axes[r, 1], "Sparse (ours)", current[idx], sparse_next[idx], C_SPARSE)
        panel(axes[r, 2], "Dense", current[idx], dense_next[idx], C_DENSE)
    fig.suptitle("Predicted per-object displacement vs. ground truth", fontsize=12)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"wrote {out_path}")


def make_headline(csv_path: Path, out_path: Path) -> None:
    rows = list(csv.DictReader(csv_path.open()))
    counts = sorted({int(r["num_objects"]) for r in rows})
    models = ["sparse", "dense", "no_op"]
    color = {"sparse": C_SPARSE, "dense": C_DENSE, "no_op": C_NOOP}
    label = {"sparse": "Sparse (ours)", "dense": "Dense", "no_op": "No-op"}

    def get(n, m, key):
        for r in rows:
            if int(r["num_objects"]) == n and r["model"] == m:
                return float(r[key])
        return 0.0

    fig, (ax_l2, ax_f1) = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    x = np.arange(len(counts)); w = 0.26
    for k, m in enumerate(models):
        means = [get(n, m, "overall_per_object_l2_mean") for n in counts]
        stds = [get(n, m, "overall_per_object_l2_std") for n in counts]
        ax_l2.bar(x + (k - 1) * w, means, w, yerr=stds, capsize=3, color=color[m], label=label[m])
    ax_l2.set_xticks(x); ax_l2.set_xticklabels([f"{n} obj" for n in counts])
    ax_l2.set_ylabel("Overall per-object L2 (lower better)")
    ax_l2.set_title("Prediction error"); ax_l2.grid(axis="y", alpha=0.2)
    # Reserve a clear band above the tallest bar+errorbar so the legend never sits on data.
    tallest = max(get(n, m, "overall_per_object_l2_mean") + get(n, m, "overall_per_object_l2_std")
                  for n in counts for m in models)
    ax_l2.set_ylim(0, tallest * 1.42)
    ax_l2.legend(fontsize=8, loc="upper center", ncol=3, framealpha=0.9, borderpad=0.4)

    for k, m in enumerate(["sparse", "dense"]):
        means = [get(n, m, "f1_mean") for n in counts]
        stds = [get(n, m, "f1_std") for n in counts]
        ax_f1.bar(x + (k - 0.5) * w, means, w, yerr=stds, capsize=3, color=color[m], label=label[m])
    ax_f1.set_xticks(x); ax_f1.set_xticklabels([f"{n} obj" for n in counts])
    ax_f1.set_ylabel("Change-detection F1 (higher better)")
    ax_f1.set_ylim(0, 1.18); ax_f1.set_title("Change detection"); ax_f1.grid(axis="y", alpha=0.2)
    ax_f1.set_yticks(np.arange(0, 1.01, 0.2))
    ax_f1.legend(fontsize=8, loc="upper center", ncol=2, framealpha=0.9, borderpad=0.4)

    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"wrote {out_path}")


def _read_csv(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open()))


def make_worldmodel(rollout_csvs: list[Path], transfer_csv: Path, sample_csv: Path, out_path: Path) -> None:
    """Three-panel world-model evidence figure: rollout, transfer, sample efficiency."""
    fig, (ax_r, ax_t, ax_s) = plt.subplots(1, 3, figsize=(10.5, 2.7), constrained_layout=True)

    # (A) Rollout: pose L2 vs horizon at N=3, mean +- std over the training seeds
    # (one CSV per seed; the no-op reference is seed-independent so its band is flat).
    per_seed = [_read_csv(path) for path in rollout_csvs]
    style = {"sparse": (C_SPARSE, "Sparse (ours)"), "dense": (C_DENSE, "Dense"), "no_op": (C_NOOP, "No-op")}
    for m, (c, lab) in style.items():
        curves = []
        for rows in per_seed:
            pts = sorted([(int(r["horizon"]), float(r["pose_l2"])) for r in rows
                          if r["model"] == m and int(r["object_count"]) == 3])
            if pts:
                curves.append(pts)
        if not curves:
            continue
        xs = [h for h, _ in curves[0]]
        ys = np.array([[v for _, v in pts] for pts in curves])
        mean, std = ys.mean(axis=0), ys.std(axis=0)
        ax_r.plot(xs, mean, color=c, label=lab, linewidth=1.8)
        if len(curves) > 1:
            ax_r.fill_between(xs, mean - std, mean + std, color=c, alpha=0.18, linewidth=0)
    ax_r.set_xlabel("Rollout horizon (steps)"); ax_r.set_ylabel("Per-object pose L2")
    ax_r.set_title("(a) Rollout error (3 obj)"); ax_r.grid(alpha=0.2); ax_r.legend(fontsize=8)

    # (B) Transfer heatmap: sparse_invariant F1, train x test.
    rows = _read_csv(transfer_csv)
    counts = sorted({int(r["train_count"]) for r in rows if r["model"] == "sparse_invariant"})
    M = np.full((len(counts), len(counts)), np.nan)
    for r in rows:
        if r["model"] == "sparse_invariant" and r["f1"]:
            M[counts.index(int(r["train_count"])), counts.index(int(r["test_count"]))] = float(r["f1"])
    im = ax_t.imshow(M, cmap="Greens", vmin=0.7, vmax=0.9, aspect="equal")
    ax_t.set_xticks(range(len(counts))); ax_t.set_xticklabels(counts)
    ax_t.set_yticks(range(len(counts))); ax_t.set_yticklabels(counts)
    ax_t.set_xlabel("test object count"); ax_t.set_ylabel("train object count")
    ax_t.set_title("(b) Transfer F1 (sparse)")
    for i in range(len(counts)):
        for j in range(len(counts)):
            ax_t.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                      color="black" if M[i, j] < 0.85 else "white")
    fig.colorbar(im, ax=ax_t, fraction=0.046, pad=0.04)

    # (C) Sample efficiency: F1 vs train samples at N=3.
    rows = _read_csv(sample_csv)
    for m, (c, lab) in {"sparse": (C_SPARSE, "Sparse (ours)"), "dense": (C_DENSE, "Dense")}.items():
        pts = sorted([(int(r["num_train_samples"]), float(r["f1"])) for r in rows
                      if r["model"] == m and int(r["object_count"]) == 3])
        if pts:
            xs, ys = zip(*pts)
            ax_s.plot(xs, ys, color=c, marker="o", markersize=4, label=lab, linewidth=1.8)
    ax_s.set_xlabel("Training transitions"); ax_s.set_ylabel("Change-detection F1")
    ax_s.set_title("(c) Sample efficiency (3 obj)"); ax_s.set_ylim(0, 1)
    ax_s.grid(alpha=0.2); ax_s.legend(fontsize=8)

    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"wrote {out_path}")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description="Generate paper figures.")
    p.add_argument("--data", type=Path, default=root / "data/transitions/splits_3obj_s0/scale_3obj_s0_hard_test.npz")
    p.add_argument("--sparse-checkpoint", type=Path, default=root / "models/checkpoints/sparse_3obj_s0.pt")
    p.add_argument("--dense-checkpoint", type=Path, default=root / "models/checkpoints/dense_3obj_s0.pt")
    p.add_argument("--main-results", type=Path, default=root / "experiments/paper_tables/main_results.csv")
    p.add_argument("--rollout-csv", type=Path, nargs="+",
                   default=[root / f"experiments/runs/rollout_heldout_s{s}/rollout_curves.csv" for s in (0, 1, 2)],
                   help="One rollout_curves.csv per training seed; panel (a) shows mean +- std.")
    p.add_argument("--transfer-csv", type=Path, default=root / "experiments/runs/compositional_generalization/transfer_matrix.csv")
    p.add_argument("--sample-csv", type=Path, default=root / "experiments/runs/sample_efficiency_3obj_s0/sample_efficiency_curves.csv")
    p.add_argument("--out-dir", type=Path, default=root / "IEEE_Conference_Template/figures")
    p.add_argument("--num-examples", type=int, default=2)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    make_qualitative(args, args.out_dir / "qualitative.png")
    make_headline(args.main_results, args.out_dir / "headline.png")
    make_worldmodel(args.rollout_csv, args.transfer_csv, args.sample_csv, args.out_dir / "worldmodel.png")


if __name__ == "__main__":
    main()
