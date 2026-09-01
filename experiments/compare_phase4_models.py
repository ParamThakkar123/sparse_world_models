from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from thop import profile

from experiments import ExperimentLogger
from experiments.generate_transitions import POSITION_EPS, YAW_EPS, wrap_angle
from experiments.train_sparse_model import build_object_features_by_mode, resolve_gate_estimator
from models import DenseStatePredictor, POSE_DIM, SparseResidualHead, StateLayout, infer_num_objects_from_state_dim


class SparseProfileWrapper(torch.nn.Module):
    def __init__(self, model: SparseResidualHead, estimator: str, temperature: float):
        super().__init__()
        self.model = model
        self.estimator = estimator
        self.temperature = temperature

    def forward(self, object_features: torch.Tensor) -> torch.Tensor:
        out = self.model(
            object_features,
            estimator=self.estimator,
            temperature=self.temperature,
            hard=True,
        )
        return out.masked_delta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare sparse, dense, and no-op baselines on Phase 4 metrics.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/transitions/splits/scripted_train_250ep_test.npz"),
    )
    parser.add_argument(
        "--dense-checkpoint",
        type=Path,
        default=Path("models/checkpoints/dense_baseline_full_v1.pt"),
    )
    parser.add_argument(
        "--sparse-checkpoint",
        type=Path,
        default=Path("models/checkpoints/phase3_sp020_10ep.pt"),
    )
    parser.add_argument("--run-name", type=str, default="phase4_comparison")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--timing-iters", type=int, default=500)
    parser.add_argument("--warmup-iters", type=int, default=50)
    parser.add_argument("--num-qualitative", type=int, default=2)
    return parser.parse_args()


def load_dataset(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    state = data["s_t"].astype(np.float32)
    num_objects = infer_num_objects_from_state_dim(int(state.shape[1]))
    layout = StateLayout(num_objects=num_objects)
    current_pose = state[:, layout.object_pose_slice].reshape(-1, num_objects, POSE_DIM).astype(np.float32)
    next_pose = data["s_t1"][:, layout.object_pose_slice].reshape(-1, num_objects, POSE_DIM).astype(np.float32)
    return {
        "state": state,
        "action": data["a_t"].astype(np.float32),
        "current_pose": current_pose,
        "next_pose": next_pose,
        "target_mask": data["object_change_mask"].astype(np.float32),
        "target_delta": data["object_delta"].astype(np.float32),
        "num_objects": np.array(num_objects, dtype=np.int32),
    }


def load_dense_model(checkpoint_path: Path, device: torch.device) -> tuple[DenseStatePredictor, dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    model = DenseStatePredictor(
        state_dim=config["state_dim"],
        action_dim=config["action_dim"],
        output_dim=config["target_dim"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config


def load_sparse_model(checkpoint_path: Path, device: torch.device) -> tuple[SparseResidualHead, dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    model = SparseResidualHead(
        object_feature_dim=config["feature_dim"],
        gate_hidden_dim=config["gate_hidden_dim"],
        gate_num_layers=config["gate_num_layers"],
        delta_hidden_dim=config["delta_hidden_dim"],
        delta_num_layers=config["delta_num_layers"],
        # Checkpoints written before the probabilistic heads existed carry no
        # 'delta_head_type', and those are all deterministic -- so defaulting to 'mse'
        # keeps every existing checkpoint loading bit-for-bit as before.
        delta_head_type=str(config.get("delta_head_type", "mse")),
        num_mixture_components=int(config.get("mixture_components", 5)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config


def predicted_change_mask_from_pose_delta(delta: np.ndarray) -> np.ndarray:
    wrapped_delta = delta.copy()
    wrapped_delta[:, :, 2] = wrap_angle(wrapped_delta[:, :, 2])
    position_change = np.linalg.norm(wrapped_delta[:, :, :2], axis=2) > POSITION_EPS
    yaw_change = np.abs(wrapped_delta[:, :, 2]) > YAW_EPS
    return np.logical_or(position_change, yaw_change).astype(np.float32)


def compute_mask_metrics(pred_mask: np.ndarray, target_mask: np.ndarray) -> dict[str, float]:
    pred = pred_mask.astype(bool)
    target = target_mask.astype(bool)
    tp = int(np.logical_and(pred, target).sum())
    fp = int(np.logical_and(pred, ~target).sum())
    fn = int(np.logical_and(~pred, target).sum())
    tn = int(np.logical_and(~pred, ~target).sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "predicted_positive_rate": float(pred.mean()),
        "target_positive_rate": float(target.mean()),
    }


def masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    count = int(mask.sum())
    if count == 0:
        return 0.0
    return float(values[mask].mean())


def compute_pose_metrics(pred_next_pose: np.ndarray, current_pose: np.ndarray, next_pose: np.ndarray, target_mask: np.ndarray) -> dict[str, float]:
    _ = current_pose
    error = pred_next_pose - next_pose
    abs_error = np.abs(error)
    l2_error = np.linalg.norm(error, axis=2)
    changed = target_mask.astype(bool)
    unchanged = ~changed
    repeated_changed = np.repeat(changed[:, :, None], POSE_DIM, axis=2)
    repeated_unchanged = np.repeat(unchanged[:, :, None], POSE_DIM, axis=2)
    return {
        "overall_pose_mse": float(np.mean(error ** 2)),
        "overall_pose_mae": float(np.mean(abs_error)),
        "overall_per_object_l2": float(np.mean(l2_error)),
        "changed_pose_mae": masked_mean(abs_error, repeated_changed),
        "unchanged_pose_mae": masked_mean(abs_error, repeated_unchanged),
        "changed_object_l2": masked_mean(l2_error, changed),
        "unchanged_object_l2": masked_mean(l2_error, unchanged),
    }


def evaluate_dense(model: DenseStatePredictor, dataset: dict[str, np.ndarray], device: torch.device, batch_size: int) -> dict[str, np.ndarray | dict[str, float]]:
    num_objects = int(dataset["num_objects"])
    state = torch.from_numpy(dataset["state"]).to(device)
    action = torch.from_numpy(dataset["action"]).to(device)
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, state.shape[0], batch_size):
            stop = min(start + batch_size, state.shape[0])
            pred = model(state[start:stop], action[start:stop]).cpu().numpy().reshape(-1, num_objects, POSE_DIM)
            predictions.append(pred)
    pred_next_pose = np.concatenate(predictions, axis=0)
    pred_delta = pred_next_pose - dataset["current_pose"]
    pred_mask = predicted_change_mask_from_pose_delta(pred_delta)
    return {
        "pred_next_pose": pred_next_pose,
        "pred_mask": pred_mask,
        "pose_metrics": compute_pose_metrics(pred_next_pose, dataset["current_pose"], dataset["next_pose"], dataset["target_mask"]),
        "mask_metrics": compute_mask_metrics(pred_mask, dataset["target_mask"]),
    }


def evaluate_sparse(model: SparseResidualHead, config: dict[str, object], dataset: dict[str, np.ndarray], device: torch.device, batch_size: int) -> dict[str, np.ndarray | dict[str, float]]:
    num_objects = int(dataset["num_objects"])
    layout = StateLayout(num_objects=num_objects)
    state = torch.from_numpy(dataset["state"]).to(device)
    action = torch.from_numpy(dataset["action"]).to(device)
    pred_next_pose_chunks: list[np.ndarray] = []
    pred_mask_chunks: list[np.ndarray] = []
    estimator = config.get("eval_estimator", resolve_gate_estimator(False, str(config["estimator"])))
    temperature = float(config["temperature"])
    feature_mode = str(config.get("feature_mode", "global"))

    with torch.no_grad():
        for start in range(0, state.shape[0], batch_size):
            stop = min(start + batch_size, state.shape[0])
            object_features = build_object_features_by_mode(state[start:stop], action[start:stop], feature_mode)
            out = model(object_features, estimator=str(estimator), temperature=temperature, hard=True)
            pred_next_pose = state[start:stop, layout.object_pose_slice].reshape(-1, num_objects, POSE_DIM) + out.masked_delta
            pred_next_pose_chunks.append(pred_next_pose.cpu().numpy())
            pred_mask_chunks.append((out.gate.probs >= 0.5).float().cpu().numpy())

    pred_next_pose = np.concatenate(pred_next_pose_chunks, axis=0)
    pred_mask = np.concatenate(pred_mask_chunks, axis=0)
    return {
        "pred_next_pose": pred_next_pose,
        "pred_mask": pred_mask,
        "pose_metrics": compute_pose_metrics(pred_next_pose, dataset["current_pose"], dataset["next_pose"], dataset["target_mask"]),
        "mask_metrics": compute_mask_metrics(pred_mask, dataset["target_mask"]),
    }


def evaluate_noop(dataset: dict[str, np.ndarray]) -> dict[str, np.ndarray | dict[str, float]]:
    pred_next_pose = dataset["current_pose"].copy()
    pred_mask = np.zeros_like(dataset["target_mask"], dtype=np.float32)
    return {
        "pred_next_pose": pred_next_pose,
        "pred_mask": pred_mask,
        "pose_metrics": compute_pose_metrics(pred_next_pose, dataset["current_pose"], dataset["next_pose"], dataset["target_mask"]),
        "mask_metrics": compute_mask_metrics(pred_mask, dataset["target_mask"]),
    }


def count_parameters(model: torch.nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def benchmark_predictor(predict_once, warmup_iters: int, timing_iters: int, device: torch.device) -> float:
    with torch.no_grad():
        for _ in range(warmup_iters):
            predict_once()
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        start = time.perf_counter()
        for _ in range(timing_iters):
            predict_once()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    return elapsed * 1000.0 / timing_iters


def profile_dense(model: DenseStatePredictor, config: dict[str, object], dataset: dict[str, np.ndarray], device: torch.device, warmup_iters: int, timing_iters: int) -> dict[str, float | int]:
    state = torch.from_numpy(dataset["state"][:1]).to(device)
    action = torch.from_numpy(dataset["action"][:1]).to(device)
    with torch.no_grad():
        macs, _ = profile(model, inputs=(state, action), verbose=False)
    latency_ms = benchmark_predictor(lambda: model(state, action), warmup_iters, timing_iters, device)
    return {
        "num_parameters": count_parameters(model),
        "macs_per_forward": float(macs),
        "flops_per_forward_estimate": float(macs * 2.0),
        "avg_inference_latency_ms": latency_ms,
    }


def profile_sparse(model: SparseResidualHead, config: dict[str, object], dataset: dict[str, np.ndarray], device: torch.device, warmup_iters: int, timing_iters: int) -> dict[str, float | int]:
    state = torch.from_numpy(dataset["state"][:1]).to(device)
    action = torch.from_numpy(dataset["action"][:1]).to(device)
    estimator = str(config.get("eval_estimator", resolve_gate_estimator(False, str(config["estimator"]))))
    temperature = float(config["temperature"])
    feature_mode = str(config.get("feature_mode", "global"))
    object_features = build_object_features_by_mode(state, action, feature_mode)
    wrapped_model = SparseProfileWrapper(model, estimator=estimator, temperature=temperature).to(device)
    with torch.no_grad():
        macs, _ = profile(wrapped_model, inputs=(object_features,), verbose=False)

    def predict_once() -> torch.Tensor:
        features = build_object_features_by_mode(state, action, feature_mode)
        out = model(features, estimator=estimator, temperature=temperature, hard=True)
        return out.masked_delta

    latency_ms = benchmark_predictor(predict_once, warmup_iters, timing_iters, device)
    return {
        "num_parameters": count_parameters(model),
        "macs_per_forward": float(macs),
        "flops_per_forward_estimate": float(macs * 2.0),
        "avg_inference_latency_ms": latency_ms,
    }


def profile_noop(dataset: dict[str, np.ndarray], device: torch.device, warmup_iters: int, timing_iters: int) -> dict[str, float | int]:
    state = torch.from_numpy(dataset["state"][:1]).to(device)
    num_objects = int(dataset["num_objects"])
    layout = StateLayout(num_objects=num_objects)

    def predict_once() -> torch.Tensor:
        return state[:, layout.object_pose_slice].reshape(-1, num_objects, POSE_DIM)

    latency_ms = benchmark_predictor(predict_once, warmup_iters, timing_iters, device)
    return {
        "num_parameters": 0,
        "macs_per_forward": 0.0,
        "flops_per_forward_estimate": 0.0,
        "avg_inference_latency_ms": latency_ms,
    }


def flatten_row(model_name: str, pose_metrics: dict[str, float], mask_metrics: dict[str, float], efficiency_metrics: dict[str, float | int]) -> dict[str, float | int | str]:
    return {
        "model": model_name,
        "overall_per_object_l2": pose_metrics["overall_per_object_l2"],
        "changed_object_l2": pose_metrics["changed_object_l2"],
        "unchanged_object_l2": pose_metrics["unchanged_object_l2"],
        "overall_pose_mae": pose_metrics["overall_pose_mae"],
        "precision": mask_metrics["precision"],
        "recall": mask_metrics["recall"],
        "f1": mask_metrics["f1"],
        "accuracy": mask_metrics["accuracy"],
        "predicted_positive_rate": mask_metrics["predicted_positive_rate"],
        "target_positive_rate": mask_metrics["target_positive_rate"],
        "num_parameters": efficiency_metrics["num_parameters"],
        "flops_per_forward_estimate": efficiency_metrics["flops_per_forward_estimate"],
        "avg_inference_latency_ms": efficiency_metrics["avg_inference_latency_ms"],
    }


def write_table(rows: list[dict[str, float | int | str]], path: Path) -> None:
    columns = [
        "model",
        "overall_per_object_l2",
        "changed_object_l2",
        "unchanged_object_l2",
        "precision",
        "recall",
        "f1",
        "num_parameters",
        "flops_per_forward_estimate",
        "avg_inference_latency_ms",
    ]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, separator]
    for row in rows:
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(rows: list[dict[str, float | int | str]], path: Path) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_qualitative_figures(
    output_dir: Path,
    dataset: dict[str, np.ndarray],
    sparse_pred: np.ndarray,
    dense_pred: np.ndarray,
    noop_pred: np.ndarray,
    num_examples: int,
) -> list[str]:
    current_pose = dataset["current_pose"]
    next_pose = dataset["next_pose"]
    max_delta = np.linalg.norm(dataset["target_delta"][:, :, :2], axis=2).max(axis=1)
    example_indices = np.argsort(max_delta)[::-1][:num_examples]
    num_objects = int(current_pose.shape[1])
    colors = [plt.get_cmap("tab10")(idx % 10) for idx in range(num_objects)]
    artifact_paths: list[str] = []

    def draw_panel(ax, title: str, start_pose: np.ndarray, end_pose: np.ndarray) -> None:
        for object_idx, color in enumerate(colors):
            ax.scatter(start_pose[object_idx, 0], start_pose[object_idx, 1], color=color, s=35)
            delta_xy = end_pose[object_idx, :2] - start_pose[object_idx, :2]
            ax.arrow(
                start_pose[object_idx, 0],
                start_pose[object_idx, 1],
                delta_xy[0],
                delta_xy[1],
                color=color,
                width=0.0015,
                head_width=0.01,
                length_includes_head=True,
                alpha=0.85,
            )
        ax.set_title(title)
        ax.set_xlim(-0.35, 0.35)
        ax.set_ylim(-0.35, 0.35)
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)

    for figure_idx, example_idx in enumerate(example_indices):
        fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), constrained_layout=True)
        draw_panel(axes[0], "Current -> Ground Truth", current_pose[example_idx], next_pose[example_idx])
        draw_panel(axes[1], "Current -> Sparse", current_pose[example_idx], sparse_pred[example_idx])
        draw_panel(axes[2], "Current -> Dense", current_pose[example_idx], dense_pred[example_idx])
        draw_panel(axes[3], "Current -> No-op", current_pose[example_idx], noop_pred[example_idx])
        fig.suptitle(f"Transition {int(example_idx)} | max GT xy delta = {max_delta[example_idx]:.4f}")
        output_path = output_dir / f"qualitative_example_{figure_idx}.png"
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
        artifact_paths.append(str(output_path))
    return artifact_paths


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dataset = load_dataset(args.data)

    dense_model, dense_config = load_dense_model(args.dense_checkpoint, device)
    sparse_model, sparse_config = load_sparse_model(args.sparse_checkpoint, device)

    logger = ExperimentLogger(run_name=args.run_name)
    output_dir = logger.run_dir
    logger.log_config(
        {
            "task": "phase4_comparison",
            "data": str(args.data),
            "num_objects": int(dataset["num_objects"]),
            "dense_checkpoint": str(args.dense_checkpoint),
            "sparse_checkpoint": str(args.sparse_checkpoint),
            "device": args.device,
            "batch_size": args.batch_size,
            "warmup_iters": args.warmup_iters,
            "timing_iters": args.timing_iters,
            "num_qualitative": args.num_qualitative,
        }
    )

    dense_eval = evaluate_dense(dense_model, dataset, device, args.batch_size)
    sparse_eval = evaluate_sparse(sparse_model, sparse_config, dataset, device, args.batch_size)
    noop_eval = evaluate_noop(dataset)

    dense_efficiency = profile_dense(dense_model, dense_config, dataset, device, args.warmup_iters, args.timing_iters)
    sparse_efficiency = profile_sparse(sparse_model, sparse_config, dataset, device, args.warmup_iters, args.timing_iters)
    noop_efficiency = profile_noop(dataset, device, args.warmup_iters, args.timing_iters)

    rows = [
        flatten_row("sparse", sparse_eval["pose_metrics"], sparse_eval["mask_metrics"], sparse_efficiency),
        flatten_row("dense", dense_eval["pose_metrics"], dense_eval["mask_metrics"], dense_efficiency),
        flatten_row("no_op", noop_eval["pose_metrics"], noop_eval["mask_metrics"], noop_efficiency),
    ]

    results_csv = output_dir / "results_table.csv"
    results_md = output_dir / "results_table.md"
    write_csv(rows, results_csv)
    write_table(rows, results_md)

    qualitative_paths = save_qualitative_figures(
        output_dir,
        dataset,
        sparse_eval["pred_next_pose"],
        dense_eval["pred_next_pose"],
        noop_eval["pred_next_pose"],
        args.num_qualitative,
    )

    detailed = {
        "sparse": {
            "pose_metrics": sparse_eval["pose_metrics"],
            "change_metrics": sparse_eval["mask_metrics"],
            "efficiency_metrics": sparse_efficiency,
        },
        "dense": {
            "pose_metrics": dense_eval["pose_metrics"],
            "change_metrics": dense_eval["mask_metrics"],
            "efficiency_metrics": dense_efficiency,
        },
        "no_op": {
            "pose_metrics": noop_eval["pose_metrics"],
            "change_metrics": noop_eval["mask_metrics"],
            "efficiency_metrics": noop_efficiency,
        },
    }
    (output_dir / "detailed_results.json").write_text(json.dumps(detailed, indent=2), encoding="utf-8")

    summary = {
        "results_table_csv": str(results_csv),
        "results_table_md": str(results_md),
        "qualitative_figures": qualitative_paths,
        "no_op_overall_l2": noop_eval["pose_metrics"]["overall_per_object_l2"],
        "dense_overall_l2": dense_eval["pose_metrics"]["overall_per_object_l2"],
        "sparse_overall_l2": sparse_eval["pose_metrics"]["overall_per_object_l2"],
        "no_op_trivially_wins": bool(
            noop_eval["pose_metrics"]["overall_per_object_l2"] <= min(
                dense_eval["pose_metrics"]["overall_per_object_l2"],
                sparse_eval["pose_metrics"]["overall_per_object_l2"],
            )
        ),
    }
    logger.log_summary(summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
