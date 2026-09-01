"""Demo 2 -- what the gate believes, as a function of where the pusher is.

``RESULTS.md`` diagnoses the Round-1 planning failure in prose: the ``global``
features key change prediction on object *velocity* and the scripted policy's
contact context, so when CEM queries teleported, near-rest states from arbitrary
approach angles "the gate never fires and the delta head returns a near-constant
drift *regardless of the pusher's position*". Round 2 replaces those features
with contact geometry and retrains on mixed-policy data, after which "the gate
now fires (prob 0.7-0.94) when the pusher is positioned to push, and the
predicted target motion points *toward* the goal when the pusher is behind the
object and *away* once it overshoots".

This renders both claims directly. Hold the scene fixed, sweep the pusher over a
grid of positions, and at each one ask the model two questions:

  * does the change gate fire for the target object?  (background heat map)
  * which way does it think the target will move?     (arrows, coloured green
    when the predicted motion points toward the goal, red when it points away)

A model that has learned contact geometry produces a bright crescent behind the
object with arrows fanning toward the goal. A model that has not produces a flat
field -- which is precisely a planner with no gradient to follow.

Honest note, stated in the figure itself: the sweep sets object velocities to
zero, because that is the state a sampling planner actually visits. That is
out-of-distribution for the velocity-keyed ``global`` model *by construction* --
which is the finding, not a handicap invented for the demo. Pass
``--keep-velocity`` to sweep with the dataset's true velocities instead.

Example
-------
python -m experiments.demos.demo_gate_field \
    --round1-checkpoint models/checkpoints/sparse_3obj_s0.pt \
    --round2-checkpoint models/checkpoints/sparse_contact_3obj_v1.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize

from experiments.compare_phase4_models import load_sparse_model
from experiments.demos.render2d import (
    GOAL_COLOR,
    GOAL_RADIUS,
    box_corners,
    object_color,
)
from experiments.train_sparse_model import build_object_features_by_mode, resolve_gate_estimator
from models import POSE_DIM, StateLayout, infer_num_objects_from_state_dim
from matplotlib.patches import Circle, Polygon

PUSHER_BOUND = 0.26


def sweep_gate_field(
    model,
    config: dict,
    base_state: np.ndarray,
    layout: StateLayout,
    num_objects: int,
    target_object: int,
    resolution: int,
    zero_velocity: bool,
    device: torch.device,
) -> dict:
    """Gate probability and predicted delta for the target object over a pusher grid.

    At every grid point the action is the unit step from the pusher *toward* the
    target object -- i.e. "if the planner tried to push from here, what would the
    model predict?". Feeding a fixed or zero action instead would confound the
    gate's pusher-position sensitivity with its action sensitivity.
    """
    axis = np.linspace(-PUSHER_BOUND, PUSHER_BOUND, resolution)
    grid_x, grid_y = np.meshgrid(axis, axis)
    pusher_grid = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1).astype(np.float32)

    states = np.tile(base_state.astype(np.float32), (pusher_grid.shape[0], 1))
    states[:, 0:2] = pusher_grid
    if zero_velocity:
        states[:, layout.object_velocity_slice] = 0.0

    object_pose = states[:, layout.object_pose_slice].reshape(-1, num_objects, POSE_DIM)
    to_object = object_pose[:, target_object, :2] - pusher_grid
    actions = to_object / np.maximum(np.linalg.norm(to_object, axis=1, keepdims=True), 1e-6)

    state_t = torch.from_numpy(states).to(device)
    action_t = torch.from_numpy(actions.astype(np.float32)).to(device)
    estimator = str(config.get("eval_estimator", resolve_gate_estimator(False, str(config["estimator"]))))
    feature_mode = str(config.get("feature_mode", "global"))

    with torch.no_grad():
        features = build_object_features_by_mode(state_t, action_t, feature_mode)
        out = model(features, estimator=estimator, temperature=float(config["temperature"]), hard=True)

    shape = (resolution, resolution)
    return {
        "axis": axis,
        "gate_prob": out.gate.probs[:, target_object].cpu().numpy().reshape(shape),
        "delta_xy": out.delta[:, target_object, :2].cpu().numpy().reshape(*shape, 2),
        "feature_mode": feature_mode,
    }


def goal_alignment(delta_xy: np.ndarray, object_xy: np.ndarray, goal_xy: np.ndarray) -> np.ndarray:
    """Cosine between predicted motion and the direction that would help (-1..1)."""
    to_goal = goal_xy - object_xy
    to_goal = to_goal / max(float(np.linalg.norm(to_goal)), 1e-6)
    norm = np.maximum(np.linalg.norm(delta_xy, axis=-1, keepdims=True), 1e-9)
    return (delta_xy / norm) @ to_goal


def draw_panel(
    ax,
    field: dict,
    object_pose: np.ndarray,
    goal_xy: np.ndarray,
    target_object: int,
    title: str,
    subtitle: str,
    quiver_stride: int,
):
    axis = field["axis"]
    mesh = ax.pcolormesh(
        axis, axis, field["gate_prob"], cmap="magma", vmin=0.0, vmax=1.0, shading="auto", zorder=0
    )

    stride = quiver_stride
    sub_x, sub_y = np.meshgrid(axis[::stride], axis[::stride])
    delta = field["delta_xy"][::stride, ::stride, :]
    alignment = goal_alignment(delta, object_pose[target_object, :2], goal_xy)
    magnitude = np.maximum(np.linalg.norm(delta, axis=-1, keepdims=True), 1e-9)
    unit = delta / magnitude
    ax.quiver(
        sub_x,
        sub_y,
        unit[..., 0],
        unit[..., 1],
        alignment,
        cmap="RdYlGn",
        norm=Normalize(-1.0, 1.0),
        scale=26,
        width=0.006,
        alpha=0.95,
        zorder=2,
    )

    ax.add_patch(
        Circle(
            (float(goal_xy[0]), float(goal_xy[1])),
            GOAL_RADIUS,
            facecolor=GOAL_COLOR,
            alpha=0.30,
            edgecolor=GOAL_COLOR,
            linewidth=2.6,
            linestyle="--",
            zorder=3,
        )
    )
    for idx in range(object_pose.shape[0]):
        ax.add_patch(
            Polygon(
                box_corners(object_pose[idx]),
                closed=True,
                facecolor=object_color(idx),
                edgecolor="white" if idx != target_object else "#ffffff",
                linewidth=2.4 if idx == target_object else 1.2,
                zorder=4,
            )
        )

    ax.set_xlim(-PUSHER_BOUND, PUSHER_BOUND)
    ax.set_ylim(-PUSHER_BOUND, PUSHER_BOUND)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=11.5, fontweight="bold", pad=8)
    ax.text(
        0.5, -0.045, subtitle, transform=ax.transAxes, ha="center", va="top",
        fontsize=9, color="#3a3a3a",
    )
    return mesh


def direction_circular_std_deg(delta_xy: np.ndarray) -> float:
    """Circular standard deviation of predicted motion direction across the grid.

    This is the number that tests the Round-1 diagnosis literally: "a near-constant
    drift *regardless of the pusher's position*" means the predicted direction
    barely varies as the pusher moves, i.e. a small circular std. A model that has
    learned contact geometry must point the object away from wherever the pusher
    is, so its direction field varies a lot.
    """
    flat = delta_xy.reshape(-1, 2)
    unit = flat / np.maximum(np.linalg.norm(flat, axis=1, keepdims=True), 1e-12)
    resultant = float(np.linalg.norm(unit.mean(axis=0)))
    resultant = min(max(resultant, 1e-9), 1.0)
    return float(np.degrees(np.sqrt(-2.0 * np.log(resultant))))


def field_stats(
    field: dict,
    object_pose: np.ndarray,
    goal_xy: np.ndarray,
    target_object: int,
) -> dict:
    """Numbers a caption can quote instead of the reader eyeballing the heat map."""
    probs = field["gate_prob"]
    object_xy = object_pose[target_object, :2]
    alignment = goal_alignment(field["delta_xy"], object_xy, goal_xy)
    fires = probs >= 0.5

    # Split the grid by whether a push from that pusher position *should* help:
    # standing on the far side of the object from the goal pushes it goalward.
    axis = field["axis"]
    grid_x, grid_y = np.meshgrid(axis, axis)
    pusher_to_object = np.stack([object_xy[0] - grid_x, object_xy[1] - grid_y], axis=-1)
    object_to_goal = goal_xy - object_xy
    object_to_goal = object_to_goal / max(float(np.linalg.norm(object_to_goal)), 1e-9)
    behind = (pusher_to_object @ object_to_goal) > 0.0

    def masked_mean(mask: np.ndarray) -> float | None:
        return round(float(alignment[mask].mean()), 3) if bool(mask.any()) else None

    return {
        "gate_prob_min": round(float(probs.min()), 3),
        "gate_prob_max": round(float(probs.max()), 3),
        "gate_prob_spread": round(float(probs.max() - probs.min()), 3),
        "fraction_of_grid_where_gate_fires": round(float(fires.mean()), 3),
        "delta_direction_circular_std_deg": round(direction_circular_std_deg(field["delta_xy"]), 1),
        # Matches the RESULTS.md claim: goalward when the pusher is behind the
        # object, away once it has overshot to the goal side.
        "goal_alignment_pusher_behind": masked_mean(behind),
        "goal_alignment_pusher_on_goal_side": masked_mean(~behind),
        "goal_alignment_behind_and_gate_fires": masked_mean(behind & fires),
        "mean_delta_magnitude_m": round(float(np.linalg.norm(field["delta_xy"], axis=-1).mean()), 4),
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    raw = np.load(args.data)
    states = raw["s_t"].astype(np.float32)
    num_objects = infer_num_objects_from_state_dim(int(states.shape[1]))
    layout = StateLayout(num_objects=num_objects)
    base_state = states[args.state_index % states.shape[0]]

    object_pose = base_state[layout.object_pose_slice].reshape(num_objects, POSE_DIM)
    goal_xy = base_state[layout.goal_slice]
    target_object = int(args.target_object)

    panels = [
        ("Round 1: velocity-keyed features", args.round1_checkpoint),
        ("Round 2: contact-aware features", args.round2_checkpoint),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.9))
    fig.patch.set_facecolor("white")
    stats = {}
    mesh = None
    for ax, (title, checkpoint) in zip(axes, panels):
        model, config = load_sparse_model(checkpoint, device)
        field = sweep_gate_field(
            model, config, base_state, layout, num_objects, target_object,
            args.resolution, not args.keep_velocity, device,
        )
        panel_stats = field_stats(field, object_pose, goal_xy, target_object)
        stats[title] = panel_stats | {"checkpoint": str(checkpoint), "feature_mode": field["feature_mode"]}
        mesh = draw_panel(
            ax, field, object_pose, goal_xy, target_object, title,
            f"{field['feature_mode']} features  |  gate fires on "
            f"{panel_stats['fraction_of_grid_where_gate_fires'] * 100:.0f}% of the grid  "
            f"|  prob range {panel_stats['gate_prob_min']:.2f}-{panel_stats['gate_prob_max']:.2f}",
            args.quiver_stride,
        )

    fig.suptitle(
        "Where does the change gate fire, and which way does it think the target will go?",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.13, 1, 0.94))

    assert mesh is not None  # both panels always draw
    colorbar = fig.colorbar(mesh, ax=axes, orientation="horizontal", fraction=0.045, pad=0.10, aspect=44)
    colorbar.set_label("gate probability for the target object (background)", fontsize=9.5)
    velocity_note = "kept from the dataset" if args.keep_velocity else "set to zero"
    fig.text(
        0.5,
        0.035,
        "Arrows: predicted target motion (points away from the pusher, as a real push would);\n"
        "green = that push sends the target toward the goal, red = away.\n"
        f"Object velocities are {velocity_note} -- the near-rest states a sampling planner actually visits.",
        ha="center",
        va="top",
        fontsize=8.5,
        color="#4a4a4a",
        linespacing=1.5,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"gate_field_{num_objects}obj" + ("_withvel" if args.keep_velocity else "")
    png_path = args.out_dir / f"{stem}.png"
    fig.savefig(png_path, dpi=200)
    plt.close(fig)

    summary = {
        "figure": str(png_path),
        "data": str(args.data),
        "state_index": int(args.state_index),
        "target_object": target_object,
        "zero_velocity": not args.keep_velocity,
        "panels": stats,
    }
    (args.out_dir / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gate-probability / predicted-motion field (blog demo).")
    parser.add_argument("--data", type=Path, default=Path("data/transitions/plan_mixed_3obj.npz"))
    parser.add_argument("--state-index", type=int, default=0, help="Row of the dataset used as the frozen scene.")
    parser.add_argument("--round1-checkpoint", type=Path, default=Path("models/checkpoints/sparse_3obj_s0.pt"))
    parser.add_argument("--round2-checkpoint", type=Path, default=Path("models/checkpoints/sparse_contact_3obj_v1.pt"))
    parser.add_argument("--target-object", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=121)
    parser.add_argument("--quiver-stride", type=int, default=9)
    parser.add_argument("--keep-velocity", action="store_true", help="Sweep with the dataset's velocities instead of zeros.")
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/runs/demos"))
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


if __name__ == "__main__":
    main()
