"""Top-down 2D rendering of tabletop-push scenes from planar state.

The env exposes state, not pixels, so demos draw the scene directly: each object
is a square footprint at its planar pose ``(x, y, theta)``, the pusher a disc,
the goal a ring. Geometry constants mirror the MuJoCo XML in
``models/envs/mujoco_tabletop.py`` (box half-extent 0.025, pusher sphere radius
0.02, goal site radius 0.045, tabletop half-width 0.34) so the drawing is to
scale with the simulator.

A ``ScenePainter`` owns the artists for one axes and is re-``update``d per frame,
which is what makes an animation cheap: patches are created once and only their
vertices change.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")  # demos always render headless to file

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon

# Geometry, mirroring models/envs/mujoco_tabletop.py.
TABLE_HALF = 0.34
BOX_HALF = 0.025
PUSHER_RADIUS = 0.02
GOAL_RADIUS = 0.045

# Object colours in checkpoint/material order (obj_red, obj_green, ... obj_slate).
OBJECT_COLORS = [
    "#d94040",
    "#33b34d",
    "#3359d9",
    "#eb8c2e",
    "#29a89e",
    "#db6199",
    "#8fa12e",
    "#597a9e",
]

TABLE_FACE = "#efece7"
TABLE_EDGE = "#c9c3b8"
PUSHER_COLOR = "#1a1a1a"
GOAL_COLOR = "#f2cc33"
TRUTH_COLOR = "#8c8c8c"


def object_color(index: int) -> str:
    return OBJECT_COLORS[index % len(OBJECT_COLORS)]


def box_corners(pose: Sequence[float], half: float = BOX_HALF) -> np.ndarray:
    """Four corners ``(4, 2)`` of a square footprint at planar pose ``(x, y, theta)``."""
    x, y, theta = float(pose[0]), float(pose[1]), float(pose[2])
    unit = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]) * half
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    rotation = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    return unit @ rotation.T + np.array([x, y])


class ScenePainter:
    """Draws one tabletop panel and re-renders it per frame.

    ``truth_poses`` passed to :meth:`update` overlays the ground-truth footprints
    as dashed outlines behind the predicted ones -- that gap *is* the prediction
    error, and it is the whole point of the model panels.
    """

    def __init__(
        self,
        ax: Axes,
        num_objects: int,
        goal_xy: Sequence[float] | None = None,
        title: str = "",
        subtitle: str = "",
        show_truth_overlay: bool = False,
        show_trails: bool = True,
        target_object: int | None = None,
        view_half: float = TABLE_HALF,
    ):
        self.ax = ax
        self.num_objects = num_objects
        self.show_truth_overlay = show_truth_overlay
        self.show_trails = show_trails
        self.target_object = target_object

        # Objects are confined well inside the table (bounds +-0.18, pusher +-0.26),
        # so cropping below the full half-width 0.34 stops the scene floating in space.
        ax.set_xlim(-view_half, view_half)
        ax.set_ylim(-view_half, view_half)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor(TABLE_FACE)
        for spine in ax.spines.values():
            spine.set_color(TABLE_EDGE)
        if title:
            ax.set_title(title, fontsize=11, fontweight="bold", pad=8)

        self.subtitle_text = ax.text(
            0.5,
            0.975,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8.5,
            color="#4a4a4a",
        )

        if goal_xy is not None:
            ax.add_patch(
                Circle(
                    (float(goal_xy[0]), float(goal_xy[1])),
                    GOAL_RADIUS,
                    facecolor=GOAL_COLOR,
                    alpha=0.35,
                    edgecolor=GOAL_COLOR,
                    linewidth=1.4,
                    zorder=1,
                )
            )

        # Ground-truth outlines sit *under* the predicted footprints.
        self.truth_patches: list[Polygon] = []
        if show_truth_overlay:
            for _ in range(num_objects):
                patch = Polygon(
                    np.zeros((4, 2)),
                    closed=True,
                    facecolor="none",
                    edgecolor=TRUTH_COLOR,
                    linestyle=(0, (3, 2)),
                    linewidth=1.2,
                    zorder=2,
                )
                ax.add_patch(patch)
                self.truth_patches.append(patch)

        self.trails: list[Line2D] = []
        self.trail_xy: list[list[tuple[float, float]]] = [[] for _ in range(num_objects)]
        if show_trails:
            for idx in range(num_objects):
                (line,) = ax.plot(
                    [], [], color=object_color(idx), linewidth=1.1, alpha=0.5, zorder=3
                )
                self.trails.append(line)

        self.object_patches: list[Polygon] = []
        for idx in range(num_objects):
            is_target = target_object is not None and idx == target_object
            patch = Polygon(
                np.zeros((4, 2)),
                closed=True,
                facecolor=object_color(idx),
                edgecolor="#20202a" if is_target else "#ffffff",
                linewidth=2.0 if is_target else 1.0,
                alpha=0.92,
                zorder=4,
            )
            ax.add_patch(patch)
            self.object_patches.append(patch)

        self.pusher = Circle(
            (0.0, 0.0),
            PUSHER_RADIUS,
            facecolor=PUSHER_COLOR,
            edgecolor="#ffffff",
            linewidth=1.0,
            zorder=5,
        )
        ax.add_patch(self.pusher)

    def update(
        self,
        poses: np.ndarray,
        pusher_xy: Sequence[float] | None = None,
        truth_poses: np.ndarray | None = None,
        subtitle: str | None = None,
    ) -> None:
        """Redraw the panel. ``poses`` is ``(num_objects, 3)``."""
        poses = np.asarray(poses, dtype=float).reshape(self.num_objects, 3)
        for idx, patch in enumerate(self.object_patches):
            patch.set_xy(box_corners(poses[idx]))
            if self.show_trails:
                self.trail_xy[idx].append((poses[idx, 0], poses[idx, 1]))
                trail = np.asarray(self.trail_xy[idx])
                self.trails[idx].set_data(trail[:, 0], trail[:, 1])

        if truth_poses is not None and self.truth_patches:
            truth_poses = np.asarray(truth_poses, dtype=float).reshape(self.num_objects, 3)
            for idx, patch in enumerate(self.truth_patches):
                patch.set_xy(box_corners(truth_poses[idx]))

        if pusher_xy is not None:
            self.pusher.set_center((float(pusher_xy[0]), float(pusher_xy[1])))

        if subtitle is not None:
            self.subtitle_text.set_text(subtitle)

    def reset_trails(self) -> None:
        self.trail_xy = [[] for _ in range(self.num_objects)]
        for line in self.trails:
            line.set_data([], [])


def truth_overlay_legend(fig: Figure, loc: str = "lower center") -> None:
    """Shared legend explaining the dashed ground-truth outline and the pusher."""
    handles = [
        Line2D([0], [0], color=TRUTH_COLOR, linestyle=(0, (3, 2)), linewidth=1.4, label="ground truth"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#9aa0aa", markersize=9, label="predicted pose"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PUSHER_COLOR, markersize=8, label="pusher"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=GOAL_COLOR, markersize=10, alpha=0.6, label="goal"),
    ]
    fig.legend(handles=handles, loc=loc, ncol=4, frameon=False, fontsize=9)


def save_gif(
    fig: Figure,
    update: Callable[[int], Iterable],
    num_frames: int,
    path: Path,
    fps: int = 5,
    hold_last_ms: int = 1800,
) -> Path:
    """Write a looping GIF that pauses on the final frame before repeating.

    Repeating the last frame index does not work: Pillow drops frames identical to
    their predecessor when writing a GIF, so the hold silently disappears. Instead
    the finished file is re-encoded with a per-frame duration list whose last entry
    is ``hold_last_ms`` -- without it a 20-step rollout loops before the reader has
    seen where it ended up.
    """
    animation = FuncAnimation(fig, update, frames=num_frames, interval=1000 // max(1, fps), blit=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(str(path), writer=PillowWriter(fps=fps))
    plt.close(fig)

    from PIL import Image, ImageSequence

    with Image.open(path) as source:
        frames = [frame.copy() for frame in ImageSequence.Iterator(source)]
    durations = [1000 // max(1, fps)] * len(frames)
    durations[-1] = hold_last_ms
    frames[0].save(
        path, save_all=True, append_images=frames[1:], duration=durations, loop=0, disposal=2
    )
    return path


def write_scene_json(path: Path, meta: dict, panels: dict[str, dict[str, np.ndarray]]) -> Path:
    """Dump per-frame trajectories so a web page can replay without recomputation.

    ``panels`` maps a panel name to arrays keyed ``poses`` ``(T, N, 3)`` and
    optionally ``pusher`` ``(T, 2)`` / ``error`` ``(T,)``.
    """
    payload = {
        "meta": meta
        | {
            "geometry": {
                "table_half": TABLE_HALF,
                "box_half": BOX_HALF,
                "pusher_radius": PUSHER_RADIUS,
                "goal_radius": GOAL_RADIUS,
            },
            "object_colors": OBJECT_COLORS[: int(meta.get("num_objects", len(OBJECT_COLORS)))],
        },
        "panels": {
            name: {key: np.asarray(value).round(5).tolist() for key, value in series.items()}
            for name, series in panels.items()
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path
