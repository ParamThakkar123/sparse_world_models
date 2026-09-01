"""Blog/demo asset generation: animated and static comparisons of the models.

These scripts do not produce any paper numbers -- they render *already-evaluated*
models into visuals for write-ups. Every demo reads the same checkpoints and
held-out data as the corresponding experiment in ``experiments/``, so a figure
here can always be traced back to a row in ``experiments/RESULTS.md``.

The environment is state-only (no MuJoCo renderer is used anywhere in the repo),
so scenes are drawn top-down from the planar state in ``render2d``: the task is
planar pushing, and a top-down view shows prediction error far more legibly than
a perspective render would.

Each demo writes both a rendered asset (GIF / PNG) and the raw per-frame
trajectory as JSON, so the same run can drive a static blog image or an
interactive web page without recomputation.
"""

from experiments.demos.render2d import (
    BOX_HALF,
    GOAL_RADIUS,
    OBJECT_COLORS,
    PUSHER_RADIUS,
    TABLE_HALF,
    ScenePainter,
    box_corners,
    save_gif,
    write_scene_json,
)

__all__ = [
    "BOX_HALF",
    "GOAL_RADIUS",
    "OBJECT_COLORS",
    "PUSHER_RADIUS",
    "TABLE_HALF",
    "ScenePainter",
    "box_corners",
    "save_gif",
    "write_scene_json",
]
