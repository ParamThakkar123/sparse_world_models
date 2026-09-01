"""Render a planar scene state to an image, so the whole study can be run from pixels.

Why this exists
---------------
Every result in this project consumes *structured per-object state*: the model is handed
poses and velocities and never has to find the objects. That is the single scope limitation
a reviewer is most likely to treat as disqualifying, and it is load-bearing for the central
finding rather than incidental to it -- the momentum shortcut is available precisely because
velocity is served to the model as an input feature. The obvious question is whether the
shortcut survives when velocity has to be *inferred from images* instead.

This renderer makes that question answerable without regenerating a single episode. It
renders directly from the flat state vector that ``generate_transitions`` already stores, so
every existing dataset -- all four domains, both benchmarks, every seed -- can be turned into
an image dataset by a pure function of data already on disk.

Rendering from state rather than from a live simulator is the standard construction for
object-centric benchmarks (CLEVRER and friends are built the same way) and loses nothing
that matters here: the image is a deterministic function of the state, so no information the
task depends on is destroyed. What changes -- and it is the entire point -- is that the model
must now *recover* objects from pixels rather than being given them.

Design choices, and which of them are load-bearing
--------------------------------------------------
* **Fixed per-index colour palette.** Object ``i`` always gets palette colour ``i``. This is
  what CLEVRER-style benchmarks do, and it makes slot-to-object correspondence recoverable.
  It is disclosed rather than hidden because it makes the perception problem *easier* than
  the general case: a reviewer should read every pixel result as an upper bound on what a
  harder perception front end would give.
* **Velocity is NOT drawn.** No motion blur, no trails, no arrows. A single frame therefore
  contains no velocity information at all, which is what makes the pixel version a genuine
  test: a model that wants the momentum shortcut has to recover velocity by differencing
  consecutive frames, exactly as a real perception system would.
* **Yaw is visible.** Objects are rendered as rotated squares rather than discs, so the third
  pose dimension is recoverable from the image. Rendering discs would have made yaw
  unobservable and quietly reduced the prediction target from 3 DoF to 2.
* **The pusher is drawn in a reserved colour** not used by any object, so it cannot be
  confused for one.

Resolution is 96x96 by default rather than the more common 64x64: at 64x64 a 5 cm object
spans about 5 pixels, which is at the edge of what a slot-attention front end can segment,
and a perception failure would then be indistinguishable from the dynamics finding under
test. At 96x96 an object spans about 8 pixels.
"""

from __future__ import annotations

import numpy as np

from models.layout import StateLayout

# Distinct, well-separated hues. Index i is always object i -- see the docstring.
OBJECT_COLORS = np.array(
    [
        (228, 26, 28), (55, 126, 184), (77, 175, 74), (152, 78, 163),
        (255, 127, 0), (255, 214, 0), (166, 86, 40), (247, 129, 191),
        (153, 153, 153), (26, 188, 156), (241, 90, 34), (106, 61, 154),
        (178, 223, 138), (251, 154, 153), (202, 178, 214), (255, 255, 153),
        (177, 89, 40), (31, 120, 180), (51, 160, 44), (227, 26, 28),
    ],
    dtype=np.float32,
)
# Reserved for the pusher: white, which appears in no object colour.
PUSHER_COLOR = np.array((255, 255, 255), dtype=np.float32)
BACKGROUND_COLOR = np.array((18, 18, 24), dtype=np.float32)

OBJECT_HALF_EXTENT = 0.025
PUSHER_RADIUS = 0.02
# Slightly wider than the +/-0.26 placement bound so objects at the edge are fully visible
# rather than clipped, which would make their yaw partially unobservable.
WORLD_HALF_SPAN = 0.30


def _world_to_pixel(xy: np.ndarray, resolution: int) -> np.ndarray:
    """Map world metres to pixel coordinates. Y is flipped so +y is up in the image."""
    normalised = (xy + WORLD_HALF_SPAN) / (2.0 * WORLD_HALF_SPAN)
    pixel = normalised * (resolution - 1)
    pixel[..., 1] = (resolution - 1) - pixel[..., 1]
    return pixel


def render_state(
    state: np.ndarray,
    num_objects: int,
    resolution: int = 96,
) -> np.ndarray:
    """Render one flat state vector to an ``(resolution, resolution, 3)`` uint8 image.

    ``state`` is the layout ``generate_transitions.flatten_state`` produces: pusher xy, then
    per-object ``(x, y, yaw)``, then per-object 6-vector velocities, then the goal.
    """
    layout = StateLayout(num_objects=num_objects)
    pose = state[layout.object_pose_slice].reshape(num_objects, 3)
    pusher_xy = state[:2]

    image = np.broadcast_to(BACKGROUND_COLOR, (resolution, resolution, 3)).copy()

    # Pixel-centre coordinates in world units, used to test membership analytically rather
    # than rasterising polygons -- exact, and fast enough to render whole datasets.
    grid = np.arange(resolution, dtype=np.float32)
    xs = (grid / (resolution - 1)) * 2.0 * WORLD_HALF_SPAN - WORLD_HALF_SPAN
    ys = WORLD_HALF_SPAN - (grid / (resolution - 1)) * 2.0 * WORLD_HALF_SPAN
    world_x, world_y = np.meshgrid(xs, ys)

    for index in range(num_objects):
        cx, cy, yaw = pose[index]
        dx = world_x - cx
        dy = world_y - cy
        cos, sin = np.cos(-yaw), np.sin(-yaw)
        # Rotate the query point into the object's frame, then test against an axis-aligned
        # square. This is what makes yaw visible in the image.
        local_x = cos * dx - sin * dy
        local_y = sin * dx + cos * dy
        inside = (np.abs(local_x) <= OBJECT_HALF_EXTENT) & (np.abs(local_y) <= OBJECT_HALF_EXTENT)
        image[inside] = OBJECT_COLORS[index % len(OBJECT_COLORS)]

    # The pusher is drawn last so it occludes objects it overlaps, which is what a real
    # camera would see and keeps the depth ordering consistent across frames.
    distance = np.sqrt((world_x - pusher_xy[0]) ** 2 + (world_y - pusher_xy[1]) ** 2)
    image[distance <= PUSHER_RADIUS] = PUSHER_COLOR

    return image.astype(np.uint8)


def render_dataset(
    states: np.ndarray,
    num_objects: int,
    resolution: int = 96,
    batch_report: int | None = None,
) -> np.ndarray:
    """Render a stack of states to ``(n, resolution, resolution, 3)`` uint8."""
    frames = np.empty((states.shape[0], resolution, resolution, 3), dtype=np.uint8)
    for index in range(states.shape[0]):
        frames[index] = render_state(states[index], num_objects, resolution)
        if batch_report and index and index % batch_report == 0:
            print(f"    rendered {index}/{states.shape[0]}", flush=True)
    return frames


def object_pixel_positions(
    state: np.ndarray, num_objects: int, resolution: int = 96
) -> np.ndarray:
    """Ground-truth object centres in pixel coordinates.

    Used only to *evaluate* slot-to-object matching, never as model input. Kept here rather
    than in the experiment so the world-to-pixel convention has exactly one definition.
    """
    layout = StateLayout(num_objects=num_objects)
    pose = state[layout.object_pose_slice].reshape(num_objects, 3)
    return _world_to_pixel(pose[:, :2].copy(), resolution)
