# Frozen Baseline: Tabletop Push 3-Object Setup

This is the frozen baseline configuration for the initial sparse/residual world model experiments.
Use this setup as the default reference point for dataset generation, training, and evaluation.

## Task

- Simulator: MuJoCo
- Domain: tabletop pushing
- Object count: 3
- Action space: continuous planar end-effector delta action
- State representation: per-object planar pose `(x, y, yaw)` with optional velocity in the logged full state

## Environment

Defined in:

- `models/envs/assets/tabletop_push_3obj.xml`
- `models/envs/mujoco_tabletop.py`

Frozen parameters:

- `control_dt = 0.05`
- `physics_dt = 0.005`
- `action_scale = 0.04`
- `pusher_bounds = (-0.26, 0.26)`
- `object_bounds = (-0.18, 0.18)`
- `min_object_separation = 0.12`
- `goal_clearance = 0.1`
- `goal_xy = (0.18, 0.18)`
- `max_steps = 100` for the verified scripted sparsity benchmark

## Scripted Policy

Defined in:

- `models/policies.py`

Frozen parameters:

- `approach_offset = 0.08`
- `gain = 6.0`
- `switch_threshold = 0.028`
- `push_distance = 0.015`

## Logged Transition Targets

Generated datasets must include:

- ground-truth transition tuple: `(s_t, a_t, s_{t+1})`
- per-object changed/unchanged mask
- per-object planar delta vector `(dx, dy, dtheta)`

Relevant saved keys:

- `s_t`, `a_t`, `s_t1`
- `object_change_mask`
- `object_delta`

## Verified Sparsity Check

Reference dataset:

- `data/transitions/scripted_eval_tuned_10x100.npz`

Verified metrics:

- success rate: `1.0`
- mean changed fraction per object-step: `0.2411`
- mean changed objects per step: `0.7234 / 3`
- steps with any object change: `0.6979`

Interpretation:

This setup is sparse enough for the current paper story and should be treated as the baseline configuration unless an experiment explicitly studies deviations from it.
