# Experiment Tracking

Use plain file-based logging for now. This keeps the project lightweight while still
recording the essentials needed for workshop experiments.

## Layout

Each run should create:

- `experiments/runs/<run_name>/config.json`
- `experiments/runs/<run_name>/metrics.csv`
- `experiments/runs/<run_name>/summary.json`

## What to log

- `config.json`: simulator, object count, model type, seed, training hyperparameters
- `metrics.csv`: per-step or per-epoch training and evaluation metrics
- `summary.json`: final best metrics and short notes

## Suggested metrics

- `train_loss`
- `val_loss`
- `rollout_mse`
- `success_rate`
- `planning_return`

## Minimal usage

```python
from experiments.logging import ExperimentLogger

logger = ExperimentLogger(run_name="mujoco_push_3obj_seed0")
logger.log_config(
    {
        "simulator": "mujoco",
        "task": "tabletop_pushing",
        "num_objects": 3,
        "model": "sparse_residual_world_model",
        "seed": 0,
    }
)

for step in range(10):
    logger.log_metrics(step, train_loss=1.0 / (step + 1), val_loss=1.2 / (step + 1))

logger.log_summary({"best_val_loss": 0.12, "notes": "initial smoke test"})
```

Upgrade to Weights & Biases later only if you need remote dashboards, shared runs, or
sweep management.

