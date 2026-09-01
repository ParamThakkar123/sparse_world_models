"""Export a trained gate + delta head, and some real episodes, for the browser demo.

The demo in `docs/` runs the **actual trained weights**, not a re-fit or an approximation.
That is the whole point of it: this project's central claim is that a one-line rule beats
these models, and a reader should be able to watch that happen on a live scene rather than
take a table's word for it. A demo running a different model would be worthless here.

Two things get exported:

* **Weights.** The gate MLP and the delta head, as plain nested lists. The models are tiny
  (< 0.1 M parameters, two-layer MLPs), so a JSON of float32 weights is a few hundred KB and
  a forward pass is a handful of matrix multiplies -- no ONNX runtime, no WASM blob, no CDN.
  The exported model must use the ``contact`` featurisation: it is fixed-width (19 per
  object) for any object count, which is what lets one exported checkpoint drive the 3-, 5-
  and 8-object scenes in the browser.

* **Episodes.** MuJoCo, Box2D and Chipmunk cannot run in a browser, so the multi-engine
  scenes are replayed from recorded state vectors while the model predicts on each frame
  live. The planar domain is different: it is ours and it is 300 lines of arithmetic, so the
  demo re-implements it in JavaScript and the user can drive it interactively.

Usage:

    python -m experiments.export_web_model                       # defaults, writes docs/assets/
    python -m experiments.export_web_model --checkpoint models/checkpoints/sparse_contact_3obj_v1.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.compare_phase4_models import load_dataset, load_sparse_model
from experiments.train_sparse_model import (
    CONTACT_RADIUS,
    PUSHER_ACTION_SCALE,
    PUSHER_BOUND,
    build_object_features_by_mode,
    resolve_gate_estimator,
)
from models.layout import StateLayout

# Two checkpoints, because the demo runs in two regimes and using one model for both would
# be dishonest in a way the viewer could not see. The live sandbox is the PLANAR environment
# (the only one that can run in a browser), so it gets the planar-trained gate; the replay tab
# shows recorded MuJoCo/Box2D/Chipmunk episodes, which the tabletop-trained gate is the right
# model for. Scoring a tabletop model on live planar physics would show a model failing at a
# domain shift and let a viewer read it as the model being bad.
DEFAULT_MODELS = {
    "planar": "models/checkpoints/sparse_planar_motion_contact_3obj_s0.pt",
    "tabletop": "models/checkpoints/sparse_contact_3obj_v1.pt",
}

# Recorded scenes for the replay tab, one per engine. Each is a contiguous window so the
# demo can show motion; a shuffled split would replay as teleporting objects.
DEFAULT_EPISODES = [
    ("tabletop", "MuJoCo", "data/transitions/scale_3obj_s0.npz", 3),
    ("billiards", "Box2D", "data/transitions/xd_billiards_5obj_s0.npz", 5),
    ("clutter", "Chipmunk2D", "data/transitions/xd_clutter_5obj_s0.npz", 5),
    ("planar", "ours", "data/transitions/xd_planar_3obj_s0.npz", 3),
]


def export_linear_stack(module: torch.nn.Module) -> list[dict]:
    """Flatten an ``nn.Sequential`` of Linear/ReLU into a list the JS forward pass can walk."""
    layers = []
    for child in module.modules():
        if isinstance(child, torch.nn.Linear):
            layers.append(
                {
                    "weight": child.weight.detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                    .tolist(),
                    "bias": child.bias.detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                    .tolist(),
                }
            )
    return layers


def export_weights(checkpoint: Path, device: torch.device) -> dict:
    model, config = load_sparse_model(checkpoint, device)
    feature_mode = str(config.get("feature_mode", "global"))
    if feature_mode != "contact":
        raise SystemExit(
            f"{checkpoint.name} uses feature_mode='{feature_mode}'. The browser demo needs "
            "'contact': it is the only fixed-width featurisation, so it is the only one that "
            "can drive scenes with a different object count from the one it was trained on."
        )
    delta_head_type = str(config.get("delta_head_type", "mse"))
    if delta_head_type != "mse":
        raise SystemExit(
            f"{checkpoint.name} has a '{delta_head_type}' delta head. The JS forward pass "
            "implements the deterministic head only; exporting a distributional head without "
            "implementing its point estimate would silently show the wrong prediction."
        )

    gate_layers = export_linear_stack(model.gate.mlp)
    delta_layers = export_linear_stack(model.delta_head)
    parameters = sum(
        np.prod(np.shape(layer["weight"])) + len(layer["bias"])
        for layer in gate_layers + delta_layers
    )
    return {
        "source_checkpoint": checkpoint.name,
        "feature_mode": feature_mode,
        "delta_head_type": delta_head_type,
        "estimator": str(
            config.get(
                "eval_estimator",
                resolve_gate_estimator(False, str(config["estimator"])),
            )
        ),
        "temperature": float(config["temperature"]),
        "gate": gate_layers,
        "delta": delta_layers,
        "num_parameters": int(parameters),
        "constants": {
            "contact_radius": float(CONTACT_RADIUS),
            "pusher_action_scale": float(PUSHER_ACTION_SCALE),
            "pusher_bound": float(PUSHER_BOUND),
        },
    }


def sample_episode(
    path: Path, count: int, length: int, start_fraction: float = 0.25
) -> dict:
    """A contiguous window of real transitions, with the labels the model is scored against."""
    data = load_dataset(path)
    total = data["state"].shape[0]
    if total < length:
        length = total
    start = int(total * start_fraction)
    start = min(start, total - length)
    stop = start + length
    layout = StateLayout(num_objects=count)
    state = data["state"][start:stop]
    return {
        "num_objects": count,
        "state": np.round(state, 5).astype(float).tolist(),
        "action": np.round(data["action"][start:stop], 5).astype(float).tolist(),
        "target_mask": data["target_mask"][start:stop].astype(int).tolist(),
        "goal_xy": np.round(state[0, layout.goal_slice], 5).astype(float).tolist(),
    }


def verify_against_torch(
    weights: dict, episode: dict, checkpoint: Path, device: torch.device
) -> dict:
    """Record what the torch model predicts, so the JS port can be checked against it.

    Without this the browser demo could quietly diverge from the paper -- a transposed
    weight or an off-by-one in the feature builder would still produce plausible-looking
    gates. `tests/test_web_export.py` runs the JS forward pass over these same inputs and
    requires agreement.
    """
    model, _ = load_sparse_model(checkpoint, device)
    state = torch.tensor(episode["state"], dtype=torch.float32, device=device)
    action = torch.tensor(episode["action"], dtype=torch.float32, device=device)
    with torch.no_grad():
        features = build_object_features_by_mode(state, action, weights["feature_mode"])
        output = model(
            features,
            estimator=weights["estimator"],
            temperature=weights["temperature"],
            hard=True,
        )
        probs = output.gate.probs.cpu().numpy()
        delta = output.delta.cpu().numpy()
    return {
        "features": np.round(features.cpu().numpy(), 6).astype(float).tolist(),
        "gate_probs": np.round(probs, 6).astype(float).tolist(),
        "delta": np.round(delta, 6).astype(float).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export the trained model for the web demo."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="name=path pairs. Defaults to a planar and a tabletop checkpoint.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("docs/assets"))
    parser.add_argument("--episode-length", type=int, default=240)
    parser.add_argument(
        "--fixture-length",
        type=int,
        default=16,
        help="Rows kept in the parity fixture the JS port is tested against.",
    )
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected = (
        DEFAULT_MODELS
        if args.models is None
        else dict(pair.split("=", 1) for pair in args.models)
    )
    exported = {}
    for name, path in selected.items():
        checkpoint = Path(path)
        if not checkpoint.exists():
            raise SystemExit(f"{name}: {checkpoint} not found")
        exported[name] = export_weights(checkpoint, device)
        print(
            f"  {name}: {exported[name]['num_parameters']} parameters "
            f"from {checkpoint.name}"
        )
    bundle = {"default": next(iter(exported)), "models": exported}
    (args.output_dir / "model.json").write_text(json.dumps(bundle), encoding="utf-8")

    episodes = {}
    for name, engine, path, count in DEFAULT_EPISODES:
        source = Path(path)
        if not source.exists():
            print(f"  skip {name}: {source} not generated")
            continue
        episode = sample_episode(source, count, args.episode_length)
        episode["engine"] = engine
        episodes[name] = episode
        print(f"  {name} ({engine}): {len(episode['state'])} frames, {count} objects")
    if not episodes:
        raise SystemExit("No episode data found. Generate at least one dataset first.")
    (args.output_dir / "episodes.json").write_text(
        json.dumps(episodes), encoding="utf-8"
    )

    # One fixture per exported model: the JS port has to agree with PyTorch for every set of
    # weights the page can load, not just the first one.
    fixture_source = next(iter(episodes.values()))
    fixture_episode = {
        key: (
            value[: args.fixture_length]
            if isinstance(value, list) and key in ("state", "action", "target_mask")
            else value
        )
        for key, value in fixture_source.items()
    }
    fixtures = {}
    for name, weights in exported.items():
        fixture = verify_against_torch(
            weights, fixture_episode, Path(selected[name]), device
        )
        fixture["state"] = fixture_episode["state"]
        fixture["action"] = fixture_episode["action"]
        fixture["num_objects"] = fixture_episode["num_objects"]
        fixtures[name] = fixture
    (args.output_dir / "parity_fixture.json").write_text(
        json.dumps(fixtures), encoding="utf-8"
    )
    print(f"parity_fixture.json: {len(fixtures)} model(s) x {args.fixture_length} rows")


if __name__ == "__main__":
    main()
