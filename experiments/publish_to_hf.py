"""Publish checkpoints to Hugging Face Hub and wire the demo to use them.

Checkpoints are gitignored locally, so the demo ships only the exported JSON in
docs/assets/model.json. This script pushes the full .pt checkpoints to HF and
optionally updates the demo to fetch weights from HF when local assets are
missing (useful for forks or for loading a new checkpoint without rebuilding
the site).

Usage:
  pip install huggingface_hub
  hf auth login
  python -m experiments.publish_to_hf --repo ParamThakkar123/sparse_world_models --include-checkpoints
  python -m experiments.publish_to_hf --repo ParamThakkar123/sparse_world_models --include-assets

The demo itself already tries local assets first, then falls back to
https://huggingface.co/datasets/<repo>/resolve/main/docs/assets/model.json,
so no code change is required after the upload. Pass --update-pointer to write
docs/assets/hf.json with the repo id, which app.js reads as a hint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_TYPE = "dataset"


def upload_folder(repo_id: str, local_dir: Path, path_in_repo: str) -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit("pip install huggingface_hub") from exc
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type=REPO_TYPE, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type=REPO_TYPE,
        folder_path=str(local_dir),
        path_in_repo=path_in_repo,
    )
    print(f"uploaded {local_dir} to {repo_id}:{path_in_repo}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publish checkpoints and demo assets to HF Hub."
    )
    parser.add_argument(
        "--repo",
        required=True,
        help="HF repo id, e.g. ParamThakkar123/sparse_world_models",
    )
    parser.add_argument(
        "--include-checkpoints", action="store_true", help="upload models/checkpoints/"
    )
    parser.add_argument(
        "--include-assets", action="store_true", help="upload docs/assets/*.json"
    )
    parser.add_argument(
        "--update-pointer",
        action="store_true",
        help="write docs/assets/hf.json with repo id",
    )
    args = parser.parse_args()

    if not args.include_checkpoints and not args.include_assets:
        args.include_checkpoints = True
        args.include_assets = True

    if args.include_checkpoints:
        ckpt = Path("models/checkpoints")
        if not any(ckpt.glob("*.pt")):
            print("no checkpoints found in models/checkpoints, skipping")
        else:
            upload_folder(args.repo, ckpt, "models/checkpoints")

    if args.include_assets:
        assets = Path("docs/assets")
        if not assets.exists():
            raise SystemExit("docs/assets not found")
        upload_folder(args.repo, assets, "docs/assets")

    if args.update_pointer:
        pointer = Path("docs/assets/hf.json")
        pointer.write_text(
            json.dumps(
                {
                    "repo": args.repo,
                    "base": f"https://huggingface.co/datasets/{args.repo}/resolve/main",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"wrote {pointer}")

    print("done. Demo will load from local assets first, then fall back to HF.")


if __name__ == "__main__":
    main()
