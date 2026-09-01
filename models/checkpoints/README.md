# Checkpoints

Full `.pt` checkpoints are not tracked in git. They are published to Hugging Face and fetched on demand.

- HF repo: `ParamThakkar123/sparse_world_models`
- Browse: `https://huggingface.co/datasets/ParamThakkar123/sparse_world_models/tree/main/models/checkpoints`
- Direct raw: `https://huggingface.co/datasets/ParamThakkar123/sparse_world_models/resolve/main/models/checkpoints/<file>.pt`

The browser demo does not need these files. It runs from the exported weights in `docs/assets/model.json`, which are tracked in git and served by GitHub Pages. The raw checkpoints are for reproduction and for training new runs.

## Download a checkpoint

```bash
pip install huggingface_hub
python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download(repo_id='ParamThakkar123/sparse_world_models', repo_type='dataset', filename='models/checkpoints/sparse_planar_motion_contact_3obj_s0.pt'))"
```

Or with curl:

```bash
curl -L -o sparse_planar_motion_contact_3obj_s0.pt \
  https://huggingface.co/datasets/ParamThakkar123/sparse_world_models/resolve/main/models/checkpoints/sparse_planar_motion_contact_3obj_s0.pt
```

## Push new checkpoints

```bash
pip install huggingface_hub
hf auth login
python -m experiments.publish_to_hf --repo ParamThakkar123/sparse_world_models --include-checkpoints
```

To publish the JSON the demo actually serves:

```bash
python -m experiments.publish_to_hf --repo ParamThakkar123/sparse_world_models --include-assets
```

The demo in `docs/assets/app.js` tries local `assets/model.json` first, then falls back to the HF raw URL, so forks work without rebuilding.

## Push to GitHub instead

If you prefer GitHub Releases over HF, attach the files to a release:

```bash
gh release create checkpoints-v1 models/checkpoints/*.pt docs/assets/*.json
```

Then reference them as `https://github.com/ParamThakkar123/sparse_world_models/releases/download/checkpoints-v1/<file>.pt`.
