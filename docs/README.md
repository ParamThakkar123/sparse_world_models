# The live demo (GitHub Pages)

A static site that runs the **trained model from the paper** in the browser, on live physics,
scored against the trivial-rule battery frame by frame. It exists because this project's
central claim is uncomfortable — a rule with no parameters beats a trained object-centric
world model — and a claim like that is better checked than asserted.

## Publishing it

GitHub Pages, no build step, no workflow file needed:

**Settings → Pages → Source: “Deploy from a branch” → Branch: `main`, folder: `/docs`.**

The site is then at `https://paramthakkar123.github.io/WorldModelsWorkshop/`. Everything it needs is in this
folder; nothing is fetched from a CDN, so it also works offline and inside a corporate proxy.

**Camera-ready:** the PDF at `paper/main.pdf` now links this URL in its Interactive demo
section. For anonymous submission this link was omitted per `SUBMISSION_RUNBOOK.md` step 3.

## Running it locally

ES modules and `fetch` need a real origin, so opening `index.html` from the filesystem will
not work:

```bash
python -m http.server 8000 --directory docs
# then http://localhost:8000/
```

## Regenerating the assets

```bash
python -m experiments.export_web_model      # -> assets/model.json, episodes.json, parity_fixture.json
python -m pytest tests/test_web_export.py   # the JS port must match PyTorch to 1e-4
```

`export_web_model.py` ships **two** checkpoints. The live sandbox is planar physics, so it
runs the planar-trained gate; the replay tab shows recorded MuJoCo/Box2D/Chipmunk episodes,
so it runs the tabletop-trained one. Using a single model for both would put a domain shift
on screen with nothing on the page to label it.

## What each file is

| file | what it is |
|---|---|
| `index.html` | the page: four tasks, one canvas, one scoreboard |
| `assets/model.js` | port of the gate + delta head and the contact featurisation. Checked against PyTorch by `tests/test_web_export.py` |
| `assets/battery.js` | port of the eleven trivial rules and the F1 bookkeeping from `onset_shortcut_audit.py` |
| `assets/planar.js` | port of `models/envs/planar_push.py`, so the sandbox is live physics rather than a recording |
| `assets/app.js` | tasks, rendering, the CEM planner, the scoreboard |
| `assets/model.json` | exported weights, one entry per checkpoint |
| `assets/episodes.json` | recorded episodes for the replay tab, one per engine |
| `assets/parity_fixture.json` | PyTorch outputs the JS port is tested against |

## Two things the page is careful about

**Frames are motion-filtered before scoring.** Only frames where some object moved more than
0.02 m enter the scoreboard, which is the `--min-max-xy-delta 0.02` filter every benchmark in
the paper is built with. Without it about 95% of steps are the pusher travelling with nothing
happening, every rule scores near zero, and no number on the page would be comparable to any
number in the paper.

**The scoreboard reports onset F1 separately.** Overall F1 on a motion-filtered stream is
dominated by continuation — an object in motion stays in motion — so it measures persistence
rather than prediction. Onset F1 scores only objects currently at rest, which can start moving
only through contact. It is the column that carries the argument.
