"""Bundle the demo JSON trajectories into one self-contained interactive page.

Every demo writes ``<name>.json`` alongside its GIF, holding the exact per-frame
poses that were rendered. This script inlines those files into a single HTML
document with a canvas replay, a scrub bar and per-panel readouts, so a reader
can stop on any horizon and compare panels themselves -- the thing a GIF cannot
do. No network requests, no build step, no external assets: the page is one file
that can be dropped into a blog post or opened directly.

Run the demos first, then:

    python -m experiments.demos.build_interactive

Missing demo JSONs are skipped with a warning rather than failing, so the page
can be built from whatever has been generated so far.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# (json stem, section title, blurb) -- order defines page order.
DEMO_SECTIONS = [
    (
        "rollout_drift_3obj_typical",
        "Closing the loop: 20-step rollout",
        "Each panel feeds its own prediction back in as the next input. The pusher, "
        "velocities and goal come from the ground truth at every step, so all three "
        "models are driven identically and the only difference is what they predict.",
    ),
    (
        "rollout_drift_3obj_active",
        "The same rollout on a high-motion window",
        "Selected from the most active 10% of launch points. Sparse still compounds far "
        "less error than dense, but this is where its own failure mode shows: the delta "
        "head can keep pushing the moving object past where it really went.",
    ),
    (
        "planning_3obj",
        "Planning: each model as the forward simulator",
        "Sampling-based MPC (CEM), replanning every step. Only the world model changes "
        "between panels -- planner, cost and seeds are identical.",
    ),
    (
        "count_transfer_from3obj",
        "One checkpoint, three scene sizes",
        "A single sparse model trained on 3-object scenes, stepping 3-, 5- and 8-object "
        "scenes with no retraining. The dense monolith cannot be executed off-count at all.",
    ),
]

PAGE_CSS = """
:root {
  --bg: #ffffff; --fg: #1b1b1f; --muted: #5c5c66; --line: #e0e0e6;
  --panel: #efece7; --accent: #1b9e77; --card: #fafafa;
}
@media (prefers-color-scheme: dark) {
  :root { --bg: #14141a; --fg: #ececf1; --muted: #a0a0ad; --line: #2c2c36;
          --panel: #24242e; --accent: #35c79b; --card: #1c1c24; }
}
:root[data-theme="dark"] {
  --bg: #14141a; --fg: #ececf1; --muted: #a0a0ad; --line: #2c2c36;
  --panel: #24242e; --accent: #35c79b; --card: #1c1c24;
}
:root[data-theme="light"] {
  --bg: #ffffff; --fg: #1b1b1f; --muted: #5c5c66; --line: #e0e0e6;
  --panel: #efece7; --accent: #1b9e77; --card: #fafafa;
}
body { margin: 0; padding: 2rem 1.25rem 4rem; background: var(--bg); color: var(--fg);
  font: 16px/1.6 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1180px; margin: 0 auto; }
h1 { font-size: 1.8rem; line-height: 1.25; margin: 0 0 .5rem; letter-spacing: -0.02em; }
h2 { font-size: 1.2rem; margin: 0 0 .35rem; letter-spacing: -0.01em; }
.lede { color: var(--muted); margin: 0 0 2.5rem; max-width: 62ch; }
section { border-top: 1px solid var(--line); padding-top: 1.75rem; margin-bottom: 3rem; }
.blurb { color: var(--muted); font-size: .93rem; margin: 0 0 1.1rem; max-width: 74ch; }
.panels { display: flex; flex-wrap: wrap; gap: .9rem; }
.panel { flex: 1 1 210px; min-width: 190px; }
.panel h3 { font-size: .82rem; text-transform: uppercase; letter-spacing: .06em;
  margin: 0 0 .4rem; color: var(--muted); font-weight: 600; }
canvas { width: 100%; height: auto; aspect-ratio: 1; background: var(--panel);
  border: 1px solid var(--line); border-radius: 8px; display: block; }
.readout { font: 12px ui-monospace, SFMono-Regular, Menlo, monospace; color: var(--muted);
  margin-top: .4rem; min-height: 1.2em; }
.controls { display: flex; align-items: center; gap: .85rem; margin-top: 1.2rem;
  background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: .7rem .9rem; }
button { font: inherit; font-size: .9rem; padding: .35rem .95rem; cursor: pointer;
  border-radius: 6px; border: 1px solid var(--line); background: var(--bg); color: var(--fg); }
button:hover { border-color: var(--accent); color: var(--accent); }
input[type=range] { flex: 1; accent-color: var(--accent); min-width: 120px; }
.frame-label { font: 13px ui-monospace, SFMono-Regular, Menlo, monospace; color: var(--muted);
  min-width: 7.5em; text-align: right; }
.meta { font-size: .78rem; color: var(--muted); margin-top: .8rem; }
.meta code { background: var(--card); padding: .1em .35em; border-radius: 4px; }
"""

PAGE_JS = """
const TRUTH_STROKE = '#8c8c8c';

function poseCorners(pose, half) {
  const [x, y, t] = pose, c = Math.cos(t), s = Math.sin(t);
  return [[-half,-half],[half,-half],[half,half],[-half,half]].map(([u,v]) =>
    [x + u*c - v*s, y + u*s + v*c]);
}

function makeView(canvas, viewHalf) {
  const size = canvas.width;
  return ([x, y]) => [ (x + viewHalf) / (2*viewHalf) * size,
                       size - (y + viewHalf) / (2*viewHalf) * size ];
}

function drawScene(canvas, demo, panel, frame) {
  const ctx = canvas.getContext('2d');
  const g = demo.meta.geometry, colors = demo.meta.object_colors;
  const viewHalf = demo.viewHalf;
  const toPx = makeView(canvas, viewHalf);
  const scale = canvas.width / (2 * viewHalf);
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const goal = demo.meta.goal_xy;
  if (goal) {
    const [gx, gy] = toPx(goal);
    ctx.beginPath(); ctx.arc(gx, gy, g.goal_radius * scale, 0, Math.PI*2);
    ctx.fillStyle = 'rgba(242, 204, 51, 0.35)'; ctx.fill();
    ctx.strokeStyle = 'rgba(242, 204, 51, 0.9)'; ctx.lineWidth = 1.5; ctx.stroke();
  }

  const truth = panel.truth || (demo.truthPanel && demo.truthPanel !== panel.name
    ? demo.panels[demo.truthPanel].poses : null);
  if (truth && panel.name !== demo.truthPanel) {
    ctx.setLineDash([5, 4]); ctx.strokeStyle = TRUTH_STROKE; ctx.lineWidth = 1.4;
    for (const pose of truth[frame]) {
      const pts = poseCorners(pose, g.box_half).map(toPx);
      ctx.beginPath(); ctx.moveTo(...pts[0]);
      pts.slice(1).forEach(p => ctx.lineTo(...p));
      ctx.closePath(); ctx.stroke();
    }
    ctx.setLineDash([]);
  }

  panel.poses[frame].forEach((pose, i) => {
    const pts = poseCorners(pose, g.box_half).map(toPx);
    ctx.beginPath(); ctx.moveTo(...pts[0]);
    pts.slice(1).forEach(p => ctx.lineTo(...p));
    ctx.closePath();
    ctx.fillStyle = colors[i % colors.length]; ctx.fill();
    ctx.strokeStyle = i === (demo.meta.target_object ?? -1) ? '#20202a' : '#ffffff';
    ctx.lineWidth = i === (demo.meta.target_object ?? -1) ? 2.2 : 1.1; ctx.stroke();
  });

  if (panel.pusher) {
    const [px, py] = toPx(panel.pusher[frame]);
    ctx.beginPath(); ctx.arc(px, py, g.pusher_radius * scale, 0, Math.PI*2);
    ctx.fillStyle = '#1a1a1a'; ctx.fill();
    ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 1.2; ctx.stroke();
  }
}

function initDemo(demo) {
  const root = document.getElementById(demo.id);
  const canvases = [...root.querySelectorAll('canvas')];
  const readouts = [...root.querySelectorAll('.readout')];
  const slider = root.querySelector('input[type=range]');
  const label = root.querySelector('.frame-label');
  const playButton = root.querySelector('button');
  let frame = 0, timer = null;

  // Fit every panel to the widest extent any panel reaches, so a drifting
  // prediction is never silently cropped out of its own panel.
  let extent = Math.max(0.2, ...(demo.meta.goal_xy || [0]).map(Math.abs));
  for (const panel of demo.panelList)
    for (const step of panel.poses)
      for (const pose of step) extent = Math.max(extent, Math.abs(pose[0]), Math.abs(pose[1]));
  demo.viewHalf = Math.min(extent + 0.05, 0.7);

  function render() {
    demo.panelList.forEach((panel, i) => {
      drawScene(canvases[i], demo, panel, frame);
      const series = panel.error || panel.distance;
      readouts[i].textContent = series
        ? `${panel.errorLabel} ${series[frame].toFixed(3)}` : '';
    });
    slider.value = frame;
    label.textContent = `step ${frame} / ${demo.frames - 1}`;
  }

  slider.max = demo.frames - 1;
  slider.addEventListener('input', () => { frame = +slider.value; render(); });
  playButton.addEventListener('click', () => {
    if (timer) { clearInterval(timer); timer = null; playButton.textContent = 'Play'; return; }
    playButton.textContent = 'Pause';
    timer = setInterval(() => {
      frame = (frame + 1) % demo.frames;
      render();
    }, 1000 / demo.fps);
  });
  render();
}

DEMOS.forEach(initDemo);
"""


def build_demo_payload(path: Path, section_id: str, fps: int) -> dict:
    """Reshape one demo JSON into what the page's renderer expects."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    panels = raw["panels"]
    frames = min(len(series["poses"]) for series in panels.values())

    panel_list = []
    for name, series in panels.items():
        panel_list.append(
            {
                "name": name,
                "poses": series["poses"],
                "pusher": series.get("pusher"),
                "truth": series.get("truth"),
                "error": series.get("error"),
                "distance": series.get("distance"),
                "errorLabel": "target-goal" if "distance" in series else "error",
            }
        )

    return {
        "id": section_id,
        "meta": raw["meta"],
        "panels": panels,
        "panelList": panel_list,
        "truthPanel": "truth" if "truth" in panels else None,
        "frames": frames,
        "fps": fps,
    }


PANEL_LABELS = {
    "truth": "Ground truth",
    "sparse": "Sparse (gate + residual)",
    "dense": "Dense (monolithic MLP)",
    "no_op": "No-op reference",
    "oracle": "Oracle (true simulator)",
    "scripted": "Scripted expert",
}


def panel_label(name: str) -> str:
    return PANEL_LABELS.get(name, name.replace("obj", "-object scene"))


def main() -> None:
    args = parse_args()
    sections: list[str] = []
    payloads: list[dict] = []

    for index, (stem, title, blurb) in enumerate(DEMO_SECTIONS):
        path = args.demo_dir / f"{stem}.json"
        if not path.exists():
            print(f"skipping '{stem}': {path} not found (run the demo to include it)")
            continue
        section_id = f"demo-{index}"
        payload = build_demo_payload(path, section_id, args.fps)
        payloads.append(payload)

        panels_html = "\n".join(
            f'        <div class="panel"><h3>{panel_label(panel["name"])}</h3>'
            f'<canvas width="420" height="420"></canvas>'
            f'<div class="readout"></div></div>'
            for panel in payload["panelList"]
        )
        source = ", ".join(
            f"<code>{Path(str(payload['meta'][key])).name}</code>"
            for key in ("sparse_checkpoint", "dense_checkpoint")
            if key in payload["meta"]
        )
        sections.append(
            f"""    <section id="{section_id}">
      <h2>{title}</h2>
      <p class="blurb">{blurb}</p>
      <div class="panels">
{panels_html}
      </div>
      <div class="controls">
        <button type="button">Play</button>
        <input type="range" min="0" value="0" step="1" aria-label="rollout step">
        <span class="frame-label"></span>
      </div>
      <p class="meta">Checkpoints: {source or "n/a"}</p>
    </section>"""
        )

    if not payloads:
        raise SystemExit(f"No demo JSON files found in {args.demo_dir}. Run the demos first.")

    html = f"""<style>{PAGE_CSS}</style>
<main>
  <h1>{args.title}</h1>
  <p class="lede">{args.lede}</p>
{chr(10).join(sections)}
</main>
<script>
const DEMOS = {json.dumps(payloads, separators=(",", ":"))};
{PAGE_JS}
</script>
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    size_kb = args.output.stat().st_size / 1024
    print(f"wrote {args.output} ({size_kb:.0f} KB, {len(payloads)} demos)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the self-contained interactive demo page.")
    parser.add_argument("--demo-dir", type=Path, default=Path("experiments/runs/demos"))
    parser.add_argument("--output", type=Path, default=Path("experiments/runs/demos/interactive.html"))
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--title", type=str, default="Sparse world models: interactive comparisons")
    parser.add_argument(
        "--lede",
        type=str,
        default=(
            "Scrub any rollout to any step and compare the models side by side. Every panel "
            "replays the exact trajectory used to render the corresponding figure."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
