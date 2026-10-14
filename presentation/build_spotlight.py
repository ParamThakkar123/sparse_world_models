"""Build the 2-minute spotlight deck for WORLDS @ IROS 2026.

A cut of the full talk (build_deck.py): same design, same numbers (all from the
camera-ready), six slides timed to about two minutes. Slide 3 embeds a rollout clip
(assets/rollout_3obj.mp4, rendered by experiments/demos/demo_rollout_drift.py) that
autoplays and loops. The spoken script lives in
the notes pane and is also written to spotlight_script.md.

    python presentation/build_spotlight.py            # writes presentation/WORLDS_IROS2026_spotlight.pptx
    python presentation/build_spotlight.py --render   # also exports PDF + PNG previews (needs PowerPoint)
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml import parse_xml
from pptx.oxml.ns import nsdecls, qn
from pptx.util import Inches

from build_deck import (DENSE, MARGIN, MUTED, OUT_DIR, PANEL, REPO_URL, SLIDE_H, SLIDE_W,
                        SPARSE, SPARSE_SOFT, Deck, bullets, color_series, draw_architecture,
                        draw_table_scene, make_qr, notes, rect, stat, style_chart, text)

OUT = OUT_DIR / "WORLDS_IROS2026_spotlight.pptx"
ASSETS = Path(__file__).resolve().parent / "assets"
VIDEO = ASSETS / "rollout_3obj.mp4"
VIDEO_POSTER = ASSETS / "rollout_3obj_poster.png"
VIDEO_MS = 10400  # clip length; the play command's duration needs it

# (slide title, seconds) for the script header; the sum is the talk length.
TIMING = [("Title", 10), ("Idea", 25), ("See it", 15), ("Results", 30), ("Honest reading", 25),
          ("Takeaway", 15)]

AUTOPLAY_TIMING = """
<p:timing {ns}><p:tnLst><p:par>
 <p:cTn id="1" dur="indefinite" restart="never" nodeType="tmRoot"><p:childTnLst>
  <p:seq concurrent="1" nextAc="seek">
   <p:cTn id="2" dur="indefinite" nodeType="mainSeq"><p:childTnLst>
    <p:par><p:cTn id="3" fill="hold"><p:stCondLst>
      <p:cond delay="indefinite"/><p:cond evt="onBegin" delay="0"><p:tn val="2"/></p:cond>
     </p:stCondLst><p:childTnLst>
     <p:par><p:cTn id="4" fill="hold"><p:stCondLst><p:cond delay="0"/></p:stCondLst><p:childTnLst>
      <p:par><p:cTn id="5" presetID="1" presetClass="mediacall" presetSubtype="0" fill="hold"
                    nodeType="afterEffect"><p:stCondLst><p:cond delay="0"/></p:stCondLst><p:childTnLst>
       <p:cmd type="call" cmd="playFrom(0.0)"><p:cBhvr><p:cTn id="6" dur="{dur}" fill="hold"/>
        <p:tgtEl><p:spTgt spid="{spid}"/></p:tgtEl></p:cBhvr></p:cmd>
      </p:childTnLst></p:cTn></p:par>
     </p:childTnLst></p:cTn></p:par>
    </p:childTnLst></p:cTn></p:par>
   </p:childTnLst></p:cTn>
   <p:prevCondLst><p:cond evt="onPrev" delay="0"><p:tgtEl><p:sldTgt/></p:tgtEl></p:cond></p:prevCondLst>
   <p:nextCondLst><p:cond evt="onNext" delay="0"><p:tgtEl><p:sldTgt/></p:tgtEl></p:cond></p:nextCondLst>
  </p:seq>
  <p:video><p:cMediaNode vol="80000" mute="1">
   <p:cTn id="7" repeatCount="indefinite" fill="hold" display="0"><p:stCondLst><p:cond delay="indefinite"/></p:stCondLst></p:cTn>
   <p:tgtEl><p:spTgt spid="{spid}"/></p:tgtEl>
  </p:cMediaNode></p:video>
 </p:childTnLst></p:cTn>
</p:par></p:tnLst></p:timing>"""


def autoplay_loop(slide, movie, dur_ms):
    """Replace python-pptx's click-to-play timing with play-on-entry, loop until the slide ends."""
    timing = parse_xml(AUTOPLAY_TIMING.format(ns=nsdecls("p"), dur=dur_ms, spid=movie.shape_id))
    old = slide._element.find(qn("p:timing"))
    if old is not None:
        old.getparent().replace(old, timing)
    else:
        slide._element.insert_element_before(timing, "p:extLst")


def build(qr_path: Path | None):
    d = Deck()

    # 1. Title (~10 s) -------------------------------------------------------------
    s = d.slide(footer=False)
    rect(s, 0, 0, Inches(0.28), SLIDE_H, fill=SPARSE)
    text(s, Inches(0.9), Inches(0.75), Inches(8), Inches(0.35),
         "WORLDS WORKSHOP  ·  IEEE/RSJ IROS 2026  ·  SPOTLIGHT", size=13, bold=True, color=SPARSE)
    text(s, Inches(0.9), Inches(1.35), Inches(7.6), Inches(2.6),
         [[("Modeling What Changes", {"size": 44, "bold": True})],
          [("Sparse, Residual World Models for Object-Centric Manipulation",
            {"size": 24, "color": MUTED})]], spacing=8)
    text(s, Inches(0.9), Inches(4.35), Inches(7.6), Inches(1.9),
         [[("Param Thakkar", {"bold": True}), ("   Veermata Jijabai Technological Institute", {"color": MUTED})],
          [("Parsika Paresh Shah", {"bold": True}), ("   Arizona State University", {"color": MUTED})],
          [("Manisha Sushant Gote", {"bold": True}), ("   ZuiGO Private Limited", {"color": MUTED})]],
         size=17, spacing=6)
    draw_table_scene(s, Inches(8.85), Inches(1.5), Inches(4.0), show_labels=False)
    notes(s, """
[~10 s] Hi, I'm Param. Our paper is about one simple idea for world models of manipulation
scenes: model what changes, and copy everything else.
""")

    # 2. Observation + model (~35 s) ------------------------------------------------
    s = d.slide("One push moves one object. So gate what changes, copy the rest.",
                eyebrow="The idea")
    text(s, MARGIN, Inches(1.55), Inches(12), Inches(0.9),
         [[("Dense world models regress ", {}), ("every", {"italic": True}),
           (" pose, every step, and so inject error into objects that never moved.", {})]],
         size=19, color=MUTED)
    draw_architecture(s, MARGIN, Inches(2.35), Inches(12), Inches(3.2))
    bullets(s, MARGIN, Inches(5.7), Inches(12), Inches(1.2), [
        [("Hard gate at inference: ", {"bold": True}),
         ("un-gated objects get exactly zero update", {})],
        [("Shared per-object heads, nothing sized to N: ", {"bold": True}),
         ("train on 3 objects, run on 8 with no retraining", {})],
    ], size=16, gap=6)
    notes(s, """
[~25 s] When a robot nudges one block, one pose changes and everything else stays put. Yet
the standard recipe re-predicts every pose at every step, so it perturbs objects that never
moved.

Our model has a per-object gate that decides whether each object changes, and a delta head
whose residual is applied only where the gate fires. Gate off: the object is copied exactly.
The heads are shared, so one model runs at any object count.
""")

    # 3. See it (~15 s): the problem, shown -----------------------------------------
    s = d.slide("Dense moves everything. Sparse moves only what was pushed.", eyebrow="See it")
    vw = SLIDE_W - 2 * MARGIN
    vh = int(vw * 480 / 1600)
    movie = s.shapes.add_movie(str(VIDEO), MARGIN, Inches(1.7), vw, vh,
                               poster_frame_image=str(VIDEO_POSTER), mime_type="video/mp4")
    autoplay_loop(s, movie, VIDEO_MS)
    text(s, MARGIN, Inches(1.7) + vh + Inches(0.2), vw, Inches(0.9),
         [[("Held-out 3-object push, 20-step closed-loop rollout, half speed. Dashed outlines are the "
            "ground truth, so the gap is the error. Final error: sparse 0.26, dense 1.34, no-op 0.35.",
            {})],
          [("Window picked by a fixed rule (median sparse error among active pushes), not by hand. "
            "Averaged over all rollouts, no-op is still lower overall (Table III).", {"italic": True})]],
         size=12, color=MUTED, spacing=4)
    notes(s, """
[~15 s] Here it is running on a held-out push. Dashed outlines are the truth. The dense model,
third panel, scatters every box, even the ones the pusher never touched. Ours moves the boxes
that were pushed and leaves the green one exactly where it was.
""")

    # 4. Results (~30 s) ------------------------------------------------------------
    s = d.slide("Far less drift than dense, at a tenth of the parameters", eyebrow="Results")
    cd = CategoryChartData()
    cd.categories = ["3 objects", "5 objects", "8 objects"]
    cd.add_series("Sparse (ours)", (0.373, 0.293, 0.182))
    cd.add_series("Dense MLP", (1.241, 1.097, 1.170))
    gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, MARGIN, Inches(1.7), Inches(7.4),
                            Inches(4.5), cd)
    style_chart(gf.chart, ymax=1.4, fmt="0.00", axis_fmt="0.0")
    color_series(gf.chart, [SPARSE, DENSE])
    text(s, MARGIN, Inches(6.2), Inches(7.4), Inches(0.7),
         "Per-object pose L2 after a 20-step closed-loop rollout on held-out trajectories, lower is "
         "better (Table III, mean of 3 seeds). No-op, which predicts no motion, is lower still: "
         "0.277 / 0.180 / 0.108.",
         size=12, color=MUTED)
    sx = Inches(8.6)
    stat(s, sx, Inches(1.7), Inches(4), "2.5 – 4.6×", "lower one-step error than dense, with 8.6–11.1× "
                                                     "fewer parameters (Table I)", SPARSE)
    stat(s, sx, Inches(3.25), Inches(4), "0.211 → 0.001", "error on objects that did not move, "
                                                         "dense → sparse (3 objects)", SPARSE, vsize=36)
    stat(s, sx, Inches(4.8), Inches(4), "99.4%", "of change-detection F1 kept on an unseen object "
                                                "count, no retraining", SPARSE)
    notes(s, """
[~30 s] Across all held-out rollouts, after 20 steps the dense model's error is three to six
times higher than ours, with about a tenth of the parameters. The reason is the objects that
did not move: their error drops from 0.21 to almost zero. And a model trained on one object
count keeps 99 percent of its detection quality on another.
""")

    # 5. Honest reading (~25 s) ------------------------------------------------------
    s = d.slide("What the headline is, and what it is not", eyebrow="Reading it honestly")
    cw = Inches(5.85)
    rect(s, MARGIN, Inches(1.75), cw, Inches(3.4), fill=SPARSE_SOFT, shape=MSO_SHAPE.ROUNDED_RECTANGLE,
         radius=0.05)
    rect(s, MARGIN + cw + Inches(0.3), Inches(1.75), cw, Inches(3.4), fill=PANEL,
         shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.05)
    text(s, MARGIN + Inches(0.35), Inches(1.95), cw - Inches(0.7), Inches(0.5), "IT IS", size=16,
         bold=True, color=SPARSE)
    bullets(s, MARGIN + Inches(0.35), Inches(2.55), cw - Inches(0.7), Inches(3.6), [
        [("A change detector ", {"bold": True}),
         ("(F1 0.80–0.87) that stops error injection into the static majority", {})],
        [("Clearly better than dense ", {"bold": True}),
         ("one-step, in rollout, and in planning (0.23 vs 0.00 success)", {})],
    ], size=21, gap=18)
    x2 = MARGIN + cw + Inches(0.65)
    text(s, x2, Inches(1.95), cw - Inches(0.7), Inches(0.5), "IT IS NOT", size=16, bold=True, color=DENSE)
    bullets(s, x2, Inches(2.55), cw - Inches(0.7), Inches(3.6), [
        [("An accurate predictor of the movers: ", {"bold": True}),
         ("an oracle gate barely helps; they sit at no-op quality", {})],
        [("Better than predicting no motion, overall: ", {"bold": True}),
         ("it only matches it. Planning is not clearly beyond random either", {})],
    ], size=21, gap=18)
    rect(s, MARGIN, Inches(5.5), Inches(12), Inches(1.0), fill=PANEL, shape=MSO_SHAPE.ROUNDED_RECTANGLE,
         radius=0.12)
    text(s, MARGIN, Inches(5.5), Inches(12), Inches(1.0),
         [[("Lesson: ", {"bold": True}),
           ("always report the no-op baseline, and split error into changed vs. unchanged objects.", {})]],
         size=19, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    notes(s, """
[~25 s] Now the honest part. The win is change detection: it avoids corrupting the static
majority. It is not accurate dynamics on the movers. Even with a perfect gate, error on
moving objects sits at the do-nothing baseline, and overall the model only matches no-op.
So our advice: always report no-op, and split error into changed and unchanged objects.
""")

    # 6. Takeaway + poster (~15 s) ---------------------------------------------------
    s = d.slide("Model what changes. The delta head is the lever.", eyebrow="Takeaway")
    bullets(s, MARGIN, Inches(1.9), Inches(8.2), Inches(3.4), [
        [("A per-object gate plus a masked residual ", {"bold": True}),
         ("is a one-line bias that stops a world model corrupting the static scene", {})],
        [("Detection works; ", {"bold": True}),
         ("regressing contact deltas does not yet. That is the next thing to fix", {})],
        [("Everything is open: ", {"bold": True}),
         ("code, data generators, checkpoints, and a live in-browser demo", {})],
    ], size=20, gap=18)
    rect(s, MARGIN, Inches(5.45), Inches(8.2), Inches(1.1), fill=SPARSE_SOFT,
         shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.12)
    text(s, MARGIN, Inches(5.45), Inches(8.2), Inches(1.1), "Come find us at the poster.  Thank you!",
         size=24, bold=True, color=SPARSE, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    px = Inches(9.4)
    rect(s, px, Inches(1.8), Inches(3.25), Inches(4.75), fill=PANEL, shape=MSO_SHAPE.ROUNDED_RECTANGLE,
         radius=0.05)
    if qr_path and qr_path.exists():
        s.shapes.add_picture(str(qr_path), px + Inches(0.45), Inches(2.05), width=Inches(2.35))
    text(s, px + Inches(0.2), Inches(4.6), Inches(2.85), Inches(1.8),
         [[("Code + demo", {"bold": True})],
          [("github.com/ParamThakkar123/\nsparse_world_models", {"color": MUTED, "size": 13})]],
         size=15, align=PP_ALIGN.CENTER, spacing=6)
    notes(s, f"""
[~15 s] So: model what changes. Detection works; predicting the push itself is the next thing
to fix, and the delta head is the lever. Code, checkpoints and a live
in-browser demo are at the QR code. Come find us at the poster. Thank you!

(Repository: {REPO_URL})
""")

    return d


def export_script(deck: Deck, path: Path):
    total = sum(t for _, t in TIMING)
    out = ["# Spotlight script: Modeling What Changes (WORLDS @ IROS 2026)", "",
           f"Generated from the notes pane of `{OUT.name}` by `build_spotlight.py`. "
           f"{len(TIMING)} slides, about {total} seconds.", ""]
    for i, (s, (name, secs)) in enumerate(zip(deck.prs.slides, TIMING), 1):
        out += [f"## {i}. {name} (~{secs} s)", "", s.notes_slide.notes_text_frame.text.strip(), ""]
    path.write_text("\n".join(out), encoding="utf-8")


def render(pptx_path: Path):
    import win32com.client  # type: ignore

    pdf = pptx_path.with_suffix(".pdf")
    app = win32com.client.Dispatch("PowerPoint.Application")
    pres = app.Presentations.Open(str(pptx_path), WithWindow=False)
    pres.SaveAs(str(pdf), 32)  # ppSaveAsPDF
    pres.Close()
    app.Quit()
    png_dir = pptx_path.parent / "preview_spotlight"
    png_dir.mkdir(exist_ok=True)
    subprocess.run(["pdftoppm", "-png", "-r", "80", str(pdf), str(png_dir / "slide")], check=True)
    return pdf, png_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", action="store_true", help="export PDF and PNG previews (PowerPoint + poppler)")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    qr = make_qr(args.out.parent / "repo_qr.png")
    deck = build(qr)
    deck.save(args.out)
    export_script(deck, args.out.parent / "spotlight_script.md")
    print(f"wrote {args.out} ({deck.n} slides) and spotlight_script.md")
    if args.render:
        pdf, png_dir = render(args.out)
        print(f"rendered {pdf} and {png_dir}/")
