"""Build the WORLDS @ IROS 2026 poster for "Modeling What Changes".

IROS 2026 boards have a 74" x 53" usable area and must be landscape (A0 portrait does not
fit). The poster is designed for 72" x 48" (3:2). PowerPoint caps a slide at 56", so the
file is laid out at half scale, 36" x 24": print it at 200% (or "fit to 72 x 48 in"). All
point sizes below are half-scale; the printed size is double (body 18 pt -> 36 pt).

Numbers are the camera-ready's, as in build_deck.py. Charts are native and vector; the two
paper figures are the PNGs from IEEE_Conference_Template/figures.

    python presentation/build_poster.py            # writes presentation/WORLDS_IROS2026_poster.pptx
    python presentation/build_poster.py --render   # also exports a print PDF + PNG preview (needs PowerPoint)
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

from build_deck import (DENSE, DENSE_SOFT, FIG, GREY_BOX, INK, LINE, MUTED, NOOP, OUT_DIR, PANEL,
                        REPO_URL, SPARSE, SPARSE_SOFT, WHITE, arrow, bullets, color_series,
                        draw_table_scene, make_qr, rect, style_chart, table, text)

OUT = OUT_DIR / "WORLDS_IROS2026_poster.pptx"

W, H = Inches(36), Inches(24)
M = Inches(0.8)                       # outer margin
GAP = Inches(0.6)                     # column gutter
NCOL = 4
COL_W = (W - 2 * M - (NCOL - 1) * GAP) // NCOL
COL_TOP = Inches(5.35)

BODY = 18
SMALL = 14
HEAD = 30


def col_x(i):
    return M + i * (COL_W + GAP)


def section(s, i, y, num, title):
    """Numbered green disc plus heading: the poster's one repeated motif. Returns next y."""
    x = col_x(i)
    d = Inches(0.62)
    rect(s, x, y, d, d, fill=SPARSE, shape=MSO_SHAPE.OVAL)
    text(s, x, y, d, d, str(num), size=22, bold=True, color=WHITE, align=PP_ALIGN.CENTER,
         anchor=MSO_ANCHOR.MIDDLE)
    text(s, x + Inches(0.85), y - Inches(0.02), COL_W - Inches(0.85), Inches(0.7), title, size=HEAD,
         bold=True, anchor=MSO_ANCHOR.MIDDLE)
    return y + Inches(0.95)


def big_stat(s, x, y, w, value, label, color=SPARSE, vsize=40):
    text(s, x, y, w, Inches(0.75), value, size=vsize, bold=True, color=color)
    text(s, x, y + Inches(0.78), w, Inches(0.9), label, size=SMALL, color=MUTED)


def draw_model(s, x, y, w):
    """Vertical redraw of the Fig. 1 block diagram to fit one column. Returns bottom y."""
    cx = x + w // 2
    bh = Inches(1.05)

    def block(bx, by, bw, label, sub, fill, line_c, color=INK):
        rect(s, bx, by, bw, bh, fill=fill, line=line_c, line_w=1.5,
             shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.18)
        text(s, bx, by, bw, bh, [[(label, {"bold": True, "color": color})],
                                 [(sub, {"size": SMALL, "color": MUTED})]],
             size=19, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)

    fw = Inches(4.2)
    block(cx - fw // 2, y, fw, "per-object features", "φᵢ(sₜ, aₜ): pose, velocity, goal, pusher, action",
          PANEL, LINE)
    bw = Inches(3.6)
    y2 = y + Inches(1.75)
    gx, dx = x + Inches(0.1), x + w - bw - Inches(0.1)
    block(gx, y2, bw, "gate net", "gᵢ ∈ {0, 1}, straight-through", SPARSE_SOFT, SPARSE, SPARSE)
    block(dx, y2, bw, "delta head", "Δᵢ ∈ ℝ³ pose residual", WHITE, INK)
    arrow(s, cx - Inches(0.4), y + bh, gx + bw // 2, y2, width=2)
    arrow(s, cx + Inches(0.4), y + bh, dx + bw // 2, y2, width=2)
    my = y2 + bh + Inches(0.55)
    d = Inches(0.7)
    rect(s, cx - d // 2, my, d, d, fill=WHITE, line=INK, line_w=1.5, shape=MSO_SHAPE.OVAL)
    text(s, cx - d // 2, my, d, d, "×", size=26, bold=True, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    arrow(s, gx + bw // 2, y2 + bh, cx - Inches(0.3), my + Inches(0.15), width=2)
    arrow(s, dx + bw // 2, y2 + bh, cx + Inches(0.3), my + Inches(0.15), width=2)
    oy = my + d + Inches(0.5)
    ow = Inches(5.6)
    arrow(s, cx, my + d, cx, oy, width=2)
    block(cx - ow // 2, oy, ow, "p̂ᵢᵗ⁺¹ = pᵢᵗ + gᵢ · Δᵢ", "masked residual added to the current pose",
          SPARSE_SOFT, SPARSE)
    text(s, x, oy + bh + Inches(0.1), w, Inches(0.5), "gate off  ⇒  object copied verbatim, exactly",
         size=17, bold=True, color=SPARSE, align=PP_ALIGN.CENTER)
    text(s, x, oy + bh + Inches(0.55), w, Inches(0.4), "shared weights, applied to every object", size=SMALL,
         italic=True, color=MUTED, align=PP_ALIGN.CENTER)
    return oy + bh + Inches(1.1)


def build(qr_path: Path | None):
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    s = prs.slides.add_slide(prs.slide_layouts[6])

    # ---- header ---------------------------------------------------------------------
    text(s, M, Inches(0.7), Inches(24), Inches(0.5),
         "WORLDS WORKSHOP  ·  IEEE/RSJ IROS 2026  ·  PITTSBURGH  ·  SPOTLIGHT 1, PAPER 7",
         size=20, bold=True, color=SPARSE)
    text(s, M, Inches(1.2), Inches(27), Inches(2.2),
         [[("Modeling What Changes", {"size": 80, "bold": True})],
          [("Sparse, Residual World Models for Object-Centric Manipulation",
            {"size": 40, "color": MUTED})]], spacing=2)
    text(s, M, Inches(3.45), Inches(27), Inches(0.6),
         [[("Param Thakkar", {"bold": True}), ("  Veermata Jijabai Technological Institute      ", {"color": MUTED}),
           ("Parsika Paresh Shah", {"bold": True}), ("  Arizona State University      ", {"color": MUTED}),
           ("Manisha Sushant Gote", {"bold": True}), ("  ZuiGO Private Limited", {"color": MUTED})]],
         size=22)
    # QR panel, top right
    qx = W - M - Inches(6.4)
    rect(s, qx, Inches(0.7), Inches(6.4), Inches(3.4), fill=PANEL, shape=MSO_SHAPE.ROUNDED_RECTANGLE,
         radius=0.08)
    if qr_path and qr_path.exists():
        s.shapes.add_picture(str(qr_path), qx + Inches(0.3), Inches(1.0), width=Inches(2.8))
    text(s, qx + Inches(3.35), Inches(1.0), Inches(2.9), Inches(2.8),
         [[("Code, data, checkpoints", {"bold": True})],
          [("and a live in-browser demo running the real models", {"color": MUTED, "size": 16})],
          [("github.com/ParamThakkar123/", {"size": 15, "color": SPARSE, "bold": True})],
          [("sparse_world_models", {"size": 15, "color": SPARSE, "bold": True})]],
         size=19, spacing=8, anchor=MSO_ANCHOR.MIDDLE)

    # ---- one-sentence message --------------------------------------------------------
    rect(s, M, Inches(4.3), W - 2 * M, Inches(0.8), fill=SPARSE_SOFT, shape=MSO_SHAPE.ROUNDED_RECTANGLE,
         radius=0.3)
    text(s, M, Inches(4.3), W - 2 * M, Inches(0.8),
         [[("A push moves one object; the rest of the scene is identical.  ", {"bold": True}),
           ("A per-object gate that copies unchanged objects exactly stops a world model from corrupting "
            "the static scene, at a tenth of the parameters.", {})]],
         size=24, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)

    # ================= column 1: problem and model =====================================
    x, y = col_x(0), COL_TOP
    y = section(s, 0, y, 1, "Scenes are sparse in change")
    sc = Inches(3.5)
    draw_table_scene(s, x, y, sc, show_labels=False)
    text(s, x + sc + Inches(0.35), y + Inches(0.1), COL_W - sc - Inches(0.35), sc,
         [[("One push. One object moves.", {"bold": True, "size": 22})],
          [("Every other object is exactly where it was.", {"color": MUTED})],
          [("Dense world models still regress ", {}), ("every", {"italic": True}),
           (" pose, every step.", {})]],
         size=BODY, spacing=10)
    y += sc + Inches(0.3)
    bullets(s, x, y, COL_W, Inches(2.6), [
        [("Capacity ", {"bold": True}), ("is spent re-producing the static majority", {})],
        [("Error injection: ", {"bold": True, "color": DENSE}),
         ("a regressor perturbs objects it should copy; in rollout the error piles up on objects "
          "that never moved", {})],
        [("No handle on “what”: ", {"bold": True}),
         ("no explicit belief about which objects will change", {})],
    ], size=BODY, gap=8)
    y += Inches(2.3)

    y = section(s, 0, y, 2, "Gate what changes, copy the rest")
    y = draw_model(s, x, y, COL_W)
    rect(s, x, y, COL_W, Inches(0.95), fill=PANEL, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.15)
    text(s, x, y, COL_W, Inches(0.95),
         [[("ℒ = BCE(g, m*)", {"bold": True, "color": SPARSE}),
           ("  +  λΔ · m* · ‖Δ − Δ*‖²  +  λs · ḡ", {})]],
         size=21, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    y += Inches(1.15)
    bullets(s, x, y, COL_W, Inches(2.2), [
        [("Hard at inference: ", {"bold": True}), ("un-gated objects get exactly zero update", {})],
        [("Count-agnostic: ", {"bold": True}),
         ("shared heads, nothing sized to N; train on 3 objects, run on 8", {})],
        [("Tiny: ", {"bold": True}), ("2-layer width-128 heads, under 0.1M parameters", {})],
    ], size=BODY, gap=6)

    # ================= column 2: setup and prediction =================================
    x, y = col_x(1), COL_TOP
    y = section(s, 1, y, 3, "Controlled tabletop pushing")
    bullets(s, x, y, COL_W, Inches(2.4), [
        [("MuJoCo tabletop, ", {"bold": True}),
         ("N ∈ {3, 5, 8} boxes, scripted pusher; structured state, not pixels", {})],
        [("Baselines: ", {"bold": True}), ("Dense MLP", {"bold": True, "color": DENSE}),
         (" regresses all poses; ", {}), ("No-op", {"bold": True, "color": NOOP}),
         (" predicts no change, the floor to beat", {})],
        [("Metric: ", {"bold": True}),
         ("per-object pose L2, split into changed vs. unchanged objects; 3 seeds", {})],
    ], size=BODY, gap=6)
    y += Inches(2.55)

    y = section(s, 1, y, 4, "Lower error, a tenth of the parameters")
    cd = CategoryChartData()
    cd.categories = ["3 objects", "5 objects", "8 objects"]
    cd.add_series("Sparse (ours)", (0.136, 0.101, 0.071))
    cd.add_series("Dense MLP", (0.347, 0.319, 0.329))
    cd.add_series("No-op", (0.128, 0.107, 0.081))
    ch_h = Inches(4.3)
    gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, x, y, COL_W, ch_h, cd)
    style_chart(gf.chart, size=16, ymax=0.4, fmt="0.000", axis_fmt="0.00", label_size=14)
    color_series(gf.chart, [SPARSE, DENSE, NOOP])
    y += ch_h
    text(s, x, y, COL_W, Inches(0.4), "Overall per-object pose L2, held-out, lower is better (Table I).",
         size=SMALL - 1, color=MUTED)
    y += Inches(0.55)
    sw = COL_W // 3
    big_stat(s, x, y, sw, "2.5–4.6×", "lower L2 than dense", vsize=34)
    big_stat(s, x + sw, y, sw, "8.6–11.1×", "fewer parameters", vsize=34)
    big_stat(s, x + 2 * sw, y, sw, "F1 0.80–0.87", "change detection; dense is degenerate", INK, vsize=30)
    y += Inches(1.75)
    pic = s.shapes.add_picture(str(FIG / "qualitative.png"), x, y, width=COL_W)
    pic.crop_top = 0.055
    pic.height = int(COL_W * 1100 / 1584 * (1 - 0.055))
    y += pic.height + Inches(0.05)
    text(s, x, y, COL_W, Inches(0.7),
         [[("Held-out scenes. ", {"bold": True}),
           ("Sparse moves only the pushed object; ", {"color": SPARSE, "bold": True}),
           ("dense hallucinates motion on all three.", {"color": DENSE, "bold": True})]],
         size=SMALL + 1)

    # ================= column 3: why it wins and world-model properties ===============
    x, y = col_x(2), COL_TOP
    y = section(s, 2, y, 5, "Where the win comes from")
    cd = CategoryChartData()
    cd.categories = ["Dense", "+ object-\ncentric", "+ residual", "+ gate\n(sparse)", "No-op"]
    cd.add_series("Overall L2", (0.367, 0.239, 0.221, 0.116, 0.128))
    ch_h = Inches(3.9)
    gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, x, y, COL_W, ch_h, cd)
    style_chart(gf.chart, legend=False, size=15, ymax=0.4, fmt="0.000", axis_fmt="0.00", label_size=14)
    ser = gf.chart.series[0]
    for i, c in enumerate([DENSE, GREY_BOX, GREY_BOX, SPARSE, NOOP]):
        pt = ser.points[i]
        pt.format.fill.solid()
        pt.format.fill.fore_color.rgb = c
        pt.format.line.fill.background()
    y += ch_h
    text(s, x, y, COL_W, Inches(0.4), "Capacity-matched ladder, one ingredient per rung (Table II, 3 objects).",
         size=SMALL - 1, color=MUTED)
    y += Inches(0.5)
    text(s, x, y, COL_W, Inches(1.1),
         [[("Unchanged-object L2:  ", {"bold": True}), ("0.211 → 0.111 → 0.086 → ", {}),
           ("0.001", {"bold": True, "color": SPARSE})],
          [("The gate is the largest single step, and the only one below no-op.", {"color": MUTED})]],
         size=BODY, spacing=4)
    y += Inches(1.3)
    text(s, x, y, COL_W, Inches(0.5), "Changed-object L2 with a perfect (oracle) gate", size=19, bold=True)
    y += Inches(0.55)
    cw = [COL_W * f // 100 for f in (16, 28, 28, 28)]
    y = table(s, x, y, cw, [
        ["N", "predicted gate", "oracle gate", "no-op"],
        ["3", "0.312", "0.314", "0.348"],
        ["5", "0.426", "0.432", "0.446"],
        ["8", "0.466", "0.474", "0.470"],
    ], size=17, row_h=Inches(0.46), colors={(0, 1): SPARSE, (0, 3): NOOP})
    y += Inches(0.2)
    text(s, x, y, COL_W, Inches(1.2),
         [[("A perfect gate barely helps. ", {"bold": True}),
           ("The win is detecting what changed and not corrupting the rest. On the movers "
            "themselves the model is at no-op quality: delta regression is the bottleneck.", {})]],
         size=BODY - 1)
    y += Inches(1.45)

    y = section(s, 2, y, 6, "A world model: rollout, transfer, data")
    pic = s.shapes.add_picture(str(FIG / "worldmodel.png"), x, y, width=COL_W)
    y += pic.height + Inches(0.15)
    bullets(s, x, y, COL_W, Inches(2.2), [
        [("(a) Rollout: ", {"bold": True}), ("3.4–6.4× less 20-step error than dense, still above no-op", {})],
        [("(b) Transfer: ", {"bold": True}),
         ("one checkpoint, any N; 99.4% of F1 kept off-count. Dense cannot run off-count", {})],
        [("(c) Data: ", {"bold": True}), ("~90% of full-data F1 from 25% of the data", {})],
    ], size=BODY - 1, gap=4)

    # ================= column 4: planning, honest reading, takeaways ==================
    x, y = col_x(3), COL_TOP
    y = section(s, 3, y, 7, "Planning: beats dense, not yet random")
    cd = CategoryChartData()
    cd.categories = ["Oracle simulator", "Scripted controller", "Sparse (contact feats)",
                     "Random actions", "Dense (diverse data)"]
    cd.add_series("Success rate", (1.00, 0.95, 0.23, 0.15, 0.00))
    ch_h = Inches(4.3)
    gf = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, x, y, COL_W, ch_h, cd)
    style_chart(gf.chart, legend=False, ymax=1.15, size=15, label_size=14)
    gf.chart.category_axis.reverse_order = True
    gf.chart.plots[0].gap_width = 45
    ser = gf.chart.series[0]
    for i, c in enumerate([GREY_BOX, GREY_BOX, SPARSE, NOOP, DENSE]):
        pt = ser.points[i]
        pt.format.fill.solid()
        pt.format.fill.fore_color.rgb = c
        pt.format.line.fill.background()
    y += ch_h
    text(s, x, y, COL_W, Inches(0.6), "Push-to-goal success, same CEM planner, only the model swapped; "
         "3 seeds × 20 episodes (Table IV).", size=SMALL - 1, color=MUTED)
    y += Inches(0.7)
    bullets(s, x, y, COL_W, Inches(2.3), [
        [("Round 1: ", {"bold": True, "color": DENSE}),
         ("both learned models score 0.00; the planner queries off-distribution states", {})],
        [("Round 2: ", {"bold": True, "color": SPARSE}),
         ("contact-aware features + diverse data: sparse 0.23 ± 0.06, dense 0.00 at every seed. "
          "Margin over random (0.15) is not significant", {})],
    ], size=BODY - 1, gap=6)
    y += Inches(2.0)

    y = section(s, 3, y, 8, "What it is, and is not")
    bw = (COL_W - Inches(0.3)) // 2
    bh = Inches(3.3)
    rect(s, x, y, bw, bh, fill=SPARSE_SOFT, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06)
    rect(s, x + bw + Inches(0.3), y, bw, bh, fill=DENSE_SOFT, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06)
    pad = Inches(0.25)
    text(s, x + pad, y + pad, bw - 2 * pad, bh - 2 * pad,
         [[("IT IS", {"bold": True, "color": SPARSE, "size": SMALL + 1})],
          [("a change detector that avoids error injection", {"bold": True})],
          [("far better than dense one-step, in rollout and in planning", {})],
          [("count-invariant, with an inspectable mask", {})]],
         size=BODY - 1, spacing=8)
    text(s, x + bw + Inches(0.3) + pad, y + pad, bw - 2 * pad, bh - 2 * pad,
         [[("IT IS NOT", {"bold": True, "color": DENSE, "size": SMALL + 1})],
          [("an accurate predictor of the movers", {"bold": True})],
          [("better than no-op in rollout", {})],
          [("significantly better than random at planning, or faster in wall-clock", {})]],
         size=BODY - 1, spacing=8)
    y += bh + Inches(0.35)

    y = section(s, 3, y, 9, "Takeaways")
    bullets(s, x, y, COL_W, Inches(3), [
        [("Model what changes: ", {"bold": True}), ("a gate + masked residual is a one-line bias", {})],
        [("Decompose the metric: ", {"bold": True}),
         ("always report no-op and the changed / unchanged split", {})],
        [("The delta head is the lever: ", {"bold": True}),
         ("detection is solved, contact-delta regression is not", {})],
    ], size=BODY, gap=6)

    return prs


def render(pptx_path: Path):
    import win32com.client  # type: ignore

    pdf = pptx_path.with_suffix(".pdf")
    app = win32com.client.Dispatch("PowerPoint.Application")
    pres = app.Presentations.Open(str(pptx_path), WithWindow=False)
    pres.SaveAs(str(pdf), 32)  # ppSaveAsPDF
    pres.Close()
    app.Quit()
    subprocess.run(["pdftoppm", "-png", "-r", "60", "-singlefile", str(pdf),
                    str(pptx_path.with_name("poster_preview"))], check=True)
    return pdf


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", action="store_true", help="export PDF and PNG preview (PowerPoint + poppler)")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    qr = make_qr(args.out.parent / "repo_qr.png")
    build(qr).save(args.out)
    print(f"wrote {args.out}")
    if args.render:
        print(f"rendered {render(args.out)} and poster_preview.png")
