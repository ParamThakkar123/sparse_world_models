"""Build the WORLDS @ IROS 2026 talk deck for "Modeling What Changes".

Every number on a slide is copied from IEEE_Conference_Template/conference_101719.tex
(the camera-ready); the table or appendix it comes from is named in the speaker notes.
Charts are native PowerPoint charts (editable), diagrams are native shapes, and the two
figures reused from the paper are the ones the paper itself reproduces from experiments/.

    python presentation/build_deck.py            # writes presentation/WORLDS_IROS2026_talk.pptx
    python presentation/build_deck.py --render   # also exports PDF + PNG contact sheet (needs PowerPoint)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "IEEE_Conference_Template" / "figures"
OUT_DIR = ROOT / "presentation"
OUT = OUT_DIR / "WORLDS_IROS2026_talk.pptx"
REPO_URL = "https://github.com/ParamThakkar123/sparse_world_models"
DEMO_URL = "https://paramthakkar123.github.io/WorldModelsWorkshop/"

# ---- design tokens ---------------------------------------------------------
INK = RGBColor(0x1C, 0x1C, 0x1E)
MUTED = RGBColor(0x6E, 0x6E, 0x73)
LINE = RGBColor(0xD9, 0xD9, 0xD6)
PANEL = RGBColor(0xF4, 0xF4, 0xF1)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
SPARSE = RGBColor(0x1B, 0x9E, 0x77)   # teal green, same hue as the paper figures
SPARSE_SOFT = RGBColor(0xDD, 0xF1, 0xEA)
DENSE = RGBColor(0xD9, 0x5F, 0x02)    # orange, same as the paper figures
DENSE_SOFT = RGBColor(0xFB, 0xE7, 0xD6)
NOOP = RGBColor(0x1F, 0x77, 0xB4)     # blue, same as the paper figures
NOOP_SOFT = RGBColor(0xDC, 0xEA, 0xF6)
GREY_BOX = RGBColor(0xB8, 0xB8, 0xB4)
FONT = "Calibri"

SLIDE_W, SLIDE_H = Inches(13.333), Inches(7.5)
MARGIN = Inches(0.7)
FOOTER = "Modeling What Changes  ·  WORLDS Workshop @ IEEE/RSJ IROS 2026"


# ---- primitives ------------------------------------------------------------
def _no_shadow(shape):
    shape.shadow.inherit = False


def rect(slide, x, y, w, h, fill=None, line=None, line_w=0.75, shape=MSO_SHAPE.RECTANGLE,
         dash=None, radius=None):
    s = slide.shapes.add_shape(shape, x, y, w, h)
    _no_shadow(s)
    if fill is None:
        s.fill.background()
    else:
        s.fill.solid()
        s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(line_w)
        if dash:
            s.line.dash_style = dash
    if radius is not None and shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = radius
    s.text_frame.text = ""
    return s


def text(slide, x, y, w, h, content, size=18, bold=False, color=INK, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, italic=False, font=FONT, spacing=None, margin=0):
    """content: str, or list of paragraphs; a paragraph is a str or a list of runs;
    a run is (text, {bold, color, size, italic})."""
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    m = Inches(margin)
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = m
    paras = content if isinstance(content, list) else [content]
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        if spacing is not None:
            p.space_after = Pt(spacing)
        runs = para if isinstance(para, list) else [(para, {})]
        for rt, opts in runs:
            r = p.add_run()
            r.text = rt
            f = r.font
            f.name = font
            f.size = Pt(opts.get("size", size))
            f.bold = opts.get("bold", bold)
            f.italic = opts.get("italic", italic)
            f.color.rgb = opts.get("color", color)
    return tb


def bullets(slide, x, y, w, h, items, size=18, color=INK, gap=10, marker="–"):
    """items: list of str or list-of-runs. A leading (text, {'bold': True}) run reads as a lead-in."""
    paras = []
    for it in items:
        runs = it if isinstance(it, list) else [(it, {})]
        paras.append([(f"{marker}  ", {"color": SPARSE, "bold": True})] + runs)
    return text(slide, x, y, w, h, paras, size=size, color=color, spacing=gap)


def arrow(slide, x1, y1, x2, y2, color=INK, width=1.5, head="triangle"):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, x1, y1, x2, y2)
    c.line.color.rgb = color
    c.line.width = Pt(width)
    ln = c.line._get_or_add_ln()
    tail = OxmlElement("a:tailEnd")
    tail.set("type", head)
    tail.set("w", "med")
    tail.set("len", "med")
    ln.append(tail)
    return c


def line(slide, x1, y1, x2, y2, color=LINE, width=1.0):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, x1, y1, x2, y2)
    c.line.color.rgb = color
    c.line.width = Pt(width)
    return c


def notes(slide, body):
    slide.notes_slide.notes_text_frame.text = body.strip()


# ---- chrome ----------------------------------------------------------------
class Deck:
    def __init__(self):
        self.prs = Presentation()
        self.prs.slide_width = SLIDE_W
        self.prs.slide_height = SLIDE_H
        self.blank = self.prs.slide_layouts[6]
        self.n = 0

    def slide(self, title=None, eyebrow=None, footer=True):
        s = self.prs.slides.add_slide(self.blank)
        self.n += 1
        if eyebrow:
            text(s, MARGIN, Inches(0.45), Inches(8), Inches(0.3), eyebrow.upper(), size=12,
                 bold=True, color=SPARSE)
        if title:
            text(s, MARGIN, Inches(0.7), Inches(12), Inches(0.8), title, size=30, bold=True)
        if footer:
            text(s, MARGIN, Inches(7.02), Inches(9), Inches(0.3), FOOTER, size=10, color=MUTED)
            text(s, Inches(12.0), Inches(7.02), Inches(0.65), Inches(0.3), str(self.n), size=10,
                 color=MUTED, align=PP_ALIGN.RIGHT)
        return s

    def save(self, path):
        self.prs.save(path)


# ---- reusable drawings -----------------------------------------------------
def draw_table_scene(slide, x, y, size, moved_color=SPARSE, show_labels=True):
    """Top-down tabletop: five boxes, the pusher nudges one; four stay exactly put."""
    rect(slide, x, y, size, size, fill=PANEL, line=LINE, shape=MSO_SHAPE.ROUNDED_RECTANGLE,
         radius=0.04)
    u = size / 10.0
    box = u * 1.15
    statics = [(1.2, 1.3), (6.8, 1.6), (2.0, 6.9), (7.2, 7.0)]
    for bx, by in statics:
        rect(slide, x + int(u * bx), y + int(u * by), int(box), int(box), fill=GREY_BOX)
    # moved box: ghost outline at old pose, solid at new pose, arrow between
    ox, oy = 4.0, 4.6
    nx, ny = 5.6, 3.6
    rect(slide, x + int(u * ox), y + int(u * oy), int(box), int(box), fill=None, line=MUTED,
         line_w=1.25, dash=MSO_LINE_DASH_STYLE.DASH)
    rect(slide, x + int(u * nx), y + int(u * ny), int(box), int(box), fill=moved_color)
    arrow(slide, x + int(u * (ox + 1.3)), y + int(u * (oy + 0.35)),
          x + int(u * (nx + 0.0)), y + int(u * (ny + 1.0)), color=moved_color, width=2.0)
    # pusher: dark disc approaching from lower left
    pr = u * 0.7
    rect(slide, x + int(u * 2.6), y + int(u * 5.7), int(pr), int(pr), fill=INK, shape=MSO_SHAPE.OVAL)
    arrow(slide, x + int(u * 3.3), y + int(u * 6.0), x + int(u * 3.9), y + int(u * 5.55), color=INK,
          width=1.5)
    if show_labels:
        text(slide, x + int(u * 5.3), y + int(u * 2.6), int(u * 4.5), int(u * 0.9), "moves", size=14,
             bold=True, color=moved_color)
        text(slide, x + int(u * 0.9), y + int(u * 2.6), int(u * 4), int(u * 0.9), "stays exactly put",
             size=14, color=MUTED)
        text(slide, x + int(u * 1.3), y + int(u * 5.3), int(u * 2.5), int(u * 0.9), "pusher", size=14,
             color=MUTED)


def draw_architecture(slide, x, y, w, h):
    """The Fig. 1 block diagram, redrawn with native shapes."""
    bh = Inches(0.9)
    bw = Inches(2.2)
    cy = y + h // 2

    def block(cx, cy_, label, sub=None, fill=WHITE, line_c=INK, width=bw, color=INK):
        s = rect(slide, cx - width // 2, cy_ - bh // 2, width, bh, fill=fill, line=line_c,
                 line_w=1.25, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.18)
        paras = [[(label, {"bold": True, "color": color})]]
        if sub:
            paras.append([(sub, {"size": 13, "color": MUTED})])
        text(slide, cx - width // 2, cy_ - bh // 2, width, bh, paras, size=17, align=PP_ALIGN.CENTER,
             anchor=MSO_ANCHOR.MIDDLE)
        return s

    x_in = x + Inches(1.3)
    x_net = x + Inches(4.6)
    x_mul = x + Inches(7.4)
    x_out = x + Inches(10.3)
    dy = Inches(1.05)

    block(x_in, cy, "per-object features", "φᵢ(sₜ, aₜ)", fill=PANEL, line_c=LINE, width=Inches(2.4))
    block(x_net, cy - dy, "gate net", "gᵢ ∈ {0, 1}  ·  straight-through", fill=SPARSE_SOFT,
          line_c=SPARSE, width=Inches(2.6), color=SPARSE)
    block(x_net, cy + dy, "delta head", "Δᵢ ∈ ℝ³  ·  pose residual", fill=WHITE, line_c=INK,
          width=Inches(2.6))
    mul = rect(slide, x_mul - Inches(0.3), cy - Inches(0.3), Inches(0.6), Inches(0.6), fill=WHITE,
               line=INK, line_w=1.25, shape=MSO_SHAPE.OVAL)
    text(slide, x_mul - Inches(0.3), cy - Inches(0.3), Inches(0.6), Inches(0.6), "×", size=20,
         bold=True, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    block(x_out, cy, "p̂ᵢᵗ⁺¹ = pᵢᵗ + gᵢ · Δᵢ", "masked residual added to current pose",
          fill=SPARSE_SOFT, line_c=SPARSE, width=Inches(3.4))

    # edges
    arrow(slide, x_in + Inches(1.2), cy - Inches(0.1), x_net - Inches(1.3), cy - dy)
    arrow(slide, x_in + Inches(1.2), cy + Inches(0.1), x_net - Inches(1.3), cy + dy)
    arrow(slide, x_net + Inches(1.3), cy - dy, x_mul - Inches(0.25), cy - Inches(0.2))
    arrow(slide, x_net + Inches(1.3), cy + dy, x_mul - Inches(0.25), cy + Inches(0.2))
    arrow(slide, x_mul + Inches(0.3), cy, x_out - Inches(1.7), cy)
    text(slide, x_net - Inches(2.3), cy - dy - Inches(0.85), Inches(4.6), Inches(0.35),
         "shared weights, applied to every object", size=12, color=MUTED, align=PP_ALIGN.CENTER,
         italic=True)
    text(slide, x_out - Inches(1.7), cy + Inches(0.5), Inches(3.4), Inches(0.5),
         "gate off  ⇒  object copied verbatim", size=13, color=SPARSE, align=PP_ALIGN.CENTER,
         bold=True)


def style_chart(chart, size=13, legend=True, ymax=None, fmt="0.00", label_size=12, axis_fmt=None):
    chart.font.name = FONT
    chart.font.size = Pt(size)
    chart.font.color.rgb = INK
    chart.has_title = False
    va = chart.value_axis
    va.has_major_gridlines = True
    va.major_gridlines.format.line.color.rgb = LINE
    va.major_gridlines.format.line.width = Pt(0.5)
    va.format.line.fill.background()
    va.tick_labels.font.size = Pt(size - 1)
    va.tick_labels.font.color.rgb = MUTED
    va.tick_labels.number_format = axis_fmt or fmt
    va.tick_labels.number_format_is_linked = False
    if ymax is not None:
        va.maximum_scale = ymax
        va.minimum_scale = 0
    ca = chart.category_axis
    ca.format.line.color.rgb = LINE
    ca.tick_labels.font.size = Pt(size)
    ca.has_major_gridlines = False
    plot = chart.plots[0]
    plot.gap_width = 60
    plot.overlap = -10
    plot.has_data_labels = True
    dl = plot.data_labels
    dl.number_format = fmt
    dl.number_format_is_linked = False
    dl.font.size = Pt(label_size)
    dl.font.color.rgb = INK
    dl.position = XL_LABEL_POSITION.OUTSIDE_END
    chart.has_legend = legend
    if legend:
        chart.legend.position = XL_LEGEND_POSITION.TOP
        chart.legend.include_in_layout = False
        chart.legend.font.size = Pt(size)


def color_series(chart, colors):
    for s, c in zip(chart.series, colors):
        s.format.fill.solid()
        s.format.fill.fore_color.rgb = c
        s.format.line.fill.background()


def stat(slide, x, y, w, value, label, color=INK, vsize=40):
    text(slide, x, y, w, Inches(0.8), value, size=vsize, bold=True, color=color)
    text(slide, x, y + Inches(0.78), w, Inches(0.6), label, size=14, color=MUTED)


def table(slide, x, y, col_w, rows, header=True, size=14, row_h=Inches(0.42), align=None,
          bold_cells=(), colors=None):
    """Minimal ruled table drawn with text boxes. rows: list of lists of str."""
    n_cols = len(col_w)
    align = align or [PP_ALIGN.LEFT] + [PP_ALIGN.CENTER] * (n_cols - 1)
    cy = y
    for r, row in enumerate(rows):
        cx = x
        is_head = header and r == 0
        for c, cell in enumerate(row):
            opts = {}
            if is_head:
                opts = {"color": MUTED, "size": size - 1, "bold": True}
            if (r, c) in bold_cells:
                opts["bold"] = True
            if colors and (r, c) in colors:
                opts["color"] = colors[(r, c)]
            text(slide, cx, cy, col_w[c], row_h, [[(cell, opts)]], size=size, align=align[c],
                 anchor=MSO_ANCHOR.MIDDLE, margin=0.04)
            cx += col_w[c]
        cy += row_h
        if is_head or r == len(rows) - 1:
            line(slide, x, cy, x + sum(col_w), cy, color=INK if is_head else LINE, width=1.0)
        else:
            line(slide, x, cy, x + sum(col_w), cy, color=LINE, width=0.5)
    return cy


# ---- slides ----------------------------------------------------------------
def build(qr_path: Path | None):
    d = Deck()

    # 1. Title -----------------------------------------------------------------
    s = d.slide(footer=False)
    rect(s, 0, 0, Inches(0.28), SLIDE_H, fill=SPARSE)
    text(s, Inches(0.9), Inches(0.75), Inches(8), Inches(0.35),
         "WORLDS WORKSHOP  ·  IEEE/RSJ IROS 2026", size=13, bold=True, color=SPARSE)
    text(s, Inches(0.9), Inches(1.35), Inches(7.6), Inches(2.6),
         [[("Modeling What Changes", {"size": 44, "bold": True})],
          [("Sparse, Residual World Models for Object-Centric Manipulation",
            {"size": 24, "color": MUTED})]], spacing=8)
    text(s, Inches(0.9), Inches(4.35), Inches(7.6), Inches(1.9),
         [[("Param Thakkar", {"bold": True}), ("   Veermata Jijabai Technological Institute", {"color": MUTED})],
          [("Parsika Paresh Shah", {"bold": True}), ("   Arizona State University", {"color": MUTED})],
          [("Manisha Sushant Gote", {"bold": True}), ("   ZuiGO Private Limited", {"color": MUTED})]],
         size=17, spacing=6)
    text(s, Inches(0.9), Inches(6.55), Inches(8), Inches(0.4),
         f"Code, data generators, checkpoints:  {REPO_URL}", size=12, color=MUTED)
    draw_table_scene(s, Inches(8.85), Inches(1.5), Inches(4.0), show_labels=False)
    notes(s, f"""
Good afternoon. I'm going to talk about a very simple idea for world models of manipulation
scenes: model what changes, and copy everything else.

(Title, authors and venue. Repository: {REPO_URL})
""")

    # 2. The observation ---------------------------------------------------------
    s = d.slide("Physical scenes are sparse in change", eyebrow="The observation")
    draw_table_scene(s, MARGIN, Inches(1.75), Inches(4.7))
    text(s, Inches(6.0), Inches(1.9), Inches(6.6), Inches(1.6),
         [[("One push. One object moves.", {"bold": True, "size": 26})],
          [("Every other object is exactly where it was.", {"size": 22, "color": MUTED})]],
         spacing=6)
    bullets(s, Inches(6.0), Inches(3.75), Inches(6.6), Inches(2.6), [
        [("Dominant recipe: ", {"bold": True}),
         ("encode the scene, regress the ", {}), ("entire", {"italic": True}),
         (" next state every step", {})],
        [("It has to re-produce the static majority ", {}),
         ("at every step, for every object", {"bold": True})],
        [("The change structure of the world is ", {}), ("thrown away", {"bold": True}),
         (" before learning starts", {})],
    ], size=18, gap=12)
    notes(s, """
Start with the observation. When a robot nudges one block on a cluttered table, exactly one
pose changes. The rest of the scene is not approximately the same; it is identical.

The dominant world-model recipe ignores that. It encodes the scene and regresses the whole
next state, monolithically, at every step. So it spends its capacity reproducing objects that
did nothing.
""")

    # 3. Three costs -------------------------------------------------------------
    s = d.slide("Re-predicting everything has three costs", eyebrow="The problem")
    cols = [
        ("1", "Capacity", "Parameters and operations are spent on the static majority, not on the "
                          "few objects that matter."),
        ("2", "Error injection", "A regressor that outputs every pose perturbs objects it should "
                                 "have copied. Closed loop, that error accumulates on objects that "
                                 "never moved: the predicted world drifts."),
        ("3", "No handle on “what”", "A monolithic predictor never says which objects it "
                                            "expects to move. For control and debugging, that belief "
                                            "is entangled in one regression output."),
    ]
    cw = Inches(3.85)
    for i, (num, head, body) in enumerate(cols):
        cx = MARGIN + i * (cw + Inches(0.25))
        rect(s, cx, Inches(1.85), cw, Inches(4.5), fill=PANEL, shape=MSO_SHAPE.ROUNDED_RECTANGLE,
             radius=0.05)
        text(s, cx + Inches(0.3), Inches(2.05), Inches(1), Inches(0.8), num, size=40, bold=True,
             color=SPARSE)
        text(s, cx + Inches(0.3), Inches(2.95), cw - Inches(0.6), Inches(0.6), head, size=22, bold=True)
        text(s, cx + Inches(0.3), Inches(3.6), cw - Inches(0.6), Inches(2.6), body, size=16,
             color=INK)
    text(s, MARGIN, Inches(6.45), Inches(12), Inches(0.45),
         "The second cost is the insidious one: it is exactly the objects that never moved that drift.",
         size=15, italic=True, color=MUTED)
    notes(s, """
Three costs. Capacity is the obvious one. Error injection is the insidious one: a regressor
that has to output every pose will inevitably perturb the objects it should have left alone,
and in a rollout that error compounds precisely on the objects that never moved. And third,
interpretability: a monolithic model has no explicit belief about what will change.

We ask whether explicitly modelling what changes addresses all three.
""")

    # 4. The idea: architecture ----------------------------------------------------
    s = d.slide("Gate what changes, copy the rest", eyebrow="Our model")
    draw_architecture(s, MARGIN, Inches(1.8), Inches(12), Inches(3.4))
    bullets(s, MARGIN, Inches(5.35), Inches(12), Inches(1.6), [
        [("Per-object gate ", {"bold": True}),
         ("decides which objects move; a ", {}), ("residual delta head ", {"bold": True}),
         ("predicts a pose increment applied only to gated objects", {})],
        [("Hard at inference: ", {"bold": True}),
         ("exactly zero update for un-gated objects. Trained end-to-end with a Gumbel "
          "straight-through gate", {})],
        [("Count-agnostic: ", {"bold": True}),
         ("heads are shared across objects and no layer is sized to N, so one model runs at any object count", {})],
    ], size=16, gap=6)
    notes(s, """
The model is deliberately simple and object-centric. For each object we build a feature
vector from its pose, velocity, relative goal and pusher positions, the action, and a
permutation-invariant aggregate over the other objects.

A gate network emits a change decision, zero or one. A delta head emits a pose residual in
R^3. The next pose is the current pose plus the masked residual. If the gate is off, the
object is copied verbatim: not approximately, exactly.

The gate is discrete, trained with a Gumbel straight-through estimator, so it is hard at
inference. And because the heads are shared and nothing is sized to the object count, a
model trained on 3 objects runs on 8 with no retraining.
""")

    # 5. Loss ------------------------------------------------------------------------
    s = d.slide("One loss, three terms", eyebrow="Training")
    rect(s, MARGIN, Inches(1.85), Inches(12), Inches(1.5), fill=PANEL, shape=MSO_SHAPE.ROUNDED_RECTANGLE,
         radius=0.08)
    text(s, MARGIN, Inches(1.85), Inches(12), Inches(1.5),
         [[("ℒ  =  BCE(g, m*)", {"color": SPARSE, "bold": True}),
           ("   +   λ", {}), ("Δ", {"size": 16}), (" · m* · ‖Δ − Δ*‖²", {}),
           ("   +   λ", {}), ("s", {"size": 16}), (" · ḡ", {})]],
         size=30, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    cw = Inches(3.85)
    terms = [
        ("Change detection", "Class-balanced binary cross-entropy against the ground-truth changed "
                             "mask m*. This is what the gate learns from.", SPARSE),
        ("Residual regression", "L2 on the delta, supervised only on objects that actually changed. "
                                "Un-gated objects contribute nothing.", INK),
        ("Sparsity", "Penalty on the mean gate activation ḡ. Trades recall for precision: "
                     "λs 0 → 1 moves precision 0.94 → 0.97, recall 0.81 → 0.71.", INK),
    ]
    for i, (head, body, col) in enumerate(terms):
        cx = MARGIN + i * (cw + Inches(0.25))
        text(s, cx, Inches(3.65), cw, Inches(0.5), head, size=20, bold=True, color=col)
        text(s, cx, Inches(4.2), cw, Inches(1.8), body, size=15)
    text(s, MARGIN, Inches(6.1), Inches(12), Inches(0.8),
         [[("Small by design: ", {"bold": True}),
           ("gate and delta heads are 2-layer, width-128 MLPs. Under 0.1M parameters, trains and "
            "plans on a laptop CPU.", {})]], size=15, color=MUTED)
    notes(s, """
Three terms. A class-balanced BCE on the gate against the ground-truth changed mask. An L2 on
the residual, supervised only where something actually changed. And a sparsity penalty on the
mean gate activation.

The sparsity weight is a precision-recall knob. A five-point sweep moves precision from 0.94
to 0.97 and recall from 0.81 to 0.71; F1 degrades only gently. We use 0.2 for prediction.
(Appendix D of the paper.)

Everything is tiny: two-layer, width-128 heads, under 0.1 million parameters.
""")

    # 6. Setup -----------------------------------------------------------------------
    s = d.slide("A controlled tabletop pushing benchmark", eyebrow="Setup")
    left = [
        [("Environment  ", {"bold": True}),
         ("procedural MuJoCo tabletop, N ∈ {3, 5, 8} free 5 cm boxes, scripted pushing policy", {})],
        [("State  ", {"bold": True}),
         ("pusher position, per-object planar pose (x, y, θ) and velocity, goal. Structured state, not pixels", {})],
        [("Action  ", {"bold": True}), ("end-effector Δxy to a position-controlled pusher", {})],
        [("Data  ", {"bold": True}),
         ("250 episodes × 100 steps per (N, seed), filtered to a hard subset with real motion; "
          "80/10/10 split with a configuration-leakage guard", {})],
        [("Seeds  ", {"bold": True}), ("headline tables are mean ± std over 3 training seeds", {})],
    ]
    bullets(s, MARGIN, Inches(1.85), Inches(7.3), Inches(4.8), left, size=17, gap=16)
    # baselines + metrics panel
    px = Inches(8.45)
    rect(s, px, Inches(1.8), Inches(4.2), Inches(4.7), fill=PANEL, shape=MSO_SHAPE.ROUNDED_RECTANGLE,
         radius=0.05)
    text(s, px + Inches(0.3), Inches(1.95), Inches(3.6), Inches(0.4), "BASELINES", size=12, bold=True,
         color=MUTED)
    text(s, px + Inches(0.3), Inches(2.3), Inches(3.6), Inches(1.5),
         [[("Dense MLP", {"bold": True, "color": DENSE}),
           ("  regresses all poses; width 256, 3 layers", {})],
          [("No-op", {"bold": True, "color": NOOP}),
           ("  predicts no change at all. The floor every world model must beat", {})]],
         size=15, spacing=6)
    text(s, px + Inches(0.3), Inches(4.05), Inches(3.6), Inches(0.4), "METRICS", size=12, bold=True,
         color=MUTED)
    text(s, px + Inches(0.3), Inches(4.4), Inches(3.6), Inches(2.0),
         [[("Per-object pose L2, ", {}), ("split into changed and unchanged objects", {"bold": True})],
          [("Change-detection precision, recall, F1", {})],
          [("Parameters, operations, latency", {})]],
         size=15, spacing=6)
    notes(s, """
The setup is deliberately controlled and I want to state its bounds up front: structured
per-object state, not pixels; three to eight rigid boxes; small MLPs.

Two baselines. A dense MLP that regresses every pose. And no-op, which predicts that nothing
moves. No-op matters: on a scene that is mostly static it is a strong floor, and I will come
back to it.

The metric that matters most today is per-object pose L2 split into changed and unchanged
objects. That split is what lets us say where a win comes from.
""")

    # 7. Headline ---------------------------------------------------------------------
    s = d.slide("Lower error at a fraction of the parameters", eyebrow="Prediction accuracy")
    cd = CategoryChartData()
    cd.categories = ["3 objects", "5 objects", "8 objects"]
    cd.add_series("Sparse (ours)", (0.136, 0.101, 0.071))
    cd.add_series("Dense MLP", (0.347, 0.319, 0.329))
    cd.add_series("No-op", (0.128, 0.107, 0.081))
    gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, MARGIN, Inches(1.7), Inches(7.4),
                            Inches(4.7), cd)
    style_chart(gf.chart, ymax=0.42, fmt="0.000", axis_fmt="0.0")
    color_series(gf.chart, [SPARSE, DENSE, NOOP])
    text(s, MARGIN, Inches(6.4), Inches(7.4), Inches(0.5),
         "Overall per-object pose L2 on held-out transitions, lower is better (Tables I and II; 3 seeds).",
         size=12, color=MUTED)
    sx = Inches(8.6)
    stat(s, sx, Inches(1.7), Inches(4), "2.5 – 4.6×", "lower overall L2 than dense, at every N", SPARSE)
    stat(s, sx, Inches(3.25), Inches(4), "8.6 – 11.1×", "fewer parameters (6.9k–10.8k vs 77k–92k)", SPARSE)
    stat(s, sx, Inches(4.8), Inches(4), "F1 0.80 – 0.87", "change detection, precision 0.92–0.97; "
                                                         "dense is degenerate (recall ≈ 1)", INK, vsize=32)
    notes(s, """
Here is the headline comparison, Table I of the paper. At every object count the sparse
model's overall L2 is 2.5 to 4.6 times lower than the dense MLP, with 8.6 to 11.1 times fewer
parameters. Change detection holds F1 0.80 to 0.87 at precision 0.92 to 0.97, where the dense
model is degenerate: it says everything moves.

Now look at the blue bars. That is no-op, and sparse essentially matches it. Hold that
thought; it is the most important thing on this slide.

Efficiency caveat, stated honestly: the win is parameters, not wall-clock. Per-object heads
scale with N, and the dense model's single matmul is faster in latency (Appendix C).
""")

    # 8. Qualitative --------------------------------------------------------------------
    s = d.slide("Sparse moves only what moves", eyebrow="Prediction accuracy")
    pic = s.shapes.add_picture(str(FIG / "qualitative.png"), MARGIN + Inches(0.2), Inches(1.7),
                               height=Inches(4.9))
    # crop the figure's own suptitle away; the slide title carries it
    pic.crop_top = 0.055
    pic.height = Inches(4.9)
    px = Inches(8.1)
    bullets(s, px, Inches(1.9), Inches(4.6), Inches(4.5), [
        [("Arrows ", {"bold": True}), ("are predicted displacement, current to next position", {})],
        [("Ground truth: ", {"bold": True}), ("one object moves per step", {})],
        [("Sparse (ours): ", {"bold": True, "color": SPARSE}),
         ("the gate fires on that object alone; the other two are copied", {})],
        [("Dense: ", {"bold": True, "color": DENSE}),
         ("hallucinates motion on every object, including the ones that should stay put", {})],
    ], size=17, gap=14)
    text(s, MARGIN + Inches(0.2), Inches(6.55), Inches(7), Inches(0.4),
         "Fig. 2 of the paper, two held-out scenes.", size=12, color=MUTED)
    notes(s, """
Qualitatively this is what it looks like. Two held-out scenes, arrows are predicted
displacement. Ground truth on the left: one object moves. Sparse in the middle: the gate fires
on that object and only that object. Dense on the right: it hallucinates motion on all three.

That is error injection, and it is what the gate removes.
""")

    # 9. Where the win comes from -------------------------------------------------------------
    s = d.slide("Where the win actually comes from", eyebrow="Why it wins")
    cd = CategoryChartData()
    cd.categories = ["Dense MLP", "+ object-centric", "+ residual", "+ gate (sparse)", "No-op"]
    cd.add_series("Overall L2", (0.367, 0.239, 0.221, 0.116, 0.128))
    gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, MARGIN, Inches(1.7), Inches(7.0),
                            Inches(3.6), cd)
    ch = gf.chart
    style_chart(ch, legend=False, ymax=0.42, size=12, fmt="0.000", axis_fmt="0.0")
    ser = ch.series[0]
    for i, c in enumerate([DENSE, GREY_BOX, GREY_BOX, SPARSE, NOOP]):
        pt = ser.points[i]
        pt.format.fill.solid()
        pt.format.fill.fore_color.rgb = c
        pt.format.line.fill.background()
    text(s, MARGIN, Inches(5.3), Inches(7.0), Inches(0.4),
         "Capacity-matched ladder, 3 objects, seed 0 (Table II). Each rung adds one ingredient.",
         size=12, color=MUTED)
    # unchanged-object row
    text(s, MARGIN, Inches(5.75), Inches(7.0), Inches(0.4),
         [[("Unchanged-object L2 down the ladder:  ", {"bold": True}),
           ("0.211  →  0.111  →  0.086  →  ", {}), ("0.001", {"bold": True, "color": SPARSE})]],
         size=16)
    text(s, MARGIN, Inches(6.2), Inches(7.0), Inches(0.6),
         "The gate is the largest single step (1.7–2.0× at every N) and the only rung that reaches "
         "below no-op.", size=14, color=MUTED)
    # right: oracle gate table
    px = Inches(8.2)
    text(s, px, Inches(1.7), Inches(4.5), Inches(0.5), "Changed-object L2 with a perfect gate", size=18,
         bold=True)
    table(s, px, Inches(2.25), [Inches(0.9), Inches(1.2), Inches(1.2), Inches(1.2)], [
        ["N", "predicted gate", "oracle gate", "no-op"],
        ["3", "0.312", "0.314", "0.348"],
        ["5", "0.426", "0.432", "0.446"],
        ["8", "0.466", "0.474", "0.470"],
    ], size=14, colors={(0, 3): NOOP, (0, 1): SPARSE})
    text(s, px, Inches(4.25), Inches(4.5), Inches(2.4),
         [[("Feeding the delta head the ground-truth mask barely moves the number.", {"bold": True})],
          [("So the win is detecting what changed and ", {}),
           ("not corrupting the static majority", {"bold": True, "color": SPARSE}),
           (". On the movers themselves, sparse is at roughly no-op quality: the bottleneck is "
            "delta regression, not detection.", {})]],
         size=15, spacing=8)
    notes(s, """
This is the slide I care most about. Why does it win?

First, it is not just smaller: a dense MLP shrunk to the same parameter budget is still 3 to
5 times worse (Appendix E). Second, the ladder on the left adds one ingredient at a time:
object-centric featurization, then an always-on residual, then the gate. Featurization earns a
third to a half of the gap, residual a bit more, but the gate is the largest single step and
the only rung that gets below no-op. The unchanged-object column shows the mechanism: error on
static objects goes from 0.211 to 0.001.

Now the table on the right, our oracle-gate diagnostic (Appendix F). If we give the delta head
the ground-truth changed mask, changed-object error barely moves. And it sits right at the
no-op floor. So, stated plainly: the win is not accurate dynamics on movers. It is change
detection that avoids injecting error into everything else. One-step contact deltas,
especially rotation, are simply hard, and the delta head is the highest-leverage thing to fix.

One missing rung we flag in the paper: a dense, non-object-centric model predicting residuals.
That is immediate future work.
""")

    # 10. World-model properties ----------------------------------------------------------------
    s = d.slide("Rollout, transfer across counts, sample efficiency", eyebrow="It is a world model")
    pic = s.shapes.add_picture(str(FIG / "worldmodel.png"), MARGIN, Inches(1.6), width=Inches(11.9))
    y0 = Inches(1.6) + pic.height + Inches(0.25)
    cw = Inches(3.85)
    caps = [
        ("(a) Rollout, 20 steps closed loop", "Sparse compounds 3.4–6.4× less error than dense at "
         "every N, but stays above no-op overall: it wins on static objects and only ties on movers."),
        ("(b) Zero-shot transfer", "One checkpoint, any object count. Off-diagonal F1 0.827 vs "
         "0.832 on the diagonal: 99.4% retention. Dense cannot even be run off-count."),
        ("(c) Sample efficiency", "Sparse reaches about 90% of its full-data F1 with 25% of the "
         "data. Dense is flat at every budget."),
    ]
    for i, (head, body) in enumerate(caps):
        cx = MARGIN + i * (cw + Inches(0.17))
        text(s, cx, y0, cw, Inches(0.45), head, size=16, bold=True,
             color=SPARSE if i else INK)
        text(s, cx, y0 + Inches(0.45), cw, Inches(1.6), body, size=14)
    notes(s, """
Three world-model properties, Figure 3 of the paper.

Rollout: close the loop for twenty steps. Dense drifts, sparse compounds far less error, 3.4 to
6.4 times less at horizon 20 (Table III). But be honest with the blue line: sparse is above
no-op at every horizon. The decomposition says why: sparse wins on unchanged objects and only
ties no-op on movers, consistent with the one-step result.

Transfer: because nothing is sized to N, a model trained on 3 objects runs on 8 with zero
retraining. Off-diagonal F1 retains 99.4 percent of the diagonal. The dense model cannot even
be executed off-count; its input dimension is wrong.

Sample efficiency: sparse reaches about 90 percent of full-data F1 with a quarter of the data.
(b and c are seed 0.)
""")

    # 11. Planning ---------------------------------------------------------------------------------
    s = d.slide("Can it plan? Separated from dense, not yet from random", eyebrow="Downstream planning")
    cd = CategoryChartData()
    cd.categories = ["Oracle (true simulator)", "Scripted controller", "Sparse (contact feats)",
                     "Random actions", "Dense (diverse data)"]
    cd.add_series("Success rate", (1.00, 0.95, 0.23, 0.15, 0.00))
    gf = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, MARGIN, Inches(1.65), Inches(6.9),
                            Inches(4.3), cd)
    ch = gf.chart
    style_chart(ch, legend=False, ymax=1.15, size=13)
    ch.category_axis.reverse_order = True
    ch.plots[0].gap_width = 45
    ser = ch.series[0]
    for i, c in enumerate([GREY_BOX, GREY_BOX, SPARSE, NOOP, DENSE]):
        pt = ser.points[i]
        pt.format.fill.solid()
        pt.format.fill.fore_color.rgb = c
        pt.format.line.fill.background()
    text(s, MARGIN, Inches(5.95), Inches(6.9), Inches(0.9),
         "Push-target-to-goal, 3 objects, success = target within 5 cm of goal. CEM planner "
         "(256 samples × 3 iterations, horizon 15, replan every step); only the model is swapped. "
         "Learned models: mean of 3 seeds × 20 episodes (Table IV).", size=12, color=MUTED)
    px = Inches(8.0)
    text(s, px, Inches(1.65), Inches(4.7), Inches(0.5), "Round 1: prediction-trained models cannot plan",
         size=17, bold=True, color=DENSE)
    text(s, px, Inches(2.15), Inches(4.7), Inches(1.4),
         "Both learned models: 0.00, worse than random. The planner visits teleported, near-rest "
         "states far from the training data. Dense hallucinates 5–10 cm of motion with no contact; "
         "the sparse gate goes silent.", size=14)
    text(s, px, Inches(3.65), Inches(4.7), Inches(0.5), "Round 2: fix the diagnosed cause",
         size=17, bold=True, color=SPARSE)
    text(s, px, Inches(4.15), Inches(4.7), Inches(1.5),
         "Contact-aware, velocity-free features and diverse mixed-policy training data. Same "
         "planner. Sparse goes 0.00 → 0.23 ± 0.06 and halves final distance. Dense stays at 0.00 at "
         "every seed.", size=14)
    rect(s, px, Inches(5.75), Inches(4.7), Inches(1.0), fill=PANEL, shape=MSO_SHAPE.ROUNDED_RECTANGLE,
         radius=0.08)
    text(s, px + Inches(0.2), Inches(5.75), Inches(4.3), Inches(1.0),
         [[("Robust finding: ", {"bold": True}),
           ("separation from dense at every seed. Not robust: the margin over random "
            "(weakest seed is exactly 0.15).", {})]], size=13, anchor=MSO_ANCHOR.MIDDLE)
    notes(s, """
Does any of this help control? Each model is the forward simulator inside the same CEM
planner on a push-to-goal task; the model is the only thing that changes.

Round 1 was a clean negative result. The oracle solves every episode, so the planner and task
are fine. Both learned models score zero, worse than random. The cause is out-of-distribution
querying: the planner visits near-rest, teleported states the models never saw. Dense
hallucinates motion with no contact; sparse's velocity features leave its gate silent.

Round 2 fixes the diagnosed cause: contact-aware, velocity-free features and diverse data. With
the planner unchanged, sparse goes from zero to 0.23 and halves its final distance, while dense
stays at zero at every seed.

But I will not oversell it. 0.23 against random's 0.15 is not significant; the weakest seed
is exactly 0.15. The robust claim is separation from dense at every seed, not that sparse
"begins to plan".
""")

    # 12. What the headline is and is not ---------------------------------------------------------
    s = d.slide("What the headline is, and what it is not", eyebrow="Reading the results honestly")
    cw = Inches(5.85)
    rect(s, MARGIN, Inches(1.75), cw, Inches(4.9), fill=SPARSE_SOFT, shape=MSO_SHAPE.ROUNDED_RECTANGLE,
         radius=0.05)
    rect(s, MARGIN + cw + Inches(0.3), Inches(1.75), cw, Inches(4.9), fill=PANEL,
         shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.05)
    text(s, MARGIN + Inches(0.35), Inches(1.95), cw - Inches(0.7), Inches(0.5), "IT IS", size=14, bold=True,
         color=SPARSE)
    bullets(s, MARGIN + Inches(0.35), Inches(2.5), cw - Inches(0.7), Inches(4.0), [
        [("A change detector that avoids error injection: ", {"bold": True}),
         ("F1 0.80–0.87 where dense is degenerate; unchanged-object error 0.211 → 0.001", {})],
        [("Far better than dense ", {"bold": True}), ("one-step (2.5–4.6×), in rollout (3.4–6.4×), "
                                                       "and in planning (0.23 vs 0.00 at every seed)", {})],
        [("Count-invariant and sample-efficient, ", {"bold": True}),
         ("with an inspectable per-object change mask", {})],
        [("8.6–11.1× fewer parameters", {"bold": True})],
    ], size=16, gap=10)
    x2 = MARGIN + cw + Inches(0.65)
    text(s, x2, Inches(1.95), cw - Inches(0.7), Inches(0.5), "IT IS NOT", size=14, bold=True, color=DENSE)
    bullets(s, x2, Inches(2.5), cw - Inches(0.7), Inches(4.0), [
        [("An accurate mover predictor: ", {"bold": True}),
         ("changed-object L2 sits at or barely above no-op; overall L2 only matches no-op at N = 5, 8", {})],
        [("Better than no-op in rollout: ", {"bold": True}),
         ("worse than no-op overall at every N and horizon", {})],
        [("Significantly better than random at planning: ", {"bold": True}),
         ("0.23 ± 0.06 vs 0.15, one planner, one task, 3 objects", {})],
        [("Faster in wall-clock, ", {"bold": True}), ("or pixel-space: structured state only", {})],
    ], size=16, gap=10)
    notes(s, """
Let me put both halves side by side, because the paper does.

It is a change detector that avoids error injection. That is a real and durable win over
dense, one-step, in rollout, and in planning, at a tenth of the parameters, with a mask you
can inspect.

It is not an accurate predictor of the objects that move. On those, it is at no-op quality.
Overall it matches no-op; in rollout it is worse than no-op; in planning it is not
significantly better than random. Phrases like "predicts dynamics far more accurately" invite
a reading the data do not support, and we retired them from the paper.

Also: F1 is partly a property of the metric (thresholding a dense regressor forces recall to
1), so we foreground the changed / unchanged L2 split instead. And the ladder is missing a
dense-residual rung, which confounds residual against object-centricity.
""")

    # 13. Takeaways ---------------------------------------------------------------------------------
    s = d.slide("Takeaways", eyebrow="Conclusion")
    takes = [
        ("Model what changes.", "A per-object gate plus a masked residual is a one-line bias that "
                                "stops a world model from corrupting the static majority of a scene."),
        ("Decompose the metric.", "Overall L2 hid the story. Splitting error into changed and "
                                  "unchanged objects, and always reporting no-op, is what made the "
                                  "claim honest."),
        ("The delta head is the lever.", "Detection is solved to F1 0.8+; regression of contact "
                                         "deltas is not. Next: a dense-residual baseline, a "
                                         "distributional delta head, DAgger on planner states, "
                                         "multi-step training, slot-based encoders."),
    ]
    for i, (head, body) in enumerate(takes):
        y = Inches(1.8) + i * Inches(1.45)
        text(s, MARGIN, y, Inches(0.7), Inches(0.8), str(i + 1), size=34, bold=True, color=SPARSE)
        text(s, MARGIN + Inches(0.75), y, Inches(7.4), Inches(0.5), head, size=21, bold=True)
        text(s, MARGIN + Inches(0.75), y + Inches(0.5), Inches(7.4), Inches(0.95), body, size=15)
    # repo panel with QR
    px = Inches(9.4)
    rect(s, px, Inches(1.8), Inches(3.25), Inches(4.4), fill=PANEL, shape=MSO_SHAPE.ROUNDED_RECTANGLE,
         radius=0.05)
    if qr_path and qr_path.exists():
        s.shapes.add_picture(str(qr_path), px + Inches(0.55), Inches(2.0), width=Inches(2.15))
    text(s, px + Inches(0.2), Inches(4.25), Inches(2.85), Inches(1.9),
         [[("Code, data generators, checkpoints, result tables", {"bold": True})],
          [("github.com/ParamThakkar123/\nsparse_world_models", {"color": MUTED, "size": 12})],
          [("Live in-browser demo runs the real checkpoints", {"size": 12, "color": MUTED})]],
         size=13, align=PP_ALIGN.CENTER, spacing=6)
    text(s, MARGIN, Inches(6.3), Inches(8.5), Inches(0.5),
         "Thank you. Questions welcome.", size=18, bold=True, color=MUTED)
    notes(s, f"""
Three takeaways. Model what changes: the gate plus masked residual is a one-line bias that
stops error injection. Decompose the metric: the changed / unchanged split and the no-op
column are what turned a headline into an honest claim. And the delta head is the lever:
detection is solved, regression of contact deltas is not.

Code, data generators, checkpoints and machine-readable tables are at {REPO_URL}. The QR code
also reaches an in-browser demo that runs the real checkpoints on live physics.

Thank you.
""")

    # 14. Since the camera-ready (optional) ------------------------------------------------------------
    s = d.slide("Since the camera-ready: run the trivial baselines", eyebrow="Postscript (optional)")
    text(s, MARGIN, Inches(1.75), Inches(12), Inches(1.1),
         [[("We asked what a one-parameter rule scores on our own change-detection benchmark. ",
            {"bold": True}),
           ("It beats the learned gate.", {"bold": True, "color": DENSE})]], size=22, spacing=4)
    cd = CategoryChartData()
    cd.categories = ["Learned gate (this paper)", "“moving or near the pusher”, 1 parameter"]
    cd.add_series("Change-detection F1", (0.823, 0.927))
    gf = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, MARGIN, Inches(2.9), Inches(6.9),
                            Inches(2.9), cd)
    ch = gf.chart
    style_chart(ch, legend=False, ymax=1.1, size=13)
    ch.category_axis.reverse_order = True
    ch.plots[0].gap_width = 40
    ser = ch.series[0]
    for i, c in enumerate([SPARSE, INK]):
        pt = ser.points[i]
        pt.format.fill.solid()
        pt.format.fill.fore_color.rgb = c
        pt.format.line.fill.background()
    text(s, MARGIN, Inches(5.85), Inches(6.9), Inches(0.5),
         "Motion-filtered benchmark, 3 objects, F1 on held-out transitions.", size=12, color=MUTED)
    px = Inches(8.0)
    bullets(s, px, Inches(2.9), Inches(4.7), Inches(3.6), [
        [("With one point-like pusher, ", {}),
         ("the changed label is recoverable from a single scalar: ", {"bold": True}),
         ("speed, or distance to the pusher", {})],
        [("Filtering the data harder just changes ", {}), ("which", {"italic": True}),
         (" one-liner wins", {})],
        [("The error-suppression mechanism stands; the ", {}), ("benchmark", {"bold": True}),
         (" does not. Every “our model predicts what changes” claim needs a trivial-baseline audit", {})],
    ], size=15, gap=12)
    text(s, MARGIN, Inches(6.2), Inches(12), Inches(0.6),
         "Follow-up work in preparation. Audit battery and benchmarks in the repository.", size=13,
         italic=True, color=MUTED)
    notes(s, """
OPTIONAL SLIDE. Use it if you want to end on the strongest possible note of scientific
honesty; drop it if the talk is short or you prefer to keep the follow-up for its own venue.

After the camera-ready we asked a question we should have asked earlier: what does a trivial
rule score on our own benchmark? "The object is moving, or it is near the pusher": one
parameter, F1 0.927. Our learned gate: 0.823.

The reason is structural. With a single point-like pusher the changed label is recoverable
from one scalar, so the benchmark is degenerate. Filtering harder only changes which one-liner
wins. The error-suppression mechanism in this paper still holds; what does not hold is any
claim that a learned gate is needed to get it on this benchmark. Our follow-up is a
trivial-baseline audit battery, and our recommendation to this community is: run it.

(Numbers: README / experiments/RESULTS.md, motion benchmark, best trivial rule vs best learned gate.)
""")

    # ---- backup ---------------------------------------------------------------------------------------
    s = d.slide("Backup: efficiency in full", eyebrow="Appendix C")
    table(s, MARGIN, Inches(1.8), [Inches(0.8), Inches(1.6), Inches(1.6), Inches(1.4), Inches(1.6),
                                    Inches(1.6), Inches(2.2)], [
        ["N", "sparse params", "dense params", "param ratio", "sparse FLOPs", "dense FLOPs",
         "latency ms (sparse / dense)"],
        ["3", "6 916", "76 809", "11.1×", "39 936", "152 576", "0.19 / 0.04"],
        ["5", "8 452", "82 959", "9.8×", "81 920", "164 864", "0.20 / 0.04"],
        ["8", "10 756", "92 184", "8.6×", "167 936", "183 296", "0.20 / 0.04"],
    ], size=15, row_h=Inches(0.5))
    text(s, MARGIN, Inches(4.2), Inches(12), Inches(1.5),
         [[("Parameter efficiency is the durable win. ", {"bold": True}),
           ("The operation-count advantage erodes from 3.8× to 1.1× as per-object heads scale with N, "
            "and the dense model's single matmul is faster in CPU latency. We claim capacity "
            "efficiency, not speed.", {})]], size=16)
    notes(s, "Backup. Appendix C, Table V: parameters, FLOP estimate and CPU latency (Intel i7-12650H).")

    s = d.slide("Backup: sparsity weight and per-seed planning", eyebrow="Appendices D and G")
    text(s, MARGIN, Inches(1.7), Inches(6), Inches(0.45), "Sparsity weight λs (3 objects, seed 0)",
         size=17, bold=True)
    table(s, MARGIN, Inches(2.2), [Inches(0.9), Inches(0.9), Inches(1.2), Inches(1.0), Inches(1.3),
                                    Inches(1.2)], [
        ["λs", "F1", "precision", "recall", "changed L2", "overall L2"],
        ["0.0", "0.873", "0.941", "0.814", "0.389", "0.146"],
        ["0.05", "0.865", "0.940", "0.801", "0.389", "0.146"],
        ["0.2", "0.849", "0.938", "0.776", "0.388", "0.145"],
        ["0.5", "0.845", "0.967", "0.750", "0.387", "0.144"],
        ["1.0", "0.815", "0.965", "0.705", "0.385", "0.143"],
    ], size=14, row_h=Inches(0.42), bold_cells={(3, 0)})
    px = Inches(7.7)
    text(s, px, Inches(1.7), Inches(5), Inches(0.45), "Contact-aware planning, per seed",
         size=17, bold=True)
    table(s, px, Inches(2.2), [Inches(1.7), Inches(0.8), Inches(0.8), Inches(0.8), Inches(1.2)], [
        ["model", "seed 0", "seed 1", "seed 2", "mean ± std"],
        ["sparse success", "0.25", "0.15", "0.30", "0.23 ± 0.06"],
        ["dense success", "0.00", "0.00", "0.00", "0.00 ± 0.00"],
        ["sparse final dist.", "0.188", "0.231", "0.193", "0.204 ± 0.019"],
    ], size=14, row_h=Inches(0.42), colors={(1, 0): SPARSE, (2, 0): DENSE})
    text(s, MARGIN, Inches(5.2), Inches(12), Inches(1.2),
         [[("Parameter-matched dense (Appendix E): ", {"bold": True}),
           ("shrinking dense to the sparse budget gives L2 0.363 / 0.363 / 0.415 and F1 0.538 / 0.388 / "
            "0.294 at N = 3 / 5 / 8. Sparse still wins on every metric at every count, so the "
            "degeneracy is not a capacity artifact.", {})]], size=15)
    notes(s, "Backup. Appendix D (Table VI), Appendix G (Table IX, 20 episodes per seed) and Appendix E (Table VII).")

    return d


def make_qr(path: Path):
    try:
        import qrcode
    except ImportError:
        return None
    img = qrcode.make(REPO_URL, border=4)  # full quiet zone: with 1 module, decoders miss it
    img.save(path)
    return path


def export_notes(deck: Deck, path: Path):
    """Write the notes pane of every slide to a markdown rehearsal script."""
    out = ["# Speaker script: Modeling What Changes (WORLDS @ IROS 2026)", "",
           "Generated from the notes pane of `WORLDS_IROS2026_talk.pptx` by `build_deck.py`. "
           "Slides 1-13 are the talk (about 10 minutes), 14 is an optional postscript, 15-16 are backup.", ""]
    for i, s in enumerate(deck.prs.slides, 1):
        titles = [sh.text_frame.text for sh in s.shapes
                  if sh.has_text_frame and 620000 < sh.top < 660000 and sh.text_frame.text.strip()]
        title = titles[0] if titles else "Title"
        out += [f"## {i}. {title.strip()}", "", s.notes_slide.notes_text_frame.text.strip(), ""]
    path.write_text(chr(10).join(out), encoding="utf-8")


def render(pptx_path: Path):
    """Export PDF through PowerPoint, then a PNG per slide via poppler (both optional tooling)."""
    import win32com.client  # type: ignore

    pdf = pptx_path.with_suffix(".pdf")
    app = win32com.client.Dispatch("PowerPoint.Application")
    pres = app.Presentations.Open(str(pptx_path), WithWindow=False)
    pres.SaveAs(str(pdf), 32)  # ppSaveAsPDF
    pres.Close()
    app.Quit()
    png_dir = pptx_path.parent / "preview"
    png_dir.mkdir(exist_ok=True)
    subprocess.run(["pdftoppm", "-png", "-r", "60", str(pdf), str(png_dir / "slide")], check=True)
    return pdf, png_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", action="store_true", help="export PDF and PNG previews (PowerPoint + poppler)")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    qr = make_qr(args.out.parent / "repo_qr.png")
    deck = build(qr)
    deck.save(args.out)
    export_notes(deck, args.out.parent / "speaker_notes.md")
    print(f"wrote {args.out} ({deck.n} slides) and speaker_notes.md")
    if args.render:
        pdf, png_dir = render(args.out)
        print(f"rendered {pdf} and {png_dir}/")
