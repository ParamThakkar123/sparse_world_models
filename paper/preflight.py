"""Mechanical pre-submission checks for the ICLR 2027 paper.

The checks here are the ones a human reliably gets wrong at 3am on the deadline: an
uncommented `\\iclrfinalcopy`, a name left in a PDF metadata field, a style file quietly
edited to claw back half a page, a retired number copied across from the workshop draft.

It does not check the science. `SUBMISSION_RUNBOOK.md` steps 1 and 2 do that, by hand,
against `experiments/RESULTS.md`.

    python paper/preflight.py            # check paper/main.tex and main.pdf if built
    python paper/preflight.py --pdf out/main.pdf

Exit status is 1 if any FAIL fires. WARNs do not fail the run -- they are things that are
usually fine and occasionally a desk rejection, so they need a human, not a gate.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# The style files exactly as downloaded from media.iclr.cc on 2026-08-18. ICLR requires the
# official style unmodified; shaving the margins to fit nine pages is the classic desk
# rejection, and it is invisible in a diff nobody runs.
PRISTINE = {
    "iclr2027_conference.sty": "797deef41724e93761426ac0cbcca46279a91cc650dd1f0ce76a4f08d2098ea6",
    "iclr2027_conference.bst": "2d67552db7ed38ccfccb5957b52f95656e25c249724761d3cf5f7922ad1844c5",
    "fancyhdr.sty": "b56ec4434b9f4607529a4b23dc68ad8d4b94f1f631c8cddaf7da78140d53a5ea",
    "natbib.sty": "88bc70c0e48461934cab5b2accef06b74a8b3ac45ad03ccd3f2a6b7e0d6d530d",
}

# Packages whose whole purpose is to change the page geometry the style file fixed.
LAYOUT_HACKS = (r"\usepackage{geometry}", r"\usepackage[", r"\setlength{\textheight}",
                r"\setlength{\textwidth}", r"\setlength{\topmargin}",
                r"\addtolength{\textheight}", r"\vspace{-", r"\small\begin{abstract}")

# Numbers from the workshop draft that a later result refuted. They are the headline table
# computed on the leaking splits, plus the two refuted claims. If any of these strings appears
# in the submission it is almost certainly a copy-paste from the superseded draft.
RETIRED_NUMBERS = {
    "0.867": "leaky-split headline sparse F1 at N=3 (superseded; see RESULTS.md leak section)",
    "0.802": "leaky-split headline sparse F1 at N=5 (superseded)",
    "0.829": "leaky-split headline sparse F1 at N=8 (superseded)",
    "99.4": "cross-count retention on leaky splits; the clean-split value is 99.5 +/- 0.4",
    "2.55x": "published sparse/dense L2 ratio at N=3; clean splits give 2.46x",
}

# Phrases that give the game away in a double-blind submission.
DEANONYMISING = (
    r"\bour (previous|earlier|prior) (paper|work|submission)\b",
    r"\bas we (showed|reported) in\b",
    r"WorldModelsWorkshop",
    r"github\.com/[A-Za-z0-9_.-]+",
    r"IEEE_Conference_Template",
)

failures: list[str] = []
warnings: list[str] = []


def fail(message: str) -> None:
    failures.append(message)
    print(f"FAIL  {message}")


def warn(message: str) -> None:
    warnings.append(message)
    print(f"WARN  {message}")


def ok(message: str) -> None:
    print(f"ok    {message}")


def check_style_files() -> None:
    for name, expected in PRISTINE.items():
        path = HERE / name
        if not path.exists():
            fail(f"{name} is missing from paper/")
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            fail(f"{name} has been modified. ICLR requires the official style file unmodified; "
                 f"re-download from media.iclr.cc rather than keeping the edit.")
        else:
            ok(f"{name} matches the official release")


def check_source(tex: Path, identity_terms: list[str]) -> None:
    if not tex.exists():
        fail(f"{tex} not found")
        return
    source = tex.read_text(encoding="utf-8", errors="replace")

    final_copy = [line for line in source.splitlines()
                  if r"\iclrfinalcopy" in line and not line.lstrip().startswith("%")]
    if final_copy:
        fail(r"\iclrfinalcopy is ACTIVE. This reveals author identity -> desk rejection. "
             "Comment it out for submission.")
    else:
        ok(r"\iclrfinalcopy is commented out")

    for term in identity_terms:
        if not term:
            continue
        for match in re.finditer(re.escape(term), source, re.IGNORECASE):
            line = source[:match.start()].count("\n") + 1
            fail(f"{tex.name}:{line} contains the identifying string {term!r}")
    if identity_terms:
        ok(f"checked {len(identity_terms)} identity term(s) against the source")

    for pattern in DEANONYMISING:
        for match in re.finditer(pattern, source, re.IGNORECASE):
            line = source[:match.start()].count("\n") + 1
            warn(f"{tex.name}:{line} may deanonymise: {match.group(0)!r}")

    # Layout hacks: `\usepackage[` is too broad to fail on, so it is only reported when the
    # option list mentions a length.
    for hack in LAYOUT_HACKS:
        if hack == r"\usepackage[":
            for match in re.finditer(r"\\usepackage\[[^\]]*\]\{geometry\}", source):
                warn(f"geometry package with options: {match.group(0)!r}")
            continue
        if hack in source:
            warn(f"possible layout hack in source: {hack!r}")

    body = source.split(r"\begin{document}", 1)[-1]
    for number, why in RETIRED_NUMBERS.items():
        if number in body:
            warn(f"retired number {number!r} appears in the body -- {why}")

    for required, label in ((r"\section*{Reproducibility statement}", "reproducibility statement"),
                            (r"\section*{Use of large language models}", "LLM-use disclosure")):
        if required not in source:
            warn(f"no {label} section found. The LLM disclosure is MANDATORY under the ICLR "
                 "2027 AI policy and does not count toward the page limit.")
        else:
            ok(f"{label} section present")


def check_pdf(pdf: Path) -> None:
    if not pdf.exists():
        warn(f"{pdf} not built; run pdflatex before the final check")
        return
    if shutil.which("pdfinfo") is None:
        warn("pdfinfo not on PATH; page count and PDF metadata unchecked")
        return
    info = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout
    fields = dict(
        (key.strip(), value.strip())
        for key, _, value in (line.partition(":") for line in info.splitlines()) if key
    )
    pages = int(fields.get("Pages", "0"))
    # The 9-page limit applies to the main text; references and appendices are exempt, so this
    # is a prompt to measure rather than a gate.
    if pages > 9:
        warn(f"{pages} pages total. The 9-page limit is on the MAIN TEXT -- confirm the "
             "references begin on or before page 10.")
    else:
        ok(f"{pages} pages total (main text within the limit)")

    for field in ("Author", "Creator", "Producer", "Title", "Subject", "Keywords"):
        value = fields.get(field, "")
        if value and field in ("Author", "Subject", "Keywords"):
            warn(f"PDF metadata {field}={value!r} -- clear it before submission")
    ok("PDF metadata inspected")


def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-submission checks for the ICLR paper.")
    parser.add_argument("--tex", type=Path, default=HERE / "main.tex")
    parser.add_argument("--pdf", type=Path, default=HERE / "main.pdf")
    parser.add_argument(
        "--identity", nargs="*", default=None,
        help="Strings that must not appear (names, emails, institutions). Defaults to the "
             "local git user name and email, which is where a real name usually leaks in.",
    )
    args = parser.parse_args()

    identity = args.identity
    if identity is None:
        identity = []
        for key in ("user.name", "user.email"):
            result = subprocess.run(["git", "config", "--get", key],
                                    capture_output=True, text=True)
            value = result.stdout.strip()
            if value:
                identity.append(value)
                identity.extend(part for part in re.split(r"[@\s.]+", value) if len(part) > 3)
        identity = sorted(set(identity))

    print("== style files")
    check_style_files()
    print("\n== source")
    check_source(args.tex, identity)
    print("\n== pdf")
    check_pdf(args.pdf)

    print(f"\n{len(failures)} failure(s), {len(warnings)} warning(s)")
    if failures:
        print("Fix every FAIL before submitting. WARNs need a human decision, not a fix.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
