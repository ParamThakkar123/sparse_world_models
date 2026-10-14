"""Build the anonymised supplementary archive for the ICLR submission.

``SUBMISSION_RUNBOOK.md`` step 3 names shipping the code as supplementary as "the single most
likely way this project deanonymises itself", and it is right: the repository carries a
non-anonymous workshop paper, a browser demo with an author card and three e-mail addresses,
a GitHub Pages URL containing the first author's username, and a git history full of names.
Doing this by hand at 3am is how papers get desk rejected.

This script copies the repository into a staging directory, drops everything that cannot be
anonymised, rewrites what can be, and then **re-scans the result and refuses to produce an
archive if a single identity string survives**. The verification is the point; the copying is
incidental.

    python -m experiments.make_supplementary                    # -> dist/supplementary.zip
    python -m experiments.make_supplementary --no-demo          # drop docs/ entirely
    python -m experiments.make_supplementary --check-only       # scan, build nothing

Exit status is non-zero if verification fails, so it can gate a submission checklist.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Directories and globs that never go in the archive. Three reasons appear here: identity
# (the workshop paper, git history), size (datasets, checkpoints, run artifacts), and noise.
EXCLUDE_DIRS = {
    ".git", ".github", ".pytest_cache", ".ruff_cache", ".benchmarks", ".playwright-mcp",
    "__pycache__", ".venv", "venv", "node_modules",
    "IEEE_Conference_Template",     # the NON-ANONYMOUS workshop paper and its PDF
    "dist",
}
EXCLUDE_GLOBS = (
    "*.zip", "*.pt", "*.npz", "*.log", "*.aux", "*.bbl", "*.blg", "*.out", "*.synctex.gz",
    "*.pyc", "header-*.png", "NUL",
    # Per-run bulk. `experiments/runs/` itself is NOT excluded: its summary.json and
    # config.json are what every claim in the paper points at, they total ~1 MB, and shipping
    # them is the difference between "trust our table" and "here is the artifact behind it".
    "experiments/runs/*/*.csv", "experiments/runs/*/*.md", "experiments/runs/*/*.png",
)
EXCLUDE_PATHS = {
    Path("data/transitions"),
    Path("paper/main.pdf"),        # the built PDF is submitted separately, not in the zip
    # Submission-process files. They discuss the authors' own identity and the workshop paper
    # by name, they are not research artifacts, and this script necessarily contains a list of
    # the very strings it forbids -- so it must not ship inside its own archive.
    Path("SUBMISSION_RUNBOOK.md"),
    Path("experiments/make_supplementary.py"),
    # Its test plants author names on purpose to prove the verifier catches them, so it must
    # be excluded for the same reason the packager itself is.
    Path("tests/test_make_supplementary.py"),
}

# Every identity string that must not survive, checked case-insensitively. Keep this list
# broader than feels necessary; a false positive costs a minute and a miss costs the paper.
IDENTITY = [
    "thakkar", "paramthakkar", "param thakkar",
    "parsika", "pshah144",
    "manisha", "gote", "zuigo",
    "vjti", "veermata jijabai",
    "asu.edu", "puthakkar",
    "worldmodelsworkshop",
    "github.com/",
    "huggingface.co/",
    "sparse_world_models",
    "iros 2026", "worlds workshop", "worlds @ iros",
    # The workshop paper's title: public, indexed, non-anonymous. Matched with enough of the
    # subtitle to distinguish it from the ordinary English phrase "modeling what changes",
    # which the ledger uses in prose when quoting a retired claim.
    "modeling what changes: sparse",
    "paramth",
]

# Textual rewrites applied to every text file in the staging copy, in order. These handle the
# structured cases; anything they miss is caught by the verification scan and reported rather
# than shipped.
REWRITES: list[tuple[str, str]] = [
    # Hugging Face / GitHub bases in the demo. The demo ships its own assets in docs/assets,
    # so removing the remote fallback degrades gracefully to the local files.
    (r"https://huggingface\.co/datasets/[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+/resolve/main", ""),
    (r"https://huggingface\.co/[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+", "#"),
    (r"https://github\.com/[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+(/[A-Za-z0-9_.\-/]*)?", "#"),
    (r"https://[A-Za-z0-9_.\-]+\.github\.io/[A-Za-z0-9_.\-]+/?", "#"),
    # Mail addresses anywhere.
    (r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", "anonymous@example.invalid"),
    # The repository name, wherever it is used as a slug.
    (r"sparse_world_models", "anonymous_submission"),
    (r"WorldModelsWorkshop", "anonymous-submission"),
    # Bare `Username/repo` slugs in publishing tooling and READMEs, which the URL patterns
    # above do not reach because they carry no scheme. The truncated mirror handle is a
    # separate string and was missed on the first pass -- hence the verification step.
    (r"ParamThakkar123", "anonymous"),
    (r"ParamTh(?![a-z])", "anonymous"),
    # The workshop paper's TITLE is an identity leak in its own right: it is public, indexed,
    # and carries the author list. Replace it with this submission's title wherever it appears.
    (r"Modeling What Changes:\s*Sparse,?\s*Residual\s*(<[^>]+>)?\s*World Models for "
     r"Object-Centric\s*(<[^>]+>)?\s*Manipulation",
     "Change-Detection Benchmarks for Object-Centric World Models Are Degenerate"),
    # The workshop paper. Naming the venue identifies the authors just as surely as naming
    # them, since that paper is public and non-anonymous. ICLR permits citing related work by
    # the same authors in the third person; it does not permit announcing it as ours.
    (r"WORLDS\s*@?\s*(Workshop\s*,?\s*)?(IEEE/RSJ\s*)?(International Conference on Intelligent "
     r"Robots and Systems\s*)?\(?IROS\)?\s*,?\s*2026", "a prior workshop"),
    (r"WORLDS Workshop[^,.\n]*", "a prior workshop"),
    (r"WORLDS @ IROS 2026", "a prior workshop"),
    (r"accepted at the WORLDS[^'\"\n]*", "presented at a prior workshop"),
    (r"Copyright \(c\) 2026 [^\n]+", "Copyright (c) 2026 Anonymous authors"),
]

# Whole blocks replaced structurally, because a regex over them would leave broken markup.
BLOCK_REWRITES: list[tuple[str, str, str]] = [
    (
        "docs/index.html",
        r"<ol class=\"authors\">.*?</ol>",
        '<ol class="authors">\n        <li class="author">\n'
        "          <strong>Anonymous authors</strong>\n"
        "          <span>Paper under double-blind review</span>\n"
        "        </li>\n      </ol>",
    ),
    (
        "docs/index.html",
        r"<pre id=\"bibtex\"><code>.*?</code></pre>",
        '<pre id="bibtex"><code>@inproceedings{anonymous2027degenerate,\n'
        "  title  = {Change-Detection Benchmarks for Object-Centric\n"
        "            World Models Are Degenerate},\n"
        "  author = {Anonymous authors},\n"
        "  booktitle = {Under review},\n"
        "  year   = {2027}\n"
        "}</code></pre>",
    ),
    (
        "docs/index.html",
        r"<p class=\"eyebrow\">.*?</p>",
        '<p class="eyebrow">Anonymous submission &middot; under double-blind review</p>',
    ),
    (
        # Emptying the remote hosts leaves `['', '']`, which builds absolute-path fallback URLs
        # that 404 after the local assets have already loaded. Drop the list instead.
        "docs/assets/app.js",
        r"const HF_BASES = \[.*?\];",
        "// Remote asset mirrors removed for anonymous review; docs/assets/ is self-contained.\n"
        "const HF_BASES = [];",
    ),
]

TEXT_SUFFIXES = {
    ".py", ".md", ".txt", ".tex", ".bib", ".html", ".js", ".css", ".json", ".sh",
    ".cfg", ".toml", ".yml", ".yaml", ".xml",
}
# Extensionless text files that still need rewriting -- LICENSE is the one that matters, and
# missing it is how the copyright holder's name reaches a reviewer.
TEXT_NAMES = {"LICENSE", "LICENCE", "NOTICE", "AUTHORS", "CITATION", "Makefile"}

# A hosting URL whose path is a placeholder identifies nobody. Without this the scan flags
# `huggingface.co/datasets/<repo>/...` and the fix would be to delete documentation.
PLACEHOLDER_URL = re.compile(
    r"(github|huggingface)\.co(m)?/[^\s'\"`)]*[<{][^\s'\"`)]*", re.IGNORECASE
)


def is_excluded(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    if any(part in EXCLUDE_DIRS for part in rel.parts):
        return True
    if any(rel == p or p in rel.parents for p in EXCLUDE_PATHS):
        return True
    return any(path.match(g) for g in EXCLUDE_GLOBS)


def stage(dest: Path, include_demo: bool) -> list[Path]:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    copied: list[Path] = []
    for src in sorted(ROOT.rglob("*")):
        if src.is_dir() or is_excluded(src):
            continue
        rel = src.relative_to(ROOT)
        if not include_demo and rel.parts and rel.parts[0] == "docs":
            continue
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        copied.append(rel)
    return copied


def rewrite(dest: Path) -> int:
    changed = 0
    for rel, pattern, replacement in BLOCK_REWRITES:
        path = dest / rel
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        new = re.sub(pattern, replacement, text, flags=re.DOTALL)
        if new != text:
            path.write_text(new, encoding="utf-8")
            changed += 1
    for path in sorted(dest.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in TEXT_NAMES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        new = text
        for pattern, replacement in REWRITES:
            new = re.sub(pattern, replacement, new)
        if new != text:
            path.write_text(new, encoding="utf-8")
            changed += 1
    return changed


def verify(dest: Path) -> list[tuple[Path, int, str]]:
    """Return every surviving identity hit as (path, line number, line)."""
    hits: list[tuple[Path, int, str]] = []
    terms = [t.lower() for t in IDENTITY]
    for path in sorted(dest.rglob("*")):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            # Strip placeholder hosting URLs before matching: they name a service, not a user.
            low = PLACEHOLDER_URL.sub("", line).lower()
            for term in terms:
                if term in low:
                    hits.append((path.relative_to(dest), number, line.strip()[:120]))
                    break
    return hits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "dist" / "supplementary.zip")
    parser.add_argument("--stage-dir", type=Path, default=ROOT / "dist" / "supplementary")
    parser.add_argument("--no-demo", action="store_true",
                        help="Exclude docs/ rather than anonymising it.")
    parser.add_argument("--check-only", action="store_true",
                        help="Stage and scan, but write no archive.")
    args = parser.parse_args()

    print(f"staging into {args.stage_dir}")
    copied = stage(args.stage_dir, include_demo=not args.no_demo)
    print(f"  {len(copied)} files copied")
    changed = rewrite(args.stage_dir)
    print(f"  {changed} files rewritten")

    hits = verify(args.stage_dir)
    if hits:
        print(f"\nFAIL  {len(hits)} identity string(s) survived anonymisation:")
        for path, number, line in hits[:40]:
            print(f"  {path}:{number}: {line}")
        if len(hits) > 40:
            print(f"  ... and {len(hits) - 40} more")
        print("\nNo archive written. Fix these, or re-run with --no-demo, before submitting.")
        return 1

    print("\nok    no identity strings survive in the staged tree")
    if args.check_only:
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(args.stage_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(args.stage_dir))
    size = args.out.stat().st_size / 1e6
    print(f"ok    wrote {args.out} ({size:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
