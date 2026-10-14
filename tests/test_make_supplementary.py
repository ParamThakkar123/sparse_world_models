"""Pin the anonymisation of the supplementary archive.

The failure mode this guards is not a crash but a *silent pass*: a packager that reports
success while an author name is still in the tree is worse than no packager at all. These
tests therefore check that the verifier actually fails on planted identity strings, that the
rewrites reach the files that historically leaked (extensionless `LICENSE`, the demo's author
card), and that placeholder hosting URLs do not trigger a false alarm.
"""

from __future__ import annotations

from pathlib import Path

from experiments import make_supplementary as ms


def _tree(root: Path, files: dict[str, str]) -> None:
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def test_verifier_catches_a_planted_name(tmp_path):
    _tree(tmp_path, {"notes.md": "written by Param Thakkar\n"})
    hits = ms.verify(tmp_path)
    assert hits, "verifier passed a tree containing an author name"
    assert hits[0][0] == Path("notes.md")


def test_verifier_ignores_placeholder_hosting_urls(tmp_path):
    _tree(tmp_path, {
        "docs.md": "falls back to https://huggingface.co/datasets/<repo>/resolve/main/x.json\n",
        "code.py": 'base = f"https://huggingface.co/datasets/{args.repo}/resolve/main"\n',
    })
    assert ms.verify(tmp_path) == []


def test_rewrites_reach_extensionless_license(tmp_path):
    """LICENSE has no suffix; an earlier version skipped it and leaked the copyright holder."""
    _tree(tmp_path, {"LICENSE": "MIT License\n\nCopyright (c) 2026 Param Thakkar\n"})
    ms.rewrite(tmp_path)
    text = (tmp_path / "LICENSE").read_text(encoding="utf-8")
    assert "Thakkar" not in text
    assert "Anonymous authors" in text
    assert ms.verify(tmp_path) == []


def test_demo_author_card_and_links_are_scrubbed(tmp_path):
    html = (
        '<p class="eyebrow">Accepted WORLDS Workshop, IROS 2026</p>\n'
        '<h2>Modeling What Changes: Sparse, Residual World Models for '
        "Object-Centric Manipulation</h2>\n"
        '<ol class="authors">\n'
        '  <li class="author"><strong>Param Thakkar</strong>'
        '<a href="mailto:puthakkar_b22@ce.vjti.ac.in">mail</a></li>\n'
        "</ol>\n"
        '<a href="https://github.com/ParamThakkar123/sparse_world_models">Code</a>\n'
        '<pre id="bibtex"><code>@inproceedings{thakkar2026modeling}</code></pre>\n'
    )
    _tree(tmp_path, {"docs/index.html": html})
    ms.rewrite(tmp_path)
    out = (tmp_path / "docs" / "index.html").read_text(encoding="utf-8")

    assert "Anonymous authors" in out
    assert "Change-Detection Benchmarks" in out          # workshop title replaced, not kept
    assert ms.verify(tmp_path) == []
    # Structure survives the block rewrites -- a broken demo helps nobody.
    assert out.count("<ol class=\"authors\">") == 1 and out.count("</ol>") == 1
    assert '<pre id="bibtex">' in out and "</code></pre>" in out


def test_workshop_venue_is_removed_but_ordinary_prose_is_not(tmp_path):
    _tree(tmp_path, {
        "README.md": "Accepted at WORLDS @ IROS 2026.\n",
        # The ledger quotes a retired claim using this phrase; it is not the paper's title.
        "RESULTS.md": 'the claim "Explicitly modeling what changes -- a gate" is retired\n',
    })
    ms.rewrite(tmp_path)
    assert ms.verify(tmp_path) == []
    assert "modeling what changes" in (tmp_path / "RESULTS.md").read_text(encoding="utf-8")


def test_excluded_paths_never_stage(tmp_path, monkeypatch):
    monkeypatch.setattr(ms, "ROOT", tmp_path)
    _tree(tmp_path, {
        "keep.py": "x = 1\n",
        "IEEE_Conference_Template/paper.tex": "non-anonymous workshop paper\n",
        "SUBMISSION_RUNBOOK.md": "internal\n",
        "experiments/make_supplementary.py": "self\n",
        "tests/test_make_supplementary.py": "plants names on purpose\n",
        "experiments/runs/r/summary.json": "{}\n",
        "experiments/runs/r/config.json": "{}\n",
        "experiments/runs/r/metrics.csv": "a,b\n",
        "data/transitions/big.npz": "binary\n",
    })
    copied = ms.stage(tmp_path / "out", include_demo=True)
    names = {p.as_posix() for p in copied}
    assert "keep.py" in names
    for forbidden in ("IEEE_Conference_Template/paper.tex", "SUBMISSION_RUNBOOK.md",
                      "experiments/make_supplementary.py",
                      "tests/test_make_supplementary.py",
                      "experiments/runs/r/metrics.csv",
                      "data/transitions/big.npz"):
        assert forbidden not in names, f"{forbidden} was staged into the archive"
    # Run provenance DOES ship: these two files are what every claim in the paper points at,
    # and they are ~1 MB in total. Only the per-run bulk is dropped.
    for required in ("experiments/runs/r/summary.json", "experiments/runs/r/config.json"):
        assert required in names, f"{required} should ship as provenance"
