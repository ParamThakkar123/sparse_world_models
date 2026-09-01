"""The browser demo must run the same model the paper reports.

`docs/assets/model.js` is a hand port of the contact featurisation and the gate/delta
forward pass. A port can be wrong in ways that still look right on screen -- a transposed
weight matrix, a swapped feature block, a missing action clip -- and a demo that quietly
shows a different model than the paper would be a worse version of the problem this project
exists to document.

So the export writes a fixture of PyTorch outputs, and this test runs the JavaScript over
the same inputs under node and requires agreement.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ASSETS = Path("docs/assets")
FIXTURE = ASSETS / "parity_fixture.json"

# Written to a temp file and run with `node --input-type=module` so no build step, bundler or
# package.json is needed anywhere in this repo.
DRIVER = """
import { contactFeatures, SparseResidualModel } from '%(model_js)s';
import { readFileSync } from 'node:fs';

const bundle = JSON.parse(readFileSync('%(weights)s', 'utf8'));
const fixtures = JSON.parse(readFileSync('%(fixture)s', 'utf8'));

// Every checkpoint the page can load is checked, not just the first: the sandbox and the
// replay tab run different weights, so one of them passing proves nothing about the other.
const result = {};
for (const [name, weights] of Object.entries(bundle.models)) {
  const fixture = fixtures[name];
  const model = new SparseResidualModel(weights);
  const features = [];
  const probs = [];
  const deltas = [];
  for (let row = 0; row < fixture.state.length; row += 1) {
    const state = fixture.state[row];
    const action = fixture.action[row];
    features.push(contactFeatures(state, action, fixture.num_objects, weights.constants));
    const out = model.predict(state, action, fixture.num_objects);
    probs.push(out.probs);
    deltas.push(out.deltas);
  }
  result[name] = { features, probs, deltas };
}
process.stdout.write(JSON.stringify(result));
"""


def as_url(path: Path) -> str:
    """A file:// URI, not a bare path: node reads an import specifier like `E:/x/model.js`
    as a URL with scheme `e:` and refuses it on Windows."""
    return path.resolve().as_uri()


@pytest.fixture(scope="module")
def javascript_output(tmp_path_factory) -> dict:
    if shutil.which("node") is None:
        pytest.skip("node is not installed; the JS port cannot be checked")
    if not FIXTURE.exists():
        pytest.skip("run `python -m experiments.export_web_model` first")
    driver = tmp_path_factory.mktemp("web") / "driver.mjs"
    driver.write_text(DRIVER % {
        "model_js": as_url(ASSETS / "model.js"),
        # readFileSync takes a path, not a URI.
        "weights": (ASSETS / "model.json").resolve().as_posix(),
        "fixture": FIXTURE.resolve().as_posix(),
    }, encoding="utf-8")
    result = subprocess.run(["node", str(driver)], capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(f"node failed:\n{result.stderr}")
    return json.loads(result.stdout)


@pytest.fixture(scope="module")
def torch_output() -> dict:
    if not FIXTURE.exists():
        pytest.skip("run `python -m experiments.export_web_model` first")
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def max_difference(left, right) -> float:
    """Largest absolute elementwise difference between two nested lists of equal shape."""
    if isinstance(left, list):
        assert len(left) == len(right), f"shape mismatch: {len(left)} vs {len(right)}"
        return max((max_difference(a, b) for a, b in zip(left, right)), default=0.0)
    return abs(left - right)


def test_every_exported_model_is_checked(javascript_output, torch_output):
    """Both checkpoints the page can load must appear here. If the export starts shipping a
    third and the fixture does not cover it, this fails rather than passing silently."""
    assert set(javascript_output) == set(torch_output)
    assert javascript_output, "no models exported"


def test_features_match(javascript_output, torch_output):
    """The featurisation is where a port goes wrong silently -- 19 numbers per object in a
    fixed order, any two of which can be swapped without the model erroring."""
    for name in javascript_output:
        difference = max_difference(javascript_output[name]["features"],
                                    torch_output[name]["features"])
        assert difference < 1e-4, f"{name}: contact features differ by {difference}"


def test_gate_probabilities_match(javascript_output, torch_output):
    for name in javascript_output:
        difference = max_difference(javascript_output[name]["probs"],
                                    torch_output[name]["gate_probs"])
        assert difference < 1e-4, f"{name}: gate probabilities differ by {difference}"


def test_deltas_match(javascript_output, torch_output):
    for name in javascript_output:
        difference = max_difference(javascript_output[name]["deltas"],
                                    torch_output[name]["delta"])
        assert difference < 1e-4, f"{name}: predicted deltas differ by {difference}"


def test_hard_gate_decisions_match(javascript_output, torch_output):
    """The demo shows a binary gate, so what a viewer actually sees is the thresholded
    decision. Probabilities agreeing to 1e-4 does not by itself guarantee this near 0.5."""
    for name in javascript_output:
        for row, (ours, theirs) in enumerate(zip(javascript_output[name]["probs"],
                                                 torch_output[name]["gate_probs"])):
            for index, (a, b) in enumerate(zip(ours, theirs)):
                assert (a >= 0.5) == (b >= 0.5), (
                    f"{name} row {row} object {index}: gate decision differs ({a} vs {b})"
                )
