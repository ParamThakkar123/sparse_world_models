# Dataset Generation Commands

Use these commands from the repository root to generate action-transition pairs.
Each generated `.npz` file stores the ground-truth tuple `(s_t, a_t, s_{t+1})`
explicitly under the keys `s_t`, `a_t`, and `s_t1`.

```bash
python -m experiments.generate_transitions --policy random --episodes 100 --max-steps 100 --output data/transitions/random_100ep.npz
python -m experiments.generate_transitions --policy scripted --episodes 100 --max-steps 100 --output data/transitions/scripted_100ep.npz
```

## The four audited benchmarks

Each script generates its episodes, builds **episode-disjoint** splits, trains the gates, and
runs the trivial-rule battery. They are the exact commands behind the paper's Table 1; run them
in this order, and **not concurrently** — the data generation is memory-hungry and running two
of these at once is what killed several earlier attempts on a 16 GB machine.

```bash
# 1. motion + onset benchmarks, tabletop (MuJoCo), 3 counts x 5 seeds
bash experiments/onset_benchmark_pipeline.sh "3 5 8" "0 1 2 3 4"
python -m experiments.onset_shortcut_audit --benchmarks motion onset \
    --counts 3 5 8 --seeds 0 1 2 3 4 --run-name onset_shortcut_audit_5seed

# 2. interaction benchmark, billiards (Box2D): indirect, chain-driven onset
python -m experiments.build_interaction_benchmark --env billiards --counts 5 8 --seeds 0 1 2
bash experiments/interaction_pipeline.sh billiards "5 8" "0 1 2"

# 3. clutter-onset benchmark, clutter (Chipmunk2D): dense packing + the onset filter.
#    12000 episodes/cell is not optional -- onset events are ~0.065% of steps here, and at
#    1500 episodes the splits are too thin to train on. Takes a few hours.
bash experiments/clutter_onset_rebuild.sh

# 4. the five published baselines, on both tabletop benchmarks
python -m experiments.literature_baselines --counts 3 5 8 --seeds 0 1 2 \
    --run-name literature_baselines_motion
python -m experiments.literature_baselines --counts 3 5 8 --seeds 0 1 2 \
    --split-template "data/transitions/splits_onset_{n}obj_s{seed}/onset_{n}obj_s{seed}_hard_{split}.npz" \
    --run-name literature_baselines_onset
```

## Rebuilding the paper from those runs

```bash
python -m experiments.make_iclr_tables   # tables + figure, straight from experiments/runs/
python paper/claims.py                   # every number in main.tex, re-derived and checked
python paper/preflight.py                # anonymity, page count, style-file integrity
```

## Auditing a benchmark that is not ours

No part of the pipeline above is required. The battery takes one `.npz` per split:

```bash
python -m experiments.audit_battery \
    --test mybench_test.npz --val mybench_val.npz --train mybench_train.npz \
    --model-predictions my_model_test_preds.npz \
    --report audit_report.md --json audit_report.json
```

Rules whose inputs are missing from your arrays are reported as **skipped**, never silently
passed. `--extra-rules` takes a Python file defining
`rules(data, radius, rest_speed) -> dict`, which is how you add the rules invented against
*your* benchmark — the step this project's own history says is the one that matters.
