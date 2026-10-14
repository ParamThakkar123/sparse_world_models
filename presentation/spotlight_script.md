# Spotlight script: Modeling What Changes (WORLDS @ IROS 2026)

Generated from the notes pane of `WORLDS_IROS2026_spotlight.pptx` by `build_spotlight.py`. 6 slides, about 120 seconds.

## 1. Title (~10 s)

[~10 s] Hi, I'm Param. Our paper is about one simple idea for world models of manipulation
scenes: model what changes, and copy everything else.

## 2. Idea (~25 s)

[~25 s] When a robot nudges one block, one pose changes and everything else stays put. Yet
the standard recipe re-predicts every pose at every step, so it perturbs objects that never
moved.

Our model has a per-object gate that decides whether each object changes, and a delta head
whose residual is applied only where the gate fires. Gate off: the object is copied exactly.
The heads are shared, so one model runs at any object count.

## 3. See it (~15 s)

[~15 s] Here it is running on a held-out push. Dashed outlines are the truth. The dense model,
third panel, scatters every box, even the ones the pusher never touched. Ours moves the boxes
that were pushed and leaves the green one exactly where it was.

## 4. Results (~30 s)

[~30 s] Across all held-out rollouts, after 20 steps the dense model's error is three to six
times higher than ours, with about a tenth of the parameters. The reason is the objects that
did not move: their error drops from 0.21 to almost zero. And a model trained on one object
count keeps 99 percent of its detection quality on another.

## 5. Honest reading (~25 s)

[~25 s] Now the honest part. The win is change detection: it avoids corrupting the static
majority. It is not accurate dynamics on the movers. Even with a perfect gate, error on
moving objects sits at the do-nothing baseline, and overall the model only matches no-op.
So our advice: always report no-op, and split error into changed and unchanged objects.

## 6. Takeaway (~15 s)

[~15 s] So: model what changes. Detection works; predicting the push itself is the next thing
to fix, and the delta head is the lever. Code, checkpoints and a live
in-browser demo are at the QR code. Come find us at the poster. Thank you!

(Repository: https://github.com/ParamThakkar123/sparse_world_models)
