# Dataset Generation Commands

Use these commands from the repository root to generate action-transition pairs.
Each generated `.npz` file stores the ground-truth tuple `(s_t, a_t, s_{t+1})`
explicitly under the keys `s_t`, `a_t`, and `s_t1`.


```bash
python -m experiments.generate_transitions --policy random --episodes 100 --max-steps 100 --output data/transitions/random_100ep.npz
python -m experiments.generate_transitions --policy scripted --episodes 100 --max-steps 100 --output data/transitions/scripted_100ep.npz
```
