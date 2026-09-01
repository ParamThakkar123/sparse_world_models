# Main Results — Sparse vs Dense vs No-op (mean ± std over seeds 0/1/2)

**Primary metrics** (change detection + changed-object error). Overall/unchanged L2 are secondary (overall L2 is dominated by unchanged objects and flatters no-op as scenes get sparser).

| N | model | F1 | precision | recall | changed-obj L2 | overall L2 | unchanged L2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | sparse | 0.849 ± 0.040 | 0.928 ± 0.048 | 0.782 ± 0.036 | 0.436 ± 0.082 | 0.185 ± 0.046 | 0.003 ± 0.002 |
| 3 | dense | 0.589 ± 0.028 | 0.418 ± 0.028 | 1.000 ± 0.000 | 0.604 ± 0.093 | 0.364 ± 0.054 | 0.190 ± 0.014 |
| 3 | no_op | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.459 ± 0.071 | 0.193 ± 0.041 | 0.000 ± 0.000 |
| 5 | sparse | 0.815 ± 0.011 | 0.951 ± 0.032 | 0.714 ± 0.005 | 0.371 ± 0.055 | 0.113 ± 0.013 | 0.001 ± 0.001 |
| 5 | dense | 0.466 ± 0.015 | 0.304 ± 0.013 | 1.000 ± 0.000 | 0.585 ± 0.052 | 0.333 ± 0.014 | 0.223 ± 0.009 |
| 5 | no_op | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.397 ± 0.045 | 0.120 ± 0.010 | 0.000 ± 0.000 |
| 8 | sparse | 0.803 ± 0.021 | 0.959 ± 0.014 | 0.691 ± 0.025 | 0.397 ± 0.079 | 0.084 ± 0.017 | 0.001 ± 0.000 |
| 8 | dense | 0.349 ± 0.023 | 0.211 ± 0.017 | 1.000 ± 0.000 | 0.644 ± 0.077 | 0.369 ± 0.025 | 0.296 ± 0.014 |
| 8 | no_op | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.420 ± 0.090 | 0.089 ± 0.019 | 0.000 ± 0.000 |

### No-op margin on overall L2 (sparse advantage; shrinks as scenes get sparser)

| N | no-op minus sparse overall L2 | any seed no-op wins? |
| --- | --- | --- |
| 3 | 0.00830 ± 0.00511 | False |
| 5 | 0.00736 ± 0.00346 | False |
| 8 | 0.00422 ± 0.00204 | False |
