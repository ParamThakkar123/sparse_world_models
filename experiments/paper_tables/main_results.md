# Main Results — Sparse vs Dense vs No-op (mean ± std over seeds 0/1/2)

**Primary metrics** (change detection + changed-object error). Overall/unchanged L2 are secondary (overall L2 is dominated by unchanged objects and flatters no-op as scenes get sparser).

| N | model | F1 | precision | recall | changed-obj L2 | overall L2 | unchanged L2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | sparse | 0.867 ± 0.021 | 0.953 ± 0.046 | 0.797 ± 0.018 | 0.359 ± 0.116 | 0.136 ± 0.046 | 0.002 ± 0.002 |
| 3 | dense | 0.546 ± 0.007 | 0.375 ± 0.006 | 1.000 ± 0.000 | 0.582 ± 0.139 | 0.347 ± 0.053 | 0.207 ± 0.004 |
| 3 | no_op | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.397 ± 0.117 | 0.149 ± 0.045 | 0.000 ± 0.000 |
| 5 | sparse | 0.802 ± 0.024 | 0.923 ± 0.024 | 0.709 ± 0.035 | 0.408 ± 0.024 | 0.101 ± 0.004 | 0.002 ± 0.001 |
| 5 | dense | 0.394 ± 0.006 | 0.245 ± 0.005 | 1.000 ± 0.000 | 0.609 ± 0.027 | 0.319 ± 0.007 | 0.225 ± 0.002 |
| 5 | no_op | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.440 ± 0.005 | 0.108 ± 0.001 | 0.000 ± 0.000 |
| 8 | sparse | 0.829 ± 0.008 | 0.966 ± 0.009 | 0.726 ± 0.008 | 0.398 ± 0.093 | 0.071 ± 0.016 | 0.001 ± 0.000 |
| 8 | dense | 0.299 ± 0.005 | 0.176 ± 0.003 | 1.000 ± 0.000 | 0.618 ± 0.086 | 0.329 ± 0.023 | 0.267 ± 0.016 |
| 8 | no_op | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.403 ± 0.093 | 0.071 ± 0.016 | 0.000 ± 0.000 |

### No-op margin on overall L2 (sparse advantage; shrinks as scenes get sparser)

| N | no-op minus sparse overall L2 | any seed no-op wins? |
| --- | --- | --- |
| 3 | 0.01311 ± 0.00132 | False |
| 5 | 0.00666 ± 0.00477 | False |
| 8 | 0.00034 ± 0.00023 | False |
