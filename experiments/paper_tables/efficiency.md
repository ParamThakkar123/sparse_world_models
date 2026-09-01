# Efficiency — Sparse vs Dense (mean ± std over seeds 0/1/2)

Parameter efficiency is the durable win. FLOP advantage erodes with N (per-object heads scale with object count); wall-clock latency favors dense (one matmul vs per-object gating), so latency is reported for transparency, not as a claim.

| N | sparse params | dense params | param ratio | sparse FLOPs | dense FLOPs | FLOP ratio | sparse lat (ms) | dense lat (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | 6916 | 76809 | 11.1× | 39936 | 152576 | 3.8× | 0.19 ± 0.00 | 0.04 ± 0.00 |
| 5 | 8452 | 82959 | 9.8× | 81920 | 164864 | 2.0× | 0.20 ± 0.02 | 0.04 ± 0.00 |
| 8 | 10756 | 92184 | 8.6× | 167936 | 183296 | 1.1× | 0.20 ± 0.01 | 0.04 ± 0.00 |
