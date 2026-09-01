# Sparsity-weight ablation (3 obj, seed 0, standard 15-epoch config)

All rows share the seed-0 hard split and identical config; only `sparsity_weight` varies. Metrics are best-epoch validation values. The model is robust across weights; 0.2 is the main-run choice.

| sparsity weight | gate F1 | gate precision | gate recall | changed-obj L2 | overall pose L2 | unchanged L2 | best epoch |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0 | 0.873 | 0.941 | 0.814 | 0.389 | 0.146 | 0.0020 | 14 |
| 0.05 | 0.865 | 0.940 | 0.801 | 0.389 | 0.146 | 0.0020 | 14 |
| 0.2 | 0.849 | 0.938 | 0.776 | 0.388 | 0.145 | 0.0020 | 14 |
| 0.5 | 0.845 | 0.967 | 0.750 | 0.387 | 0.144 | 0.0006 | 14 |
| 1.0 | 0.815 | 0.965 | 0.705 | 0.385 | 0.143 | 0.0006 | 14 |
