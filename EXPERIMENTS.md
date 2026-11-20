# Experiment log

Scores are total instructions (lower is better). "LB" = Kaggle leaderboard (identical to local; no split).

| id | date | what / why | config | local | LB | conclusion |
|---|---|---|---|---:|---:|---|
| E000 | 2026-11-21 | Sanity: score sample submission with own verifier | `src/santa.py` | 1,490,379 | – | verifier matches LB number for sample (24 LB teams sit exactly at 1,490,379) |
| E001 | 2026-11-21 | Baseline: per-puzzle merge of every public program CSV (datasets in `ext/`, fetched with `scripts/fetch_public.sh`) | `python src/merge.py out/public_merge.csv ext data/sample_submission.csv` | 133,928 | 133,928 | pipeline verified end-to-end; alexchilton/rudolph file is best on every puzzle |
| E002 | 2026-11-21 | Are independent targets short trees over the start values? | `eda/tree_reach.py practice_0005 6 3` | – | – | no target within 10 tree ops; independent targets are effectively random |
| E003 | 2026-11-21 | Depth<=5 MITM from register snapshot (previous target) in best program | `eda/snapshot_mitm.py scored_0001` | – | – | <=5 found only where public already used <=5; no gain at depth 5 (matches Asterios E05) |
| E004 | 2026-11-21 | Same with depth<=7 (C++ probe, S3 Bloom) | `cpp/mitm_probe.cpp`, `eda/make_snapshots.py` | – | – | standalone cost from snapshot is 6–7 or >7 where public spent 6–12 incl. shared helpers; deeper search alone is a modest lever |
