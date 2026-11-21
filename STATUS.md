# Status: Santa 2026 (Elf Calculator)

_Last updated: 2026-11-21_

## Current best
* **LB 133,860** (= local score; there is no public/private split). File: `out/public_merge.csv` (not in
  git; regenerate as below). It is a verified per-puzzle merge of public program datasets; none of our
  own programs beat it yet.
* LB context (2026-11-21): leader 128,330 (ryan), 2nd 128,830, top-10 ~130.4k, best public ~133.86k.

## Reproduce
```bash
python -m venv .venv && .venv/Scripts/python -m pip install kaggle numpy pandas numba six ziglang ortools
bash scripts/fetch_public.sh                       # public program datasets -> ext/
.venv/Scripts/python src/merge.py out/public_merge.csv ext data/sample_submission.csv
# own engine (C++20, built with zig's clang on Windows; g++ -O3 -std=c++20 -pthread works on Linux)
.venv/Scripts/python -m ziglang c++ -O3 -march=native -std=c++20 cpp/elf.cpp -o elf.exe
.venv/Scripts/python src/io_txt.py puzzles p.txt scored every10        # puzzle subset -> engine format
./elf.exe p.txt -t 11 -w 30 -n 160 -nd 100 -la 12 -lalt 4 -lwidth 6 -p 1 > out.txt   # best config
.venv/Scripts/python scripts/cmp.py out.txt                              # compare with public best
.venv/Scripts/python src/merge.py merged.csv out/public_merge.csv out.txt   # merge any wins (verified)
```
Verification: `src/santa.py` (exact Python ints); it reproduces the LB score of the sample submission and
of both our submissions exactly.

## Code map
* `src/santa.py` verifier/IO, `src/merge.py` per-puzzle merge (accepts CSVs and engine outputs),
  `src/io_txt.py` engine text format, `src/recombine.py` CP-SAT recipe recombination -> DAG file.
* `cpp/elf.cpp` engine:
  * planner: cheapest-target-first (`-la 1`), one-step lookahead (`-la K -lalt A -lwidth W -lw w`, with an
    incremental dictionary), program-level beam (`-B b`);
  * per-target search: backward dictionary beam (`-w`, leaves `-n/-nd`), optional Bloom S3 level and root
    MITM (`-b 33 -mitm 1`), root full-factor search (`-rf 1`);
  * constants palette: `-p 2` eager/free (good for greedy), `-p 1` virtual/charged (good with lookahead);
  * register allocator: rematerialisation-aware Belady, tries 4 schedule/seed modes, `-reorder N`
    hill-climbs target order to remove spills; reproduces every public program's length exactly;
  * modes: `-lns progs.txt -lsec S -lk K` (window LNS), `-dag file` (compile a DAG).
* `cpp/bases.cpp` hidden-base discovery (pairwise gcd of offset targets + binomial significance test).
* `cpp/mitm_probe.cpp` exact depth<=7 reachability probe; `eda/` analyses; `scripts/` helpers.
* `kaggle/push_shards.py` + `kaggle/run_kernel.py`: private dataset + private CPU kernels, one shard each
  (smoke-tested; `dataset`, `launch NAME N "ARGS" [p_all.txt] [shards]`, `status`, `fetch`).

## Where our engine stands (ratio = our length / public best; lower is better)
| setting | sample | ratio |
|---|---|---:|
| greedy `-w 10 -n 88 -nd 72 -p 2` | 100 puzzles (every 10th) | 1.239 |
| + lookahead `-la 6` | 20 puzzles | 1.214 |
| with discovered bases as helper targets | 104 base puzzles | 1.395 -> 1.330 |
| **best: `-w 30 -n 160 -nd 100 -la 12 -lalt 4 -lwidth 6 -p 1`** | 100 puzzles | **1.145** (0 wins) |

Improvement passes on the public best found nothing: window LNS with our planner (k<=4 targets, 20 s per
puzzle, 100 puzzles) and CP-SAT recombination of public + own programs (with and without a pool of all
1-op recipes between pooled values) both return the public programs unchanged.

## Assessment
Our engine is ~15% behind the public frontier on a representative sample and has no per-puzzle wins.
Public programs are planner output polished by many LNS passes (exchange, quartet, anneal,
recombination), and those passes are saturated on the public frontier. The LB leader (128.3k, 4% ahead
of public) must have a substantially stronger constructive method. Running public *code* is off the
table for us (safety rule on executing downloaded code); public *programs* are fine as data.

## Next steps (in order)
0. Refresh the public merge daily (`scripts/fetch_public.sh` + `src/merge.py`); so far the only source
   of LB gains (133,928 -> 133,860).
1. Structured puzzles (742, ~87k instructions): plan *cluster chains* (order targets so that each next
   target is 1-2 ops from the previous using cheap differences built from start values) and discover
   second-level hidden multipliers (targets ~ m * base-product +- small). This is where public beats us
   most (e.g. scored_0127: 96 vs our 116 with bases).
2. Independent puzzles (242, ~47k): our later targets cost ~7.3 vs public ~6.6 instructions; better
   sharing via deeper lookahead / beam with a better remaining-cost estimate.
3. Once the engine produces wins, run it at scale with `kaggle/push_shards.py` and merge.
