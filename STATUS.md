# Status — Santa 2026 (Elf Calculator)

_Last updated: 2026-11-21_

## Current best
* **LB 133,860** (= local; there is no public/private split). File: `out/public_merge.csv` (not in git;
  regenerate as below). It is a verified per-puzzle merge of public program datasets; none of our own
  programs beat it yet.
* LB context (2026-11-21): leader 128,330 (ryan), 2nd 128,830, top-10 ~130.4k, best public ~133.86k.

## Reproduce
```bash
python -m venv .venv && .venv/Scripts/python -m pip install kaggle numpy pandas numba six ziglang
bash scripts/fetch_public.sh                       # public program datasets -> ext/
.venv/Scripts/python src/merge.py out/public_merge.csv ext data/sample_submission.csv
# own engine (C++20, built with zig's clang on Windows; g++ -O3 -std=c++20 works on Linux)
.venv/Scripts/python -m ziglang c++ -O3 -march=native -std=c++20 cpp/elf.cpp -o elf.exe
.venv/Scripts/python src/io_txt.py puzzles p.txt scored every10        # puzzle subset -> engine format
./elf.exe p.txt -t 11 -w 10 -n 88 -nd 72 -p 2 > out.txt                  # planner run
.venv/Scripts/python scripts/cmp.py out.txt                              # compare with public best
.venv/Scripts/python src/merge.py merged.csv out/public_merge.csv <engine csv>   # (merge wins)
```
Verification: `src/santa.py` (exact Python ints; reproduces the LB score of the sample submission and of
our submissions exactly).

## Code map
* `src/santa.py` verifier/IO, `src/merge.py` per-puzzle merge, `src/io_txt.py` engine text format.
* `cpp/elf.cpp` engine: planner (cheapest-target-first + optional lookahead), backward dictionary beam
  search, eager/virtual constant palette, optional Bloom S3 level + root MITM (`-b 33 -mitm 1`), root
  full-factor search (`-rf 1`), rematerialisation-aware register allocator (tries 4 schedule/seed modes),
  LNS mode (`-lns progs.txt -lsec S`) that re-plans windows of targets of an existing program.
* `cpp/bases.cpp` hidden-base discovery (pairwise gcd of offset targets + binomial significance test).
* `cpp/mitm_probe.cpp` exact depth<=7 reachability probe; `eda/` analyses; `scripts/` helpers.

## Where our engine stands (ratio = our length / public best, lower is better)
| setting | sample | ratio |
|---|---|---:|
| greedy `-w 10 -n 88 -nd 72 -p 2` | 100 puzzles (every 10th) | 1.239 |
| + lookahead `-la 6` | 20 puzzles | 1.214 |
| heavy `-w 30 -n 128 -la 12 -lalt 4 -lwidth 6` | 5 big puzzles | 1.074 |
| with discovered bases as helper targets | 104 base puzzles | 1.395 -> 1.330 |
| **best: `-w 30 -n 160 -nd 100 -la 12 -lalt 4 -lwidth 6 -p 1`** | 100 puzzles | 1.145 (0 wins) |
LNS from the public best (k<=4 targets re-planned, 20 s/puzzle): 0 improvements on 100 puzzles —
public programs are locally tight w.r.t. this planner.

## Assessment
Our engine is ~15% behind the public frontier on a representative sample and has no per-puzzle wins.
Public programs are planner output polished by many LNS passes (exchange, quartet, anneal, recombination);
those passes are saturated on the public frontier. The LB leader (128.3k, 4% ahead) must have a stronger
constructive method. Running public *code* is off the table for us (safety rule on executing downloaded
code); public *programs* are fine as data.

## Next steps (in order)
0. Refresh the public merge daily (`scripts/fetch_public.sh` + `src/merge.py`) � the only source of LB
   gains so far (133,928 -> 133,860).
1. Program-level beam search over planner states (generalises 1-step lookahead) — sharing between
   targets is where our planner loses (later targets ~7.3 vs public ~6.6 instr on independent puzzles).
2. Portfolio over palette mode / leaf counts / bases / seeds, run at scale (Kaggle CPU kernels), merge.
3. Second-level hidden multiplier discovery for structured puzzles (targets ~ m * base-product ± small).
4. Re-run LNS from our own (different) constructions, and refresh public merges daily
   (`scripts/fetch_public.sh`).
