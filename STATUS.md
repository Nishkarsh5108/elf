# Status: Santa 2026 (Elf Calculator)

_Last updated: 2026-11-21_

## Current best
* **Verified local best 133,772** (`out/public_merge.csv`, refreshed via E038 from 13 datasets + 7 notebook outputs; **LB 133,785** from E036 — saving daily submissions for solver wins below 133,772).
* LB context (2026-11-21): leader 128,330 (ryan), 2nd 128,766, 3rd 128,830, top-10 ~130.4k, best public 133,772.

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
* `cpp/elf2.cpp` next-gen engine (E039–E048): 5-engine `DeepSearch` (exact `<=4` MITM, `S01×S012`/`S2×S2`/`P1±P1`/`T±C` product-diff, 64-bit `T±C` factoring + 3-factor + S3-cofactor, target-diff/div wrappers, product-residual beam), minimal-perturbation peak-liveness scheduler (`eval_order`, bottleneck `fix_pressure_sa`, Kahn's + Belady fallback), hidden-base discovery + hub injection + live-pressure lookahead in `-construct` (with `-multi N`), E-graph + voted-hub `-pool` exporter, `-dag` compiler, `-lns`, and Soft-Peak Herfindahl DAG-SA (`-sa`).
* `cpp/bases.cpp` hidden-base discovery (pairwise gcd of offset targets + binomial significance test).
* `cpp/mitm_probe.cpp` exact depth<=7 reachability probe; `eda/` analyses; `scripts/` helpers.
* `kaggle/push_shards.py` + `kaggle/run_kernel.py`: private dataset + private CPU kernels, one shard each
  (smoke-tested; `dataset`, `launch NAME N "ARGS" [p_all.txt] [shards]`, `status`, `fetch`).

## Where our engine stands (ratio = our length / public best; lower is better)
| engine / setting | sample | ratio |
|---|---|---:|
| `elf.cpp` greedy `-w 10 -n 88 -nd 72 -p 2` | 100 puzzles (`every10`) | 1.239 |
| `elf.cpp` best `-w 30 -n 160 -nd 100 -la 12 -lalt 4 -lwidth 6 -p 1` | 100 puzzles (`every10`) | 1.145 (0 wins) |
| `elf2.cpp` `-construct 1 -t 11 -n 96` (E039) | `bench12` (`PUB=2,303`) | 1.162 (2,675) |
| `elf2.cpp` `-construct 1 -n 160 -k2 64 -la 12 -lalt 6` (E043b) | `bench12` (`PUB=2,303`) | 1.093 (prog 2,518 / raw DAG 2,476 = 1.075) |
| **`elf2.cpp` `-construct 1 -multi 4 -lsec 20` + `-sa` (E048)** | `bench12` (`PUB=2,303`) | **1.073** (prog **2,472**, raw DAG **2,463 = 1.069**, indep **1.042**) |

What moved vs public:
* `-pool` (algebraic E-graph rewrites + voted hubs) + 2-stage CP-SAT (`src/recombine.py`) found `elf2`'s first strict DAG reduction against `public_merge` (`scored_0002: 245 -> 244 opt`, E044/E044b/E044c/E045), though compiling `244` without spills remains blocked by 2 register spills across `PUB`'s 39 saturated `live==48` steps.
* Fast L3-cache `-sa` (`88/24/100k`) cuts `-26` instructions in 30 s on our own `E043b` constructions (`2,518 -> 2,492`, E047c), but `public_merge` itself is locally saturated under `k<=4` LNS/SA (0 wins on `every10` in E046 & E047d).

## Assessment
* Our constructive engine (`elf2.cpp`) has caught up dramatically from `1.145–1.162x` down to **`1.073x` overall and `1.042x` on 50–60b independent puzzles** from scratch, with `10/12` `bench12` puzzles compiling at zero spills.
* `public_merge` (`133,772`, already produced by ~1,100+ CPU-hours of konbu17 portfolio + weeks of community LNS/annealing like `goodpjw2008`) has `617/1000` puzzles at `max_peak <= 44` and `193/1000` pinned at `max_peak == 48`, with `45.7%` of all instructions in `sh3+` backbones (`fanout >= 3`). Breaking past `133,772` meaningfully toward `#1` (`128,330`) requires stronger constructive search + true deeper/GPU MITM (`S3`, RTX 4050) + multi-start constructive portfolio + global helper discovery/recombination rather than local `k<=4` tweaks on fixed `PUB` DAGs.

## Next steps (in order)
1. **True `S3` / GPU-accelerated Meet-in-the-Middle** (RTX 4050 6 GB / multi-thread CPU) to discover exact 6–7 op cones (`priv >= 6`: 1,503 targets in `133,772`) and novel `S2/S3` multi-target hubs (`k = 2..8` target subsets).
2. **Global constructive leapfrog on structured + independent families**: add explicit target pair/cluster chain search + multiplicative quotient chains (`scored_0577` +56, `scored_0030` +18) and portfolio diversity (`elf2 -construct -multi 8..16` -> CP-SAT recombination over fresh diverse runs without `PUB` cycle bloat) to break individual puzzles below `PUB`.
3. **Register-aware DAG optimization & E-graph refinement**: fix `E046` CP-SAT cycle bloat (`~303`/puzzle backward 1-op cycles) and improve `-dag` / SA compilation when `max_peak == 49..50` (`scored_0002 = 244`), then scale winning pipelines via `kaggle/push_shards.py` (private kernels) and submit verified scores `< 133,772`.
