# Santa 2026 — research notes

Competition: https://www.kaggle.com/competitions/santa-2026 (Nov 18 2026 – Jan 28 2027, 100 subs/day,
no public/private split: the LB score is exactly the local score).

## Problem in one paragraph
1000 puzzles. Each: 1–5 small starting values in registers 0..k-1 of a 48-register machine, ~8–32 large
targets (up to 1e18), a `value_limit` L. An instruction `[op,i,j,d]` (op in + - * /) writes `reg[i] op reg[j]`
to `reg[d]`; every result must be a positive integer <= L (subtraction must stay positive, division exact).
A target is covered once it appears in any register. Score = total instructions over all programs (lower
is better). Trivial lower bound = number of new targets = **26,898**. Sample submission = 1,490,379.

## EDA findings (eda/*.py)
* `independent` family == puzzles whose largest starting value is >= 100 (45/45 practice; **242 scored**).
  Targets behave like uniform random numbers in roughly [L/10, L]: an exhaustive tree search (eda/tree_reach.py)
  shows none of them is reachable from the starting values in <= 10 tree ops, and the reachable-set
  counting says this is what random targets would look like. No hidden program to recover.
* `shared` / `multiplicative` / `mixed` (742 scored, labels hidden): targets are built from a few hidden
  "bases" (e.g. scored_0127: 13541, 33386, 47727) — products of bases, plus small offsets, times small
  factors. Clusters of near-duplicates (gaps of a few hundred) are common. Example practice_0006:
  hidden 572, 639; targets 572^2-4, 572*639, 639^2, 2*572^2, 17*572^2, 70*572^2.
* `tiny`: 16 scored puzzles with L < 100, trivial.
* Public best (133,928) cost split (eda/cost_breakdown.py):
  | group | puzzles | targets | instr | per target |
  |---|---:|---:|---:|---:|
  | independent | 242 | 6,809 | 47,070 | 6.91 |
  | other (shared/mult/mixed) | 742 | 20,048 | 86,801 | 4.33 |
  | tiny | 16 | 41 | 57 | 1.39 |
  Independent per-target cost rises with size: L~1e12: 5.75, ~1e15: 7.19, ~1e17: 8.61.
* The first target of every puzzle costs ~20+ instructions (growing the few small starting values into a
  useful register set); ~21.5k instructions total go there (from Asterios' analysis, consistent with ours).

## Reachability / information argument (eda/snapshot_mitm.py, cpp/mitm_probe.cpp)
From a register set R of size n, values reachable by exactly-k-op trees: S1 ~ 2n^2, S2 ~ S1*n*5,
S3 ~ S2*n*5 + S1^2*2.5 ... With n~40 live values: S1~3e3, S2~2.5e5, S3~4e7.
Probe on scored_0001 (registers right after each previous target in the best public program):
the exact min tree cost of the next target from those registers alone is <=5 only when the public
program also used <=5; typical answers are 6, 7 or >7, while the public program spent 6–12 *including*
helpers it shares with later targets. Conclusion: per-target deeper search alone is a modest lever
(maybe ~0.5–1 op/target on random targets); planning (sharing, bootstrap, register use) matters as much.

## Public work (as of 2026-11-20/21)
| source | core idea | score | leaves on the table |
|---|---|---:|---|
| Tony Li, *Elf Calculator With Register Reuse* | per-target backward beam: t = q*f ± r with f,r from a dictionary of 0/1/2-op values over ~56 leaves; Belady-like register allocator with rebuilds | 191,867 | greedy target order, tiny leaf set |
| konbu17, *Wider Dictionary + Portfolio* (+dataset) | more leaves (72/88), 50-setting portfolio, register-aware planner "cheapest target first", plan with 128 virtual registers then squeeze to 48 (<100 extra instr total), 1-step lookahead (own + 0.4*remaining), recombination of programs via integer programming | 142,940 | planner code not published; >90 leaves hurt because of evictions |
| thisray, Elias, Ertugrul, IceFire, ... | variants of the same beam + portfolios | 138–175k | |
| Nilanjan_Saha, Bikram Basnet | graph rewrites, exact MILP re-pricing, suffix rebuilder | small gains | local |
| Atakan Aldemir, *Joint Recipes* | cut a shared helper together with all consumers, re-price jointly; multiplier-frontier optimizer | ~tens of instr | |
| goodpjw2008, *DAG-LNS + CP-SAT recombine* | LNS kicks + recombination | 133,945 | |
| Asterios Panoras, *Best public + fails* | merge of all public outputs + 17 documented negative experiments (exact depth<=5 re-synthesis, suffix replanning, joint cuts: ~0 gain) | 134,185 | |
| E-Max AI (royalacecat) | merges + LNS archives | 133,928 | |
Literature pointers (from the forum): Voronenko & Püschel Hcub (multiple constant multiplication,
choose intermediates by cumulative benefit to all targets), Aksoy et al. exact MCM search, Massalin /
STOKE superoptimization, Hutton's Countdown pruning rules, Bos–Coster addition sequences.

**Frontier**: best public merge 133,928 (verified, = our LB). LB leader (ryan) 128,330 on 2026-11-20;
top-10 ~130.5k. Many people grind LNS on the same public programs for ~10s of instructions/hour.

## Where the headroom probably is
1. Independent family (47k): random targets -> per-target search depth/quality + a register set chosen
   to maximise reach. A 50-bit target needs ~6 ops in principle from 40 good registers
   (|S6| ~ 2^52) vs ~7 in public programs.
2. Bootstrap: first target ~20 instr x 1000 puzzles.
3. Structured families: global choice of hidden bases / shared helpers (Hcub-style benefit) rather
   than target-by-target greed.

## Findings from building our own engine (2026-11-21)
* Reproduced the public dictionary-beam planner from scratch (`cpp/elf.cpp`); Tony Li's settings give
  1.45x public best, konbu17-style settings ~1.24x, our best so far (virtual palette + 160 leaves +
  12-candidate lookahead) ~1.05x on 5 big puzzles.
* **Register allocation matters for wide planning.** Plans built from 128–200 leaves are 3–5% smaller,
  but a naive Belady allocator gave it all back (cascading rebuilds of evicted chains). Fixes that
  worked: emit each target's cone depth-first just before use; let a dying source register receive the
  result; cost-aware eviction; when a live value is evicted, add pseudo-uses for its inputs
  (rematerialisation-aware Belady); allow evicting start values once nothing can need them. With plan
  order + evictable seeds the allocator reproduces every public program's length exactly.
* **Constants palette**: making 1, 2^k and small odd constants free leaves (Tony Li) helps greedy
  planning on independent puzzles (they are a reduction basis), but with lookahead the planner is much
  better when constants are charged at their real build cost (virtual palette): 935 -> 893 on 5 big
  puzzles.
* Deeper per-target search (Bloom S3 level, exact 6-op MITM, 2-op factors at the root) changes totals
  by <3%; per-target search is not the bottleneck. Sharing across targets is.
* **Hidden bases** can be recovered statistically from the targets alone (pairwise gcd of slightly
  offset targets with a multiple-testing-corrected binomial test); 104 scored puzzles have them. Seeding
  them as helper targets improves our planner ~5% on those puzzles, but public programs are still ~20%
  better there because they also exploit cluster chains (each next target 1–2 ops from the previous one
  using cheap differences derived from the start values) and second-level multipliers.
* LNS that removes k<=4 targets and re-plans them with our planner finds nothing on public programs
  (locally tight). Public LNS tools are saturated on the public frontier (goodpjw2008's history).

## Structured puzzles look near-optimal in public (2026-11-21, E030-E032)
On scored_0127 the public program's first 25 instructions reproduce what looks like the generator's own
chain (17=7+10, 289=17^2, 271=289-18, 260=271-11, 586=260+326, 521=586-65, 847=326+521, 13552=16*847,
13541=13552-11, ...); those intermediates then reappear inside targets as multipliers/offsets
(t = m*(b_i*b_j) + d with m in {65, 260, 521, 1043, 16672, ...}). Later targets cost 2.3 instructions on
average. CP-SAT over a pool of 47k structure-derived recipes (hinted with public) cannot shorten it.
Working hypothesis: on structured puzzles public is close to the hidden program; the leaderboard gap
(leader 4% ahead) is more likely on independent (random-target) puzzles.

## Deep EDA & Forum Synthesis (2026-11-21)
* **`konbu17` (142,940) vs `public_merge` (133,785) by predicted family** (using Atlas predictions):
  * `independent` (249): 47,974 -> 47,273 (-701, -1.46%); helper Jaccard overlap 0.712 (208/249 > 0.5); 171/249 barely improved (`pub >= konbu17 - 3`, 32,157 instr).
  * `mixed` (204): 33,858 -> 32,405 (-1,453, -4.29%); helper Jaccard 0.494; 91/204 barely improved (13,723 instr).
  * `shared` (295): 37,516 -> 32,003 (-5,513, -14.70%); helper Jaccard 0.250; 66/295 barely improved (6,216 instr).
  * `multiplicative` (235): 23,513 -> 22,025 (-1,488, -6.33%); helper Jaccard 0.266; 49/235 barely improved (3,740 instr).
  * `tiny` (17): 79 -> 79 (0.00%).
* **Why `mixed` is half-hard**: On `practice_puzzles.csv` (`hard` tier), `mixed` puzzles have median 48% of their targets in `[L/10, L]` (uniform random range), vs 4% in `shared` and 4% in `multiplicative`. Thus `mixed` puzzles are roughly half structured + half independent random targets.
* **Independent scale wall**: Across bit widths (`<=35b`: 4.22/t, `36-44b`: 5.74/t, `45-53b`: 7.19/t, `54-61b`: 8.61/t), public holds roughly constant ~6.8-7.0 bits/op. Register snapshots in `scored_0001` (`Lbits=50.4`) concentrate 30/48 registers in `[0.4L, 0.55L]` (20-26b mid-scale factors), while `scored_0002` (`Lbits=59.8`) has 24/48 registers `<0.25L` and 14/48 `>0.8L` (completed targets) late in the run. Peak live registers hit `>=45` in 201/249 `independent` and 140/204 `mixed` public programs (avg `max_live` 44.0 and 42.2).
* **Why existing search engines miss random-target improvements**:
  1. `cpp/elf.cpp` uses `q*a ± r` with `r = T % a < a` (`a` in `S0∪S1`), missing `T = A*B ± C` when `C > A`, `A,B ∈ S2`, `T ± C` exact 64-bit factorizations (`A*B` where `A ∈ S3` or 3-factor products), and `k*T = A*B ± C`; it also mixes large completed targets and 28 mostly unused palette constants into `S1`/`S2` leaves.
  2. Public LNS (`elfopt`, `quartet`) only searches exact depth `<=4` + a shallow depth-5 window, too shallow to resynthesize 6-8 op independent targets or synthesize new shared helpers for 2-4 random targets.
* **Forum insights (2026-11-21)**:
  * `hengck23` (thread 760095): 66% of 158k->146k gains from deeper per-target steps on fixed helpers, 15% from tearing out target groups + building a new common helper, 12% from cheaper helper synthesis; Georgy Mamarin noted unconstrained (>48 reg) plans are ~9% shorter.
  * `hh hhhokk` (760326): `D_48` (deciding `<= k` instructions with 48 registers) is NP-complete via grid Hamilton path.
  * `Georgy Mamarin` (760300, 760285): Hcub cumulative-benefit intermediate selection, Aksoy DFS pruning, STOKE MCMC uphill moves, Hutton Countdown symmetry pruning; `independent` = `max(start) >= 100`.
  * `Santiago Zazo` (760674): register liveness visualization, backward DCE, unlabeled smoothness + excess hit-rate detector.
  * `Farrukh Hussain` (760688): public frontier (~134.6k) relies on Rust polish, joint recipes, MITM, and `<=6`-step chain cutting; first target averages ~17.75 instr.
* **Deep DAG & Search Diagnostics on `133,772` & `elf2` (E038–E040)**:
  * Across all 1,000 `public_merge` programs (`133,772` instr): **private target cones = 55,359 (41.4%)**, `sh2` (shared by 2 targets) = 17,305 (12.9%), `sh3+` (shared backbone `>= 3` targets) = **61,108 (45.7%)**. Only 1,503 / 26,898 targets have `priv >= 6` (546 with `priv >= 7`, 95 with `priv >= 8`). Thus small `k <= 4` LNS only touches local cones while global `sh3+` backbone selection requires large-window moves (`k >= 5`) or global CP-SAT recombination.
  * On `PUB`'s hard 50–60b targets (`priv = 6..8` in `scored_0001`, `0002`, `0808`), **pairwise large-value sums/diffs `U1 ± U2`** and **full-pool `A*B`** slash the residual by 8–19 bits (e.g., `60b -> 41–47b`, `49b -> 34b`). Exiling `> L^0.78` nodes and large targets from `PoolDict` leaves blocked these recipes in `E039`/`E040`.
  * `PUB` interleaves target cones to stay at `peak = 48` (e.g., `scored_0002` spends 118/245 steps at peak `>= 46`), causing greedy `schedule_min_live` to fail `0/32` on all 6 hard 50–60b `PUB` DAGs and triggering `rejp = 115` in `E040`; SA / topological insertion from `pref_ord` is needed for zero-spill scheduling.
  * In `E039` (`-construct`), blind constant lookahead fallback on 50–60b targets and `build_binary` fallback on 2-start 60b puzzles (`scored_0808` `t0 = 84`) crippled constructive synthesis.
