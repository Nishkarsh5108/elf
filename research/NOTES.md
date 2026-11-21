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

## Register Saturation, E-Graph + CP-SAT, & Engine v2 Milestones (2026-11-21, E041–E048)
* **1,000-puzzle `public_merge` (`133,772`) family & register-peak anatomy**:
  * Atlas predicted families across all 1,000 scored puzzles: `independent` (249, `47,272` ops, `6.88`/t, `6.76` b/op), `mixed` (204, `32,404` ops, `5.78`/t, `7.27` b/op), `shared` (295, `31,996` ops, `4.02`/t, `8.99` b/op), `multiplicative` (235, `22,021` ops, `3.44`/t, `6.93` b/op), `tiny` (17, `79` ops).
  * Exact `max_peak` live-register distribution across `133,772`: **`617` puzzles `<= 44`**, `56` at `45`, `54` at `46`, `80` at `47`, **`193` touching `48`** (`0 > 48`). Thus 80% of puzzles have register headroom, while the hardest 19% (like `scored_0002`, with `live == 48` across steps `80..178` and 39 steps at `48`) reject new recipes if they extend lifetimes of early-dying intermediates across the saturated plateau.
  * Worst non-independent targets (`scored_0493`, `0518`, `0054`, `0819`, `0415`) show repeated medium primes and strong continued-fraction rational ratios `T[j]/T[i] ~ p/q` (saving 12–34 bits), many of which `PUB` already exploits directly (`25/1`, `31/1`, `502/1`, `6048/1`), while `konbu17 -> pub` left 110/249 `independent`, 57/204 `mixed`, 41/295 `shared`, and 28/235 `multiplicative` puzzles barely moved (`diff >= -2`).
* **E-Graph + CP-SAT findings (`E044`–`E046`)**:
  * Diffing public frontier improvements (`scored_0279`) showed simultaneous **additive reassociation + factor-shifting** (`A*B -> (A/g)*(B*g)`); encoding Rules 1–6 in `-pool` immediately discovered `scored_0002: 245 -> 244 (opt)` (E044/E044b/E044c: T[10] dropped `8 -> 7` ops).
  * However: (1) un-tie-broken CP-SAT churns ~80 defs cleanly breaking delicate `48`-register schedules (`24` spills in E044); hint-preserving 2-stage CP-SAT (`<= pub_n` + `PUB` hints, then `ClearHints` + barrier-penalized tie-break) fixes all neutral churn (`chg=0`, `0` spills on 11/12 `bench12` and 100/100 `every10`); (2) OR-Tools CP-SAT loses >6x–10x speed when `num_workers` drops from 8 to 2 (`2.58 s -> >15 s` on `0122`); (3) adding 13 extra CSVs + ~303/puzzle backward 1-op cycles among existing `pub_vals` caused `75/100` timeouts at `10 s` in E046.
* **Fast Soft-Peak Herfindahl DAG-SA (`-sa`) & Constructive Leapfrog (`E047`–`E048`)**:
  * Shrinking SA dicts into L3 cache (`88/24/100k` `pdict` + `64/16/60k` `wdict`) raised SA throughput from `4–10 it/s` (`E047`, 66% time in 64 MB dict rebuilds) to `16–24 it/s` (`E047b`), cutting `-26` ops on `E043b` constructions in 30 s (`2,518 -> 2,492`, `scored_0819: -10`), though `public_merge` stays locally tight at 12–60 s (`E047b/d`).
  * Combining near-divisibility lookahead (`x % a` within `+-64`), fast pairwise/offset GCD hidden-base seeding, multi-target hub injection, live-pressure lookahead penalty (`age > 42`), `-multi 4`, `20k` SA compile, and `20 s` `-sa` (`E048`) pushed `elf2 -construct` on `bench12` down from `2,675 (1.162x)` to **`2,472 (1.073x PUB, raw DAG 1.069x, indep 1.042x)`** with only `9` total spills across 12 puzzles (`10/12` at zero spills).

## Forensics of `133,772 -> 132,634` (`-1,138` ops) & Frontier Methodology Audit (2026-11-21, E049)
* **Family & bit-width breakdown of `133,772 -> 132,634` (`570/1000` puzzles improved)**:
  * `shared` (295): `31,996 -> 31,582` (`-414`, `-1.29%`, `200/295` improved).
  * `multiplicative` (235): `22,021 -> 21,696` (`-325`, `-1.48%`, `150/235` improved).
  * `independent` (249): `47,272 -> 47,046` (`-226`, `-0.48%`, `115/249` improved — incl. `scored_0137` `54->49 (-5)`, `scored_0253` `78->74 (-4)`).
  * `mixed` (204): `32,404 -> 32,232` (`-172`, `-0.53%`, `104/204` improved).
  * `tiny` (17): `79 -> 78` (`-1`, `1/17`).
  * By `value_limit` bit-width: `<=35b` (`-155`, `-2.06%`), `36-44b` (`-273`, `-0.87%`), `45-53b` (`-351`, `-0.80%`), `54-61b` (`-359`, `-0.70%`).
* **Where the `-1,138` ops actually came from structurally (crucial!)**:
  * Across all `570` improved puzzles (`73,847 -> 72,709` ops): **`sh3+` dropped by `-1,372`** (`33,963 -> 32,591`) and **`sh2` dropped by `-341`** (`9,254 -> 8,913`), while **`priv` rose `+575`** (`30,630 -> 31,205`)! Mean intermediate-value Jaccard between `133,772` and `132,634` is only **`0.465`** (`0.198` on `scored_0137`, `0.126` on `scored_0253`), and **only `6 / 570` preserved target emission order**.
  * Inspection of big drops (`0791: -10`, `0996: -9`, `0137: -5`, `0253: -4`) proves that the gains did NOT come from deeper local `k=1..4` cuts on fixed helpers — they came from **replacing the entire `sh3+` early backbone / bootstrap + target order** (e.g., `scored_0253` dropped `{1, 2, 4, 8, 32, 1024}` completely in favor of `{15, 297, 729, 744, ...}`!).
* **Frontier techniques extracted from pulled public notebooks (read-only JSON inspection)**:
  * `goodpjw2008` (`history.csv` `133,785 -> 132,513` in dataset; notebook `132,636`): trajectory-harvested **free-order CP-SAT** over plateau-walked equal-length LNS archives + `wangjakie` **`*_diverse`** / **`*_seeded`** crossover (`wj_anneal_diverse` `+95 ops`, `wj_incremental_diverse` `+31`, `wj_anneal_seeded` `+13` since `133,785`) + `atakan_frontier`/`atakan_closure` + iterated local search **kicks** (`0.3` plateau acceptance + random target kick `+2` after 200 stalled steps) + **depth-5 near-miss `T = X*Y +- R` via sorted neighbors of `T/X`** + **helper-rooted subtree cuts**. Negative result documented: full depth-5 MITM (`~10-19M` 2-step values) gave `0` hits on 9 targets of `48-59b`, and fixed-order CP-SAT gave `0` gains (`<=48` register preservation / free-order level vars matter).
  * `wangjakie` (`gpu-recipe-fusion-and-dag-search`): **CUDA NVRTC multi-witness MITM** (`frontier_kernel.cu` — `atomicMin` storing `K` distinct operand witnesses per value in open-addressed hash table so blocking one operand still leaves valid builds), **CUDA parallel patch fusion (`patch_score.cu`)** evaluating combinations of 2–4 near-neutral rewrites sharing newly unlocked/released helpers in parallel, **CUDA greedy closure pruning (`greedy_cuda.cu`)** over random deletion permutations + bitmask DAG closure, and **CUDA portfolio bitmask crossover (`portfolio.cu`)** mixing target-cone selections across parents.
  * `pablodiegoo` (`deterministic-atlas-solver`): C++20 multi-pass pipeline featuring **free-order bitset graph knockout + quartet cluster closure (`dag_knockout_and_quartet_pass`)**, **private-cone balanced/chain rotation (`dag_cone_rotate_pass`)**, **square-peeling & dependency reversal (`dag_deep_chain_and_reversal`)**, cross-trace hypergraph recombination, and fan-out scaffold exchange with subset bitmask DP.
  * `vigneshsankar1962` (`free-order-cp-sat-dag-optimizer`): `bridge_recipes` (2,000 top 1-op candidates bridging `>=2` existing graph nodes via inverse tables) + OR-Tools CP-SAT with integer `lev` variables (`best=132,673`).
  * `lynnsakurai` (`santa-gauss-and-the-r-a-a-p`): canonicalized persistent plateau archives (`Q(G) = |E(G) \ U| / (1 + max(0, C(G)-B))`), up to 12 complementary donors per puzzle within delta `<=3`.
  * `ishankarnick` (`hidden-bases-and-what-didn-t-work`) & `lucifer19` (`x-ray`): confirmed ~15% targets sit on CF-convergent bases `q*B+r` (already captured in public `sh3+` backbones), dead-code count is `0` across all 1,000 frontier programs, and local cone/MITM/one-helper cuts alone are heavily saturated on fixed `sh3+` backbones (`#1 ryan = 127,200` sits `-5,434` below `132,634` via global whole-program construction/backbone synthesis).
  * `discussion/760822` (`#18` Pardheev Krishna): emphasized structural decomposition beyond fixed per-target local recipes.

### E050–E052b Follow-up Observations (vs `132,634`)
* **E050 (`cpp/s2_mitm.cpp`)**: Full-DAG 64-segment L3-friendly `S2` (`10.9M` entries/puzzle) + 6 MITM shapes (`M1..M6` including `S2*S2 ± C` and `T ± C_s1 = A_V * B_S2`) scored `2,287 -> 2,287` (`0` wins in `21.7s`) on `bench12`, confirming local `priv` MITM on fixed `sh3+` backbones is saturated.
* **E051 (`src/export_donors.py` + `cpp/elf3.cpp`)**: Exported `4,759` diverse verified donors across `1,000` puzzles (`avg 4.76/puzzle`), verified `elf3` zero-spill compiler `ok=100/100 exact=100/100` on `every10`, and observed `0` wins from 1-bridge `repair_ban` + basic neutral walk on `every10` (`46.6s`) and `bench12` (`10.7s`).
* **E052a / E052b (`cpp/elf2.cpp -macro`)**: CF/GCD Macro-SA harvested rich near-best topological diversity (`arch=41` on `bench12` in `64.1s`; **`arch=556`** across `75/100` `every10` puzzles in `170.5s`, including `64` novel `+0` equal-length DAGs), while simple `elf3` crossover still scored `0` wins — identifying the next high-leverage gap: free-order CP-SAT recombination over the Macro-SA + multi-donor archives (`E053a`) and adding shared-bottleneck `downstream` `2/3/4`-step rebuild + factoring + `2-bridge` closure + `+1` uphill kicks to `elf3` (`E053b`).
* **E053a / E053b & Frontier Saturation Analysis**:
  * **E053a (`src/recombine.py --donors`)**: Recombining up to `~2,095` donor + `Macro-SA` recipes across `61` targets (`49 every10 arch>=4 + bench12`) yielded `0` DAG wins (`38 opt`, `23 feas`, `chg=0`).
  * **E053b (`cpp/elf3.cpp` `v2`)**: Adding `try_cone_rebuild` (`1..4`-step + 64-bit Pollard-Rho factoring on `dom_cut`), `2-bridge {h1, h2}` closure, `+1` uphill kicks, and `24`-DAG crossover still passed `100/100` zero-spill selftest cleanly, but scored `0` wins on both `every10` (`12,131 -> 12,131`, `65.5s`) and `bench12` (`2,287 -> 2,287`, `12.0s`).
  * **Attribution (`ext/` vs `132,634`)**: `out/public_merge.csv` (`132,634`) equals `ext/alexchilton__santa-2026-rudolph-programs/rudolph_programs.csv` (`1000/1000` ties), while `goodpjw2008` LNS + CP-SAT sits at `132,636` (`998/1000` ties), `lynnsakurai` at `132,712` (`934/1000`), and `wangjakie` at `133,811` (`438/1000`). Thus local 1..3-hole knockout, downstream 2..4-step cone rebuilds, and donor CP-SAT recombination are **saturated around `132,634`**; breaking toward `#1 ryan (127,200)` requires global whole-program construction (joint multi-target backbone/base synthesis, bottom-up GPU/CPU DAG discovery, or structural decomposition across `sh3+` hubs) rather than local repair on `132,634`.
