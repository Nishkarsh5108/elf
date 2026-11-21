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
