"""Experiment: CP-SAT over a structure-derived candidate pool for one puzzle.
Pool = union of given programs' values + small helpers (<=2 ops from starts) + bases + base products +
every target's decompositions t = m*X + d (X a base/base product), closed under 1-op recipes.
usage: python eda/struct_pool.py PID BASES(comma) TIME SOURCES..."""
import sys, os, itertools, time
sys.path.insert(0, 'src')
from santa import load_puzzles
from recombine import load_progs, recipes_of, recipe_seq, pool_recipes, solve

pid, bases, tl = sys.argv[1], [int(x) for x in sys.argv[2].split(',') if x], float(sys.argv[3])
srcs = sys.argv[4:]
P = {p.id: p for p in load_puzzles()}; p = P[pid]; L = p.limit
recs = set(); hint = None; best = None
for f in srcs:
    S = load_progs(f)
    if pid in S:
        recs |= recipes_of(p, S[pid])
        if best is None or len(S[pid]) < best: best = len(S[pid]); hint = recipe_seq(p, S[pid])
vals = set(p.start) | {r[0] for r in recs}
# small helpers: <= 2 ops from the starts (bounded size)
s0 = list(p.start); s1 = set()
for a in s0:
    for b in s0:
        for r in (a + b, a * b, a - b, a // b if a % b == 0 else 0):
            if 0 < r <= min(L, 10**7): s1.add(r)
small = set(s0) | s1
for a in list(s1):
    for b in s0 + [a]:
        for r in (a + b, a * b, abs(a - b), a // b if a % b == 0 else 0, b // a if b % a == 0 else 0):
            if 0 < r <= min(L, 10**6): small.add(r)
vals |= small
X = set(bases)
for b, c in itertools.combinations_with_replacement(bases, 2):
    if b * c <= L: X.add(b * c)
vals |= X
for t in p.targets:
    for x in X:
        if x > t: continue
        for m in {t // x, t // x + 1}:
            if m < 1 or m * x > L: continue
            d = t - m * x
            if abs(d) <= 10**6:
                vals.add(m * x)
                if m <= 10**7: vals.add(m)
                if d: vals.add(abs(d))
print('pool values', len(vals), flush=True)
t0 = time.time()
recs |= pool_recipes(vals, L)
print('recipes', len(recs), f'{time.time()-t0:.1f}s', flush=True)
res = solve(p, recs, tl, hint)
if res:
    chosen, opt = res
    print(f'{pid}: best source {best} -> dag {len(chosen)} ({"opt" if opt else "feas"})')
    with open(f'scratch/dag_{pid}.txt', 'w') as fo:
        fo.write(f'{pid} {len(chosen)}\n')
        for r in chosen: fo.write(f'{r[0]} {r[1]} {r[2]} {r[3]}\n')
