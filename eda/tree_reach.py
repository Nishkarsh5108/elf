"""Hypothesis test: are targets cheap *trees* over the starting values alone?
Build min-cost dict for tree expressions up to cost K over start values (+ derived 1),
then for each target find min cost x op y with y cost<=a, x cost<=K (meet in the middle)."""
import sys, time, collections
sys.path.insert(0, 'src')
from santa import load_puzzles

def build(start, limit, K):
    cost = {}
    levels = [set()]
    for v in start:
        cost[v] = 0
        levels[0].add(v)
    for k in range(1, K + 1):
        new = set()
        for i in range(0, k):
            j = k - 1 - i
            if j < i: break
            A, B = levels[i], levels[j]
            for a in A:
                for b in B:
                    for r in (a + b, a * b, abs(a - b), a // b if b and a % b == 0 else 0, b // a if a and b % a == 0 else 0):
                        if 0 < r <= limit and r not in cost:
                            new.add(r)
        for r in new: cost[r] = k
        levels.append(new)
        print(f'  level {k}: {len(new)} new values', flush=True)
    return cost, levels

def target_cost(t, cost, levels, maxy):
    if t in cost: return cost[t]
    best = None
    for cy in range(0, maxy + 1):
        for y in levels[cy]:
            for x in (t - y, t + y, t // y if t % y == 0 else 0, t * y):
                c = cost.get(x)
                if c is not None and x > 0:
                    tot = c + cy + 1
                    if best is None or tot < best: best = tot
    return best

if __name__ == '__main__':
    pid = sys.argv[1]; K = int(sys.argv[2]); maxy = int(sys.argv[3])
    P = {p.id: p for p in load_puzzles('practice')} | {p.id: p for p in load_puzzles('scored')}
    p = P[pid]
    print(pid, p.family, p.start, len(p.targets), p.limit)
    t0 = time.time()
    cost, levels = build(p.start, p.limit, K)
    res = collections.Counter()
    for t in p.targets:
        c = target_cost(t, cost, levels, maxy)
        res[c] += 1
    print('cost histogram (None = > %d):' % (K + maxy + 1), sorted(res.items(), key=lambda x: (x[0] is None, x[0])), f'{time.time()-t0:.1f}s')
