"""EDA pass 1: per-family relationships between targets in practice puzzles."""
import sys, math, collections
sys.path.insert(0, 'src')
from santa import load_puzzles

P = load_puzzles('practice')

def rel_stats(p):
    T = sorted(set(p.targets)); S = sorted(set(p.start))
    allv = set(T) | set(S)
    res = collections.Counter()
    for t in T:
        others = [x for x in allv if x != t]
        # 1-op from others
        found = None
        for a in others:
            if t - a in allv and t - a > 0 and t - a != t: found = '+'; break
            if a > t and a - t in allv: found = '-'; break
            if t % a == 0 and t // a in allv and t // a != t: found = '*'; break
        res[found or 'none'] += 1
    return res

for fam in ['independent', 'shared', 'multiplicative', 'mixed', 'tiny']:
    ps = [p for p in P if p.family == fam]
    agg = collections.Counter()
    for p in ps: agg += rel_stats(p)
    print(fam, dict(agg))

# consecutive differences in shared
for fam in ['shared', 'mixed', 'multiplicative', 'independent']:
    p = [p for p in P if p.family == fam][1]
    T = sorted(p.targets)
    print('\n==', fam, p.id, 'start', p.start, 'limit', p.limit)
    for a, b in zip(T, T[1:]):
        g = math.gcd(a, b)
        print(f'{b:>22d} diff={b-a:<20d} ratio={b/a:10.4f} gcd={g}')
