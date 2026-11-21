"""For each target, list structural decompositions t = m * X + d with X a base or base product."""
import sys, itertools
sys.path.insert(0, 'src')
from santa import load_puzzles
pid = sys.argv[1]; bases = [int(x) for x in sys.argv[2].split(',')]
P = {p.id: p for p in load_puzzles()}; p = P[pid]
X = {}
for b in bases: X[f'b{b}'] = b
for b, c in itertools.combinations_with_replacement(bases, 2): X[f'{b}*{c}'] = b * c
for t in sorted(p.targets):
    best = []
    for name, x in X.items():
        if x > t: continue
        m = round(t / x); d = t - m * x
        if m >= 1 and abs(d) <= max(2000, m):
            best.append((len(str(m)) + len(str(abs(d))), name, m, d))
    best.sort()
    print(t, best[:3])
