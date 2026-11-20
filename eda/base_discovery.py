"""Prototype: discover hidden bases from pairwise gcd(t_i - d_i, t_j - d_j) with small offsets."""
import sys, math, collections, time
sys.path.insert(0, 'src')
from santa import load_puzzles
pid = sys.argv[1]; D = int(sys.argv[2]) if len(sys.argv) > 2 else 40
P = {p.id: p for p in load_puzzles()} | {p.id: p for p in load_puzzles('practice')}
p = P[pid]; T = sorted(set(p.targets))
t0 = time.time()
cnt = collections.Counter(); support = collections.defaultdict(set)
for a in range(len(T)):
    for b in range(a + 1, len(T)):
        for da in range(-D, D + 1):
            x = T[a] - da
            for db in range(-D, D + 1):
                y = T[b] - db
                g = math.gcd(x, y)
                if g >= 500 and g * g <= min(x, y) * 4:
                    cnt[g] += 1; support[g].add(a); support[g].add(b)
print(pid, p.start, 'n', len(T), f'{time.time()-t0:.1f}s')
# rank by number of targets supported
npairs = len(T) * (len(T) - 1) / 2
exp = lambda g: npairs * ((2 * D + 1) / g) ** 2
best = sorted([g for g in cnt if len(support[g]) >= 3], key=lambda g: -cnt[g] / exp(g))[:25]
for g in best:
    f = []; x = g
    for q in range(2, 2000):
        while x % q == 0: f.append(q); x //= q
    if x > 1: f.append(x)
    print(g, 'targets', len(support[g]), 'pairs', cnt[g], 'surprise %.1f' % (cnt[g] / exp(g)), 'factors', f)
