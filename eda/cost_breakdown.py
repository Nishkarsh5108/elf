"""Where does the public-best score go? Break down by inferred family / size."""
import sys, math, collections
sys.path.insert(0, 'src')
from santa import load_puzzles, load_submission
P = load_puzzles(); S = load_submission(sys.argv[1] if len(sys.argv) > 1 else 'out/public_merge.csv')
rows = []
for p in P:
    nt = len(set(p.targets) - set(p.start))
    L = len(S[p.id])
    grp = 'independent' if max(p.start) >= 100 else ('tiny' if p.limit < 100 else 'other')
    bits = sum(math.log2(t) for t in set(p.targets) - set(p.start))
    rows.append((grp, nt, L, bits, p))
agg = collections.defaultdict(lambda: [0, 0, 0, 0.0])
for g, nt, L, bits, p in rows:
    a = agg[g]; a[0] += 1; a[1] += nt; a[2] += L; a[3] += bits
for g, (n, nt, L, bits) in agg.items():
    print(f'{g:12s} puzzles={n:4d} targets={nt:6d} instr={L:7d} per_target={L/nt:5.2f} avg_bits/target={bits/nt:5.1f} bits/instr={bits/L:5.2f}')
# by log10 of limit
agg2 = collections.defaultdict(lambda: [0, 0, 0])
for g, nt, L, bits, p in rows:
    k = (g, int(math.log10(p.limit)))
    a = agg2[k]; a[0] += 1; a[1] += nt; a[2] += L
for k in sorted(agg2):
    n, nt, L = agg2[k]
    print(k, n, nt, L, f'{L/max(nt,1):.2f}')
