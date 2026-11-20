"""Experiment: at the moment each target is first produced in a program, take the 48-register
snapshot R and compute the min tree cost (<=5) of that target from R alone (meet in the middle
with an S<=2 hash). Compare to the instructions the program spent since the previous target."""
import sys, collections
sys.path.insert(0, 'src')
from santa import load_puzzles, load_submission

def ops(a, b, L):
    out = [a + b, a * b]
    if a > b: out.append(a - b)
    elif b > a: out.append(b - a)
    if a % b == 0: out.append(a // b)
    if b % a == 0: out.append(b // a)
    return [x for x in out if 0 < x <= L]

def build_sets(R, L):
    R = list(set(R)); cost = {v: 0 for v in R}
    S1 = set()
    for i, a in enumerate(R):
        for b in R[i:]:
            for x in ops(a, b, L):
                if x not in cost: cost[x] = 1; S1.add(x)
    S2 = set()
    for x in S1:
        for c in R + [x]:
            for y in ops(x, c, L):
                if y not in cost: cost[y] = 2; S2.add(y)
    return R, S1, S2, cost

def inv(t, b, L):
    """values a such that a op b == t or b op a == t."""
    out = [t - b, t + b, b - t]
    if t % b == 0: out.append(t // b)
    out.append(t * b)
    if b % t == 0: out.append(b // t)
    return [a for a in out if 0 < a <= L]

def dist(t, R, S1, S2, cost, L, maxd=5):
    if t in cost: return cost[t]
    best = None
    # final op combining A (cost ca) and B (cost cb): total ca+cb+1
    for B, cb in [(b, 0) for b in R] + [(b, 1) for b in S1] + ([(b, 2) for b in S2] if maxd >= 5 else []):
        for A in inv(t, B, L):
            ca = cost.get(A)
            if ca is not None:
                tot = ca + cb + 1
                if best is None or tot < best: best = tot
    return best

pid = sys.argv[1]
P = {p.id: p for p in load_puzzles()}; p = P[pid]; prog = load_submission('out/public_merge.csv')[pid]
reg = [None] * 48
for i, v in enumerate(p.start): reg[i] = v
T = set(p.targets) - set(p.start); last = 0; res = []
snap = [v for v in reg if v is not None]
for k, (op, i, j, d) in enumerate(prog):
    a, b = reg[i], reg[j]
    r = {'+': a + b, '-': a - b, '*': a * b, '/': a // b if b else 0}[op]
    if r in T:
        R, S1, S2, cost = build_sets(snap, p.limit)
        dd = dist(r, R, S1, S2, cost, p.limit)
        res.append((k - last + 1, dd, r.bit_length()))
        print(f'step {k:4d} target {r:>20d} bits={r.bit_length():2d} spent_since_prev={k-last+1:3d} mitm_from_snapshot={dd}', flush=True)
        T.discard(r); last = k + 1
    reg[d] = r
    if r in set(p.targets):
        snap = [v for v in reg if v is not None]
