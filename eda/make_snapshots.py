"""Emit probe queries: for each target in a program, the register snapshot right after the
previous target was produced. Line: limit target n regs... ; also prints spent-since-prev to stderr file."""
import sys
sys.path.insert(0, 'src')
from santa import load_puzzles, load_submission
pids = sys.argv[1].split(','); sub = sys.argv[2]; meta = open(sys.argv[3], 'w')
P = {p.id: p for p in load_puzzles()}; S = load_submission(sub)
for pid in pids:
    p = P[pid]; prog = S[pid]
    reg = [None] * 48
    for i, v in enumerate(p.start): reg[i] = v
    T = set(p.targets) - set(p.start); snap = [v for v in reg if v is not None]; last = 0
    for k, (op, i, j, d) in enumerate(prog):
        a, b = reg[i], reg[j]
        r = {'+': a + b, '-': a - b, '*': a * b, '/': a // b if b else 0}[op]
        reg[d] = r
        if r in T:
            print(p.limit, r, len(snap), *snap)
            meta.write(f'{pid} {k} {k - last + 1}\n')
            T.discard(r); last = k + 1
            snap = [v for v in reg if v is not None]
