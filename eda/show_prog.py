"""Pretty-print a program with values, marking targets, first-use and reuse counts."""
import sys, collections
sys.path.insert(0, 'src')
from santa import load_puzzles, load_submission
pid = sys.argv[1]; sub = sys.argv[2] if len(sys.argv) > 2 else 'out/public_merge.csv'
P = {p.id: p for p in load_puzzles()}; p = P[pid]; prog = load_submission(sub)[pid]
reg = [None] * 48
for i, v in enumerate(p.start): reg[i] = v
T = set(p.targets); uses = collections.Counter()
vals = []
for op, i, j, d in prog:
    a, b = reg[i], reg[j]
    r = {'+': a + b, '-': a - b, '*': a * b, '/': a // b if b else 0}[op]
    uses[a] += 1; uses[b] += 1
    vals.append((op, a, b, r)); reg[d] = r
print(pid, 'start', p.start, 'limit %.3g' % p.limit, 'ntargets', len(T), 'len', len(prog))
since = 0
for k, (op, a, b, r) in enumerate(vals):
    mark = 'T' if r in T else ' '
    print(f'{k:4d} {mark} {a:>20d} {op} {b:<20d} = {r:<20d} uses={uses[r]}')
