"""Write a puzzle txt file where discovered bases are appended as extra (helper) targets.
usage: with_bases.py BASES_FILE OUT_TXT [only_with_bases=1] [maxk]"""
import sys, copy
sys.path.insert(0, 'src')
from santa import load_puzzles
from io_txt import write_puzzles_txt
bf, out = sys.argv[1], sys.argv[2]
only = int(sys.argv[3]) if len(sys.argv) > 3 else 1
maxk = int(sys.argv[4]) if len(sys.argv) > 4 else 99
P = {p.id: p for p in load_puzzles()} | {p.id: p for p in load_puzzles('practice')}
res = []
for line in open(bf):
    f = line.split(); pid, k = f[0], int(f[1])
    if only and k == 0: continue
    q = copy.deepcopy(P[pid])
    bases = [int(x.split(':')[0]) for x in f[2:2 + min(k, maxk)]]
    q.targets = sorted(set(q.targets) | set(b for b in bases if b <= q.limit))
    res.append(q)
write_puzzles_txt(out, res)
print(len(res), 'puzzles')
