"""Attribute instructions of a program to targets: a value belongs to the first target whose cone
(restricted to values not yet attributed) contains it. Prints per-target costs side by side for two programs."""
import sys, json
sys.path.insert(0, 'src')
from santa import load_puzzles, load_submission


def dag(p, prog):
    reg = [None] * 48
    for i, v in enumerate(p.start): reg[i] = v
    defs = {}; order = []
    for op, i, j, d in prog:
        a, b = reg[i], reg[j]
        r = {'+': a + b, '-': a - b, '*': a * b, '/': a // b}[op]
        if r not in defs and r not in p.start: defs[r] = (op, a, b); order.append(r)
        reg[d] = r
    return defs, order


def attribute(p, prog):
    defs, order = dag(p, prog)
    T = set(p.targets) - set(p.start)
    seen = set(p.start); res = []
    for v in order:
        if v in T and v not in [x[0] for x in res]:
            stack = [v]; cone = []
            while stack:
                x = stack.pop()
                if x in seen or x not in defs: continue
                seen.add(x); cone.append(x)
                stack += [defs[x][1], defs[x][2]]
            res.append((v, len(cone), sorted(c for c in cone if c != v)))
    return res, len(order)


if __name__ == '__main__':
    pid = sys.argv[1]
    P = {p.id: p for p in load_puzzles()}; p = P[pid]
    A = load_submission('out/public_merge.csv')[pid]
    m = {}
    for line in open(sys.argv[2]):
        k, n, js = line.split(' ', 2); m[k] = json.loads(js)
    B = m[pid]
    for name, prog in [('public', A), ('mine', B)]:
        res, n = attribute(p, prog)
        print(f'== {name}: {len(prog)} instr, dag {n}')
        for v, c, cone in res: print(f'  {v:>20d} cost {c:2d}  helpers {cone[:8]}')
