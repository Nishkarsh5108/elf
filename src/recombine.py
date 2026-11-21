"""Recombination: pool the value recipes of several programs for the same puzzle (plus, optionally, every
one-operation recipe between pooled values) and pick, with CP-SAT, the smallest acyclic set of values that
covers all targets. The chosen DAG is written for the C++ engine to compile (`elf -dag`).

usage: python src/recombine.py OUT_DAG.txt IDS(comma|every10|all) --sources f1 f2 ... [--pool 1] [--time 20]
Sources: submission CSVs or engine outputs (id len json).
"""
import argparse, csv, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from santa import load_puzzles
csv.field_size_limit(1 << 30)


def load_progs(path):
    out = {}
    if path.endswith('.csv'):
        with open(path, newline='', encoding='utf-8') as f:
            for r in csv.DictReader(f):
                out[r['id']] = json.loads(r['program'])
    else:
        for line in open(path):
            p = line.split(' ', 2)
            if len(p) == 3: out[p[0]] = json.loads(p[2])
    return out


def recipes_of(p, prog):
    reg = [None] * 48
    for i, v in enumerate(p.start): reg[i] = v
    rec = set()
    for op, i, j, d in prog:
        a, b = reg[i], reg[j]
        r = {'+': a + b, '-': a - b, '*': a * b, '/': a // b}[op]
        rec.add((r, op, a, b)); reg[d] = r
    return rec


def recipe_seq(p, prog):
    reg = [None] * 48
    for i, v in enumerate(p.start): reg[i] = v
    seq = []
    for op, i, j, d in prog:
        a, b = reg[i], reg[j]
        r = {'+': a + b, '-': a - b, '*': a * b, '/': a // b}[op]
        seq.append((r, op, a, b)); reg[d] = r
    return seq


def pool_recipes(vals, L):
    """all v = a op b with a, b, v in vals (numpy, int64; values <= 1e18)."""
    V = np.array(sorted(vals), dtype=np.int64)
    S = set(vals); out = set()
    for b in V:
        b = int(b)
        a = V
        for op in ('+', '-', '*', '/'):
            if op == '+': r = a + b; ok = (r <= L) & (r > 0)
            elif op == '-': r = a - b; ok = r > 0
            elif op == '*':
                ok = a <= L // b; r = np.where(ok, a, 0) * b
            else:
                ok = (a % b) == 0; r = a // b
            idx = np.nonzero(ok & np.isin(r, V))[0]
            for k in idx:
                rv = int(r[k]); av = int(a[k])
                if rv != av and rv != b: out.add((rv, op, av, b))
    return out


def solve(p, recs, tlimit, hint=None):
    from ortools.sat.python import cp_model
    starts = set(p.start)
    vals = set(starts) | {r[0] for r in recs}
    byv = {}
    for r in recs:
        if r[0] in starts: continue
        byv.setdefault(r[0], []).append(r)
    m = cp_model.CpModel()
    use = {v: m.NewBoolVar('') for v in vals if v not in starts}
    lvl = {v: m.NewIntVar(0, len(vals), '') for v in vals}
    for v in starts: m.Add(lvl[v] == 0)
    ch = {}
    for v, rs in byv.items():
        xs = []
        for r in rs:
            x = m.NewBoolVar(''); xs.append(x); ch[r] = x
            for a in (r[2], r[3]):
                if a not in starts:
                    if a not in use: m.Add(x == 0); continue
                    m.AddImplication(x, use[a])
                m.Add(lvl[v] >= lvl[a] + 1).OnlyEnforceIf(x)
        m.Add(sum(xs) == use[v])
    for v in use:
        if v not in byv: m.Add(use[v] == 0)
    for t in p.targets:
        if t not in starts:
            if t not in use: return None
            m.Add(use[t] == 1)
    if hint:  # hint: list of (v, op, a, b) of the best known program, in creation order
        hv = {}; 
        for k, r in enumerate(hint):
            if r[0] not in starts and r[0] not in hv: hv[r[0]] = (k + 1, r)
        hset = {r for (_, r) in hv.values()}
        for v, x in use.items(): m.AddHint(x, 1 if v in hv else 0)
        for r, x in ch.items(): m.AddHint(x, 1 if r in hset else 0)
        for v in lvl:
            if v in hv: m.AddHint(lvl[v], hv[v][0])
            elif v in starts: m.AddHint(lvl[v], 0)
    m.Minimize(sum(use.values()))
    s = cp_model.CpSolver(); s.parameters.max_time_in_seconds = tlimit; s.parameters.num_workers = 8
    st = s.Solve(m)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE): return None
    chosen = [r for r, x in ch.items() if s.Value(x)]
    chosen.sort(key=lambda r: s.Value(lvl[r[0]]))
    return chosen, st == cp_model.OPTIMAL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('ids')
    ap.add_argument('--sources', nargs='+', required=True)
    ap.add_argument('--pool', type=int, default=0); ap.add_argument('--time', type=float, default=20)
    a = ap.parse_args()
    P = load_puzzles()
    if a.ids == 'all': ids = [p.id for p in P]
    elif a.ids.startswith('every'): ids = [p.id for p in P][::int(a.ids[5:])]
    else: ids = a.ids.split(',')
    Pd = {p.id: p for p in P}
    srcs = [load_progs(f) for f in a.sources]
    with open(a.out, 'w') as fo:
        for pid in ids:
            p = Pd[pid]; t0 = time.time()
            recs = set(); best = None; hint = None
            for S in srcs:
                if pid in S:
                    recs |= recipes_of(p, S[pid])
                    if best is None or len(S[pid]) < best:
                        best = len(S[pid]); hint = recipe_seq(p, S[pid])
            if a.pool:
                vals = set(p.start) | {r[0] for r in recs}
                recs |= pool_recipes(vals, p.limit)
            res = solve(p, recs, a.time, hint)
            if res is None: print(pid, 'no solution'); continue
            chosen, opt = res
            print(f'{pid}: best source {best} -> dag {len(chosen)} ({"opt" if opt else "feas"}) recipes {len(recs)} {time.time()-t0:.1f}s', flush=True)
            fo.write(f'{pid} {len(chosen)}\n')
            for r in chosen: fo.write(f'{r[0]} {r[1]} {r[2]} {r[3]}\n')


if __name__ == '__main__':
    main()
