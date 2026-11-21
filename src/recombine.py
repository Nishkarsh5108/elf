"""Recombination: pool the value recipes of several programs for the same puzzle (plus, optionally, every
one-operation recipe between pooled values) and pick, with CP-SAT, the smallest acyclic set of values that
covers all targets. The chosen DAG is written for the C++ engine to compile (`elf -dag`).

usage: python src/recombine.py OUT_DAG.txt IDS(comma|every10|all) --sources f1 f2 ... [--pool 1] [--time 20]
Sources: submission CSVs or engine outputs (id len json).
"""
import argparse, concurrent.futures, csv, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from santa import load_puzzles
csv.field_size_limit(1 << 30)


def canon(r, op, a, b):
    if op in ('+', '*') and a > b:
        a, b = b, a
    return (r, op, a, b)


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
        rec.add(canon(r, op, a, b)); reg[d] = r
    return rec


def recipe_seq(p, prog):
    reg = [None] * 48
    for i, v in enumerate(p.start): reg[i] = v
    seq = []
    for op, i, j, d in prog:
        a, b = reg[i], reg[j]
        r = {'+': a + b, '-': a - b, '*': a * b, '/': a // b}[op]
        seq.append(canon(r, op, a, b)); reg[d] = r
    return seq


def pool_recipes(vals, L):
    """all v = a op b with a, b, v in vals (numpy, int64; values <= 1e18)."""
    V = np.array(sorted(vals), dtype=np.int64)
    out = set()
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
                if rv != av and rv != b: out.add(canon(rv, op, av, b))
    return out


def solve(p, recs, tlimit, hint=None, num_workers=8):
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
    hset = set()
    if hint:  # hint: list of (v, op, a, b) of the best known program, in creation order
        hv = {}
        for k, r in enumerate(hint):
            if r[0] not in starts and r[0] not in hv:
                hv[r[0]] = (k + 1, r)
        hset = {r for (_, r) in hv.values()}
        if all(r in ch for r in hset) and all(t in starts or t in hv for t in p.targets):
            pub_n = len(hv)
            hint_seq = [r for _, r in sorted(hv.values())]
            lu = {}
            for i, (v, op, a, b) in enumerate(hint_seq):
                lu[a] = i
                lu[b] = i
            live = {v for v in starts if v in lu}
            step_peak = []
            for i, (v, op, a, b) in enumerate(hint_seq):
                freed = sum(1 for x in {a, b} if lu.get(x, -1) == i)
                pk = len(live) + (1 if freed == 0 else 0)
                step_peak.append(pk)
                if lu.get(v, -1) > i:
                    live.add(v)
                for x in {a, b}:
                    if lu.get(x, -1) == i and x in live:
                        live.remove(x)
            max_peak = max(step_peak) if step_peak else 0
            if max_peak >= 47:
                thresh = 48 if max_peak >= 48 else 47
                last_sat = max(i for i, pk in enumerate(step_peak) if pk >= thresh)
                early_dead = {v for v in hv if lu.get(v, -1) >= 0 and lu[v] < last_sat}
            else:
                early_dead = set()

            # Stage 1: pure minimization with PUB hint and <= pub_n cutoff
            for v, x in use.items():
                m.AddHint(x, 1 if v in hv else 0)
            for r, x in ch.items():
                m.AddHint(x, 1 if r in hset else 0)
            for v in lvl:
                if v in hv:
                    m.AddHint(lvl[v], hv[v][0])
                elif v in starts:
                    m.AddHint(lvl[v], 0)
            m.Add(sum(use.values()) <= pub_n)
            m.Minimize(sum(use.values()))
            s = cp_model.CpSolver()
            s.parameters.max_time_in_seconds = tlimit
            s.parameters.num_workers = num_workers
            st = s.Solve(m)
            if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                return hint_seq, (st == cp_model.OPTIMAL), 0
            best_n = int(round(s.ObjectiveValue()))
            if best_n >= pub_n:
                return hint_seq, (st == cp_model.OPTIMAL), 0

            # Stage 2: minimum-perturbation + register-safe tie-break at fixed best_n
            chosen1 = [r for r, x in ch.items() if s.Value(x)]
            chosen1.sort(key=lambda r: s.Value(lvl[r[0]]))
            stage1_opt = (st == cp_model.OPTIMAL)

            m.ClearHints()
            m.Add(sum(use.values()) == best_n)
            for v, x in use.items():
                m.AddHint(x, int(s.Value(x)))
            for r, x in ch.items():
                m.AddHint(x, int(s.Value(x)))
            for v in lvl:
                m.AddHint(lvl[v], int(s.Value(lvl[v])))

            pen = []
            for r, x in ch.items():
                if r not in hset:
                    w = 1 + 10 * ((1 if r[2] in early_dead else 0) + (1 if r[3] in early_dead else 0))
                    pen.append(w * x)
            m.Minimize(sum(pen))
            s2 = cp_model.CpSolver()
            s2.parameters.max_time_in_seconds = min(5.0, max(2.0, tlimit * 0.25))
            s2.parameters.num_workers = num_workers
            st2 = s2.Solve(m)
            if st2 in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                chosen = [r for r, x in ch.items() if s2.Value(x)]
                chosen.sort(key=lambda r: s2.Value(lvl[r[0]]))
            else:
                chosen = chosen1
            return chosen, stage1_opt, len(set(chosen) - hset)
        else:
            for v, x in use.items(): m.AddHint(x, 1 if v in hv else 0)
            for r, x in ch.items(): m.AddHint(x, 1 if r in hset else 0)
            for v in lvl:
                if v in hv: m.AddHint(lvl[v], hv[v][0])
                elif v in starts: m.AddHint(lvl[v], 0)
            m.Minimize(sum(use.values()))
    else:
        m.Minimize(sum(use.values()))
    s = cp_model.CpSolver(); s.parameters.max_time_in_seconds = tlimit; s.parameters.num_workers = num_workers
    st = s.Solve(m)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE): return None
    chosen = [r for r, x in ch.items() if s.Value(x)]
    chosen.sort(key=lambda r: s.Value(lvl[r[0]]))
    changed = len(set(chosen) - hset) if hint else len(chosen)
    return chosen, st == cp_model.OPTIMAL, changed


def load_extra_recipes(path):
    """Read per-puzzle recipe pools: 'pid N' followed by N lines of 'r op a b'."""
    out = {}
    with open(path, encoding='utf-8') as f:
        while True:
            line = f.readline()
            if not line:
                break
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            pid, m = parts[0], int(parts[1])
            s = set()
            for _ in range(m):
                rp = f.readline().strip().split()
                if len(rp) == 4:
                    s.add(canon(int(rp[0]), rp[1], int(rp[2]), int(rp[3])))
            out[pid] = s
    return out


def load_donor_progs(paths):
    import json
    by_pid = {}
    for path in paths:
        if not path:
            continue
        with open(path) as f:
            lines = iter(f)
            for line in lines:
                s = line.strip()
                if not s or not s.startswith('PUZ '):
                    continue
                _, pid, _, nd = s.split()
                lst = by_pid.setdefault(pid, [])
                for _ in range(int(nd)):
                    dl = next(lines).strip()
                    _, prog_s = dl.split(None, 1)
                    lst.append([tuple(op) for op in json.loads(prog_s)])
    return by_pid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('ids')
    ap.add_argument('--sources', nargs='*', default=[])
    ap.add_argument('--donors', nargs='*', default=[])
    ap.add_argument('--extra-recipes', default='')
    ap.add_argument('--pool', type=int, default=0); ap.add_argument('--time', type=float, default=20)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--parallel', type=int, default=1)
    a = ap.parse_args()
    P = load_puzzles()
    if a.ids == 'all': ids = [p.id for p in P]
    elif a.ids.startswith('every'): ids = [p.id for p in P][::int(a.ids[5:])]
    elif a.ids.startswith('@'):
        want = [x.strip() for x in open(a.ids[1:]).read().replace('\n', ',').split(',') if x.strip()]
        ids = want
    else: ids = [x for x in a.ids.split(',') if x]
    Pd = {p.id: p for p in P}
    srcs = [load_progs(f) for f in a.sources]
    donors = load_donor_progs(a.donors)
    extra = load_extra_recipes(a.extra_recipes) if a.extra_recipes else {}

    def run_one(pid):
        p = Pd[pid]; t0 = time.time()
        recs = set(); best = None; hint = None
        for S in srcs:
            if pid in S:
                recs |= recipes_of(p, S[pid])
                if best is None or len(S[pid]) < best:
                    best = len(S[pid]); hint = recipe_seq(p, S[pid])
        for prog in donors.get(pid, ()):
            recs |= recipes_of(p, prog)
            if best is None or len(prog) < best:
                best = len(prog); hint = recipe_seq(p, prog)
        if pid in extra:
            recs |= extra[pid]
        if a.pool:
            vals = set(p.start) | {r[0] for r in recs}
            recs |= pool_recipes(vals, p.limit)
        res = solve(p, recs, a.time, hint, num_workers=a.threads)
        dt = time.time() - t0
        if res is None:
            return pid, best, len(recs), dt, None
        chosen, opt, changed = res
        return pid, best, len(recs), dt, (chosen, opt, changed)

    results = {}
    if a.parallel > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.parallel) as ex:
            futs = {ex.submit(run_one, pid): pid for pid in ids}
            for fut in concurrent.futures.as_completed(futs):
                pid, best, nrecs, dt, payload = fut.result()
                results[pid] = payload
                if payload is None:
                    print(f'{pid}: no solution {dt:.1f}s', flush=True)
                else:
                    chosen, opt, changed = payload
                    print(f'{pid}: best source {best} -> dag {len(chosen)} ({"opt" if opt else "feas"}, chg={changed}) recipes {nrecs} {dt:.1f}s', flush=True)
    else:
        for pid in ids:
            pid, best, nrecs, dt, payload = run_one(pid)
            results[pid] = payload
            if payload is None:
                print(f'{pid}: no solution {dt:.1f}s', flush=True)
            else:
                chosen, opt, changed = payload
                print(f'{pid}: best source {best} -> dag {len(chosen)} ({"opt" if opt else "feas"}, chg={changed}) recipes {nrecs} {dt:.1f}s', flush=True)

    with open(a.out, 'w') as fo:
        for pid in ids:
            payload = results.get(pid)
            if payload is None:
                continue
            chosen, _, _ = payload
            fo.write(f'{pid} {len(chosen)}\n')
            for r in chosen:
                fo.write(f'{r[0]} {r[1]} {r[2]} {r[3]}\n')


if __name__ == '__main__':
    main()
