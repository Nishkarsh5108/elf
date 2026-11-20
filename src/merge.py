"""Per-puzzle merge: scan CSV files with (id, program) columns, verify every program exactly,
keep the shortest valid program per puzzle. Usage: python src/merge.py OUT.csv [csv or dir ...]"""
import csv, json, os, sys, glob
sys.path.insert(0, os.path.dirname(__file__))
from santa import load_puzzles, verify, InvalidProgram, write_submission
csv.field_size_limit(1 << 30)


def iter_csvs(paths):
    for p in paths:
        if os.path.isdir(p):
            yield from sorted(glob.glob(os.path.join(p, '**', '*.csv'), recursive=True))
        else:
            yield p


def merge(out, paths, puzzles=None, verbose=True):
    P = {p.id: p for p in (puzzles or load_puzzles())}
    best, src = {}, {}
    for f in iter_csvs(paths):
        try:
            with open(f, newline='', encoding='utf-8') as fh:
                r = csv.DictReader(fh)
                if not r.fieldnames or 'id' not in r.fieldnames or 'program' not in r.fieldnames:
                    continue
                n_ok = n_bad = n_imp = 0
                for row in r:
                    pid = row['id']
                    if pid not in P:
                        continue
                    try:
                        prog = json.loads(row['program'])
                        L = verify(P[pid], prog)
                    except (InvalidProgram, ValueError, TypeError) as e:
                        n_bad += 1
                        continue
                    n_ok += 1
                    if pid not in best or L < len(best[pid]):
                        best[pid] = prog; src[pid] = f; n_imp += 1
        except Exception as e:
            if verbose: print('skip', f, e)
            continue
        if verbose and (n_ok or n_bad):
            print(f'{f}: ok={n_ok} bad={n_bad} improved={n_imp}')
    missing = [k for k in P if k not in best]
    total = sum(len(v) for v in best.values())
    print('puzzles covered', len(best), 'missing', len(missing), 'total', total)
    if out and not missing:
        write_submission(out, best)
    return best, src


if __name__ == '__main__':
    merge(sys.argv[1], sys.argv[2:])
