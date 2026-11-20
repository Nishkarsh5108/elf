"""Convert puzzles to the engine's text format and engine output back to a verified submission."""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
from santa import load_puzzles, verify, InvalidProgram, write_submission


def write_puzzles_txt(path, puzzles):
    with open(path, 'w') as f:
        for p in puzzles:
            f.write(f"{p.id} {p.limit} {len(p.start)} {' '.join(map(str, p.start))} {len(p.targets)} {' '.join(map(str, p.targets))}\n")


def read_engine_out(path, puzzles):
    P = {p.id: p for p in puzzles}; progs = {}
    for line in open(path):
        parts = line.split(' ', 2)
        if len(parts) < 3: continue
        pid, n, js = parts
        prog = json.loads(js)
        verify(P[pid], prog)
        progs[pid] = prog
    return progs


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'puzzles':  # io_txt.py puzzles OUT [scored|practice] [filter: ids comma or 'indep']
        which = sys.argv[3] if len(sys.argv) > 3 else 'scored'
        P = load_puzzles(which)
        if len(sys.argv) > 4:
            flt = sys.argv[4]
            if flt == 'indep': P = [p for p in P if max(p.start) >= 100]
            elif flt == 'other': P = [p for p in P if max(p.start) < 100]
            elif flt.startswith('every'): k = int(flt[5:]); P = P[::k]
            else: ids = set(flt.split(',')); P = [p for p in P if p.id in ids]
        write_puzzles_txt(sys.argv[2], P)
        print(len(P), 'puzzles')
