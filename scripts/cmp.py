"""Compare an engine output file against the public best on the puzzles it contains."""
import sys; sys.path.insert(0, 'src')
from santa import load_puzzles, load_submission
from io_txt import read_engine_out
P = load_puzzles(); pub = load_submission(sys.argv[2] if len(sys.argv) > 2 else 'out/public_merge.csv')
mine = read_engine_out(sys.argv[1], P)
Pd = {p.id: p for p in P}
tm = sum(len(v) for v in mine.values()); tp = sum(len(pub[k]) for k in mine)
ti = sum(len(v) for k, v in mine.items() if max(Pd[k].start) >= 100); tpi = sum(len(pub[k]) for k in mine if max(Pd[k].start) >= 100)
wins = sum(1 for k, v in mine.items() if len(v) < len(pub[k]))
print(f'n={len(mine)} mine={tm} public={tp} ratio={tm/tp:.3f} | indep {ti}/{tpi}={ti/max(tpi,1):.3f} other {tm-ti}/{tp-tpi}={(tm-ti)/max(tp-tpi,1):.3f} | wins={wins}')
