"""Export diverse verified donor programs per puzzle from best.csv + deduplicated ext/ CSVs."""
import csv, glob, hashlib, json, os, sys
from collections import defaultdict
from pathlib import Path

csv.field_size_limit(1 << 30)
sys.path.insert(0, os.path.dirname(__file__))
from santa import load_puzzles, load_submission, verify, InvalidProgram


def canonical_recipes(p, prog):
    regs = list(p.start) + [None] * (p.registers - len(p.start))
    edges = set()
    for op, i, j, d in prog:
        a, b = regs[i], regs[j]
        if op == "+":
            v = a + b
            edges.add(("+", min(a, b), max(a, b), v))
        elif op == "*":
            v = a * b
            edges.add(("*", min(a, b), max(a, b), v))
        elif op == "-":
            v = a - b
            edges.add(("-", a, b, v))
        else:
            v = a // b
            edges.add(("/", a, b, v))
        regs[d] = v
    return frozenset(edges)


def export_donors(out_path: str, best_csv: str, ext_dir: str, max_donors: int = 12, max_slack: int = 6):
    puzzles = {p.id: p for p in load_puzzles()}
    best = load_submission(best_csv)
    for pid, p in puzzles.items():
        verify(p, best[pid])

    # Gather unique CSV paths by file SHA-256
    seen_sha = set()
    csv_files = [best_csv]
    for fp in sorted(glob.glob(os.path.join(ext_dir, "**", "*.csv"), recursive=True)):
        raw = Path(fp).read_bytes()
        h = hashlib.sha256(raw).hexdigest()
        if h not in seen_sha:
            seen_sha.add(h)
            csv_files.append(fp)

    raw_seen = defaultdict(set)      # pid -> set of compact json strings
    cands = defaultdict(list)        # pid -> list of (len, compact_js, frozenset_recipes)

    for fp in csv_files:
        try:
            with open(fp, newline="", encoding="utf-8") as fh:
                r = csv.DictReader(fh)
                if not r.fieldnames or "id" not in r.fieldnames or "program" not in r.fieldnames:
                    continue
                for row in r:
                    pid = row["id"]
                    if pid not in puzzles:
                        continue
                    best_len = len(best[pid])
                    s_raw = row["program"].strip()
                    if len(s_raw) > 131072 or s_raw in raw_seen[pid]:
                        continue
                    try:
                        prog = json.loads(s_raw)
                    except Exception:
                        continue
                    if not isinstance(prog, list) or len(prog) > best_len + max_slack:
                        continue
                    raw_seen[pid].add(s_raw)
                    try:
                        Lv = verify(puzzles[pid], prog)
                    except InvalidProgram:
                        continue
                    js = json.dumps(prog, separators=(",", ":"))
                    edges = canonical_recipes(puzzles[pid], prog)
                    cands[pid].append((Lv, js, edges))
        except Exception:
            continue

    tot_donors = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for pid in sorted(puzzles):
            bprog = best[pid]
            blen = len(bprog)
            bjs = json.dumps(bprog, separators=(",", ":"))
            bedges = canonical_recipes(puzzles[pid], bprog)

            # Deduplicate by canonical recipe set, keeping shortest per distinct DAG
            by_dag = {}
            by_dag[bedges] = (blen, bjs, bedges)
            for Lv, js, edges in cands[pid]:
                if edges not in by_dag or Lv < by_dag[edges][0]:
                    by_dag[edges] = (Lv, js, edges)

            pool = [v for k, v in by_dag.items() if k != bedges]
            selected = [(blen, bjs, bedges)]
            covered = set(bedges)
            for _ in range(max_donors - 1):
                if not pool:
                    break
                best_idx = -1
                best_q = -1.0
                for idx, (Lv, js, edges) in enumerate(pool):
                    novel = len(edges - covered)
                    if novel == 0:
                        continue
                    q = novel / (1.0 + max(0, Lv - blen))
                    if q > best_q:
                        best_q = q
                        best_idx = idx
                if best_idx < 0:
                    break
                chosen = pool.pop(best_idx)
                selected.append(chosen)
                covered |= chosen[2]

            tot_donors += len(selected)
            out.write(f"PUZ {pid} {blen} {len(selected)}\n")
            for Lv, js, _ in selected:
                out.write(f"{Lv} {js}\n")

    print(f"Exported {tot_donors} diverse verified donors across {len(puzzles)} puzzles to {out_path} "
          f"(avg {tot_donors / len(puzzles):.2f}/puzzle)")


if __name__ == "__main__":
    out_p = sys.argv[1] if len(sys.argv) > 1 else "scratch/donors_132634.txt"
    best_p = sys.argv[2] if len(sys.argv) > 2 else "out/public_merge.csv"
    ext_p = sys.argv[3] if len(sys.argv) > 3 else "ext"
    export_donors(out_p, best_p, ext_p)
