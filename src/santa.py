"""Core utilities for Santa 2026: loading puzzles, exact program verification, scoring, IO.

All arithmetic uses Python ints (exact). A program is a list of [op, i, j, d].
"""
from __future__ import annotations
import csv, json, os
from dataclasses import dataclass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
NREG = 48
MAX_INSTR = 4096
MAX_BYTES = 131072


@dataclass
class Puzzle:
    id: str
    start: list
    targets: list
    limit: int
    registers: int = 48
    family: str | None = None
    tier: str | None = None
    lower_bound: int | None = None


def load_puzzles(which: str = "scored") -> list[Puzzle]:
    fn = {"scored": "puzzles.csv", "practice": "practice_puzzles.csv"}[which]
    out = []
    with open(os.path.join(DATA, fn), newline="") as f:
        for r in csv.DictReader(f):
            out.append(Puzzle(
                id=r["id"], start=json.loads(r["starting_values"]), targets=json.loads(r["targets"]),
                limit=int(r["value_limit"]), registers=int(r["registers"]),
                family=r.get("family"), tier=r.get("design_tier"),
                lower_bound=int(r["lower_bound"]) if r.get("lower_bound") else None))
    return out


class InvalidProgram(Exception):
    pass


def run_program(p: Puzzle, prog) -> set:
    """Execute exactly per the rules; return the set of all values ever present. Raises InvalidProgram."""
    if len(prog) > MAX_INSTR:
        raise InvalidProgram(f"{p.id}: too many instructions {len(prog)}")
    reg = [None] * NREG
    for k, v in enumerate(p.start):
        reg[k] = v
    seen = set(p.start)
    L = p.limit
    for t, ins in enumerate(prog):
        if len(ins) != 4:
            raise InvalidProgram(f"{p.id}#{t}: bad instr {ins}")
        op, i, j, d = ins
        for x in (i, j, d):
            if not (isinstance(x, int) and 0 <= x < NREG):
                raise InvalidProgram(f"{p.id}#{t}: bad register {ins}")
        a, b = reg[i], reg[j]
        if a is None or b is None:
            raise InvalidProgram(f"{p.id}#{t}: undefined source {ins}")
        if op == "+":
            r = a + b
        elif op == "-":
            if a <= b:
                raise InvalidProgram(f"{p.id}#{t}: non-positive subtraction {ins}")
            r = a - b
        elif op == "*":
            r = a * b
        elif op == "/":
            if a % b:
                raise InvalidProgram(f"{p.id}#{t}: inexact division {ins}")
            r = a // b
        else:
            raise InvalidProgram(f"{p.id}#{t}: bad op {ins}")
        if r > L:
            raise InvalidProgram(f"{p.id}#{t}: value {r} exceeds limit {L}")
        reg[d] = r
        seen.add(r)
    return seen


def verify(p: Puzzle, prog) -> int:
    """Return program length if valid and covers all targets; else raise InvalidProgram."""
    seen = run_program(p, prog)
    missing = [t for t in p.targets if t not in seen]
    if missing:
        raise InvalidProgram(f"{p.id}: {len(missing)} targets missing, e.g. {missing[:3]}")
    s = json.dumps(prog, separators=(",", ":"))
    if len(s.encode()) > MAX_BYTES:
        raise InvalidProgram(f"{p.id}: program cell too large")
    return len(prog)


def load_submission(path: str) -> dict:
    with open(path, newline="") as f:
        return {r["id"]: json.loads(r["program"]) for r in csv.DictReader(f)}


def write_submission(path: str, progs: dict):
    ids = sorted(progs)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "program"])
        for i in ids:
            w.writerow([i, json.dumps(progs[i], separators=(",", ":"))])


def score_submission(puzzles, progs) -> int:
    tot = 0
    for p in puzzles:
        tot += verify(p, progs[p.id])
    return tot
