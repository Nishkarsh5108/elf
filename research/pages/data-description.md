## Files

**puzzles.csv** - Contains the 1,000 scored puzzles. Each row is a separate puzzle.

- `id` - unique puzzle identifier, from `scored_0001` through `scored_1000`.
- `starting_values` - JSON array of starting positive integers. Values occupy registers 0 through `len(starting_values) - 1` at the start.
- `targets` - JSON array of positive integers the program must produce at least once, in increasing order.
- `value_limit` - largest value allowed in any register during execution. This is an integer between 1 and  \\(10^{18}\\).
- `registers` - number of available registers. This is 48 for every puzzle.

**practice_puzzles.csv** - Contains 200 unscored puzzles, with IDs from `practice_0001` through `practice_0200`. It has the same five puzzle columns as `puzzles.csv`, plus:

- `family` - generator family: `independent`, `shared`, `multiplicative`, `mixed`, or `tiny`.
- `design_tier` - generator difficulty tier: `hard`, `medium`, or `easy`.
- `lower_bound` - number of targets not already present in `starting_values`, calculated as `len(set(targets) - set(starting_values))`. A single instruction can produce at most one new target, so this is a lower bound on the instruction count. It is generally loose and is not guaranteed to be achievable.

The practice set uses the same puzzle families and arithmetic rules as the scored set, with a larger share of tiny puzzles for debugging. Family proportions differ between the two sets.

**sample_submission.csv** - Contains a valid baseline program for every puzzle. 

## Calculator rules

Each puzzle has 48 registers, indexed from 0 through 47. At the start, the values in `starting_values` occupy the first registers in their listed order. All other registers are undefined.

An instruction `["op", i, j, d]` reads source registers `i` and `j`, applies the operator, and writes the result to destination register `d`. Writing to a register replaces its previous value. Both operands are read before the write, so the destination may be one of the source registers.

1. **Values must be positive integers.** Subtraction is legal only when the first operand is greater than the second.
2. **Division must be exact.** The second operand must divide the first with no remainder.
3. **Every result must be at most `value_limit`.** This includes intermediate values, even if a later instruction would reduce them.
4. **Source registers must be defined.** An undefined register becomes defined after an instruction writes to it.
5. **Targets only need to appear once.** A target is covered when its value appears in any register, including at the start. It remains covered after that register is overwritten.

Repeated source operands are allowed. For example, `["+", 3, 3, 7]` doubles the value in register 3, and `["/", 0, 0, 4]` produces 1 from any positive value in register 0. Initial registers may also be overwritten.

All instructions count toward the score, including those after the last target has been produced. The Evaluation page lists the instruction and cell-size limits.

## Worked example

This small example illustrates the file format and rules. It is not one of the scored puzzles.

```csv
id,starting_values,targets,value_limit,registers
example_0001,[3],"[1, 2, 5, 8]",9,48
```

The following program covers all four targets in four instructions:

```json
[["/", 0, 0, 1], ["-", 0, 1, 2], ["+", 0, 2, 3], ["+", 0, 3, 4]]
```

| Step | Calculation | Destination | Covered targets |
|---:|---|---|---|
| 1 | 3 / 3 = 1 | Register 1 | 1 |
| 2 | 3 - 1 = 2 | Register 2 | 1, 2 |
| 3 | 3 + 2 = 5 | Register 3 | 1, 2, 5 |
| 4 | 3 + 5 = 8 | Register 4 | 1, 2, 5, 8 |
