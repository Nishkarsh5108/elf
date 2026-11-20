## Scoring

Submissions are evaluated by replaying the arithmetic program for each of the **1,000 scored puzzles**. Your score is the total number of instructions across all 1,000 programs. **Lower is better.** There is no public/private leaderboard split.

$$\text{Score} = \sum_{p=1}^{1000} |\text{program}_p|$$

Every instruction costs one, including instructions that use the same source twice, produce a value you have already calculated, or run after all targets have been covered. All instructions must be legal. There is **no partial credit**. Each program must follow the calculator rules on the Data page and produce every target at least once. If any program is invalid, the submission is rejected.

## Submission format

Submit a UTF-8 CSV file with exactly two columns, `id` and `program`, in that order. Include every ID from `scored_0001` through `scored_1000` exactly once. Row order does not affect the score.

Each `program` cell contains a JSON array of instructions. Each instruction is a four-element array:

```json
["+", 0, 1, 2]
```

This example adds the values in registers 0 and 1 and writes the result to register 2.

- The operator must be one of `"+"`, `"-"`, `"*"`, or `"/"`.
- The next two elements are integer source register indices from 0 through 47. Both registers must contain defined values.
- The final element is the integer destination register index, also from 0 through 47.
- Both source values are read before the destination is overwritten. The destination may be the same register as either source, and both sources may be the same register.

The following CSV shows a complete solution for the **toy puzzle on the Data page**. Its ID is illustrative and must not be included in a competition submission. Use `sample_submission.csv` for a complete submission with the actual scored IDs.

```csv
id,program
example_0001,"[[""/"",0,0,1],[""-"",0,1,2],[""+"",0,2,3],[""+"",0,3,4]]"
```

Use a CSV writer to escape the quotation marks inside each JSON cell.

## Program limits

| Parameter | Limit |
|---|---:|
| Registers per puzzle | 48, indexed from 0 through 47 |
| Instructions per program | 4,096 |
| UTF-8 bytes per `program` cell | 131,072 bytes (128 KiB) |
| Value of any register | A positive integer no greater than that puzzle's `value_limit` |

Every puzzle has a `value_limit` no greater than \\(10^{18}\\) The sample submission uses at most 2,873 instructions for any puzzle.
