#!/usr/bin/env bash
# Download public program datasets used for the per-puzzle merge baseline into ext/ (gitignored).
# Re-run periodically: public sources improve daily. Then: python src/merge.py out/public_merge.csv ext data/sample_submission.csv
set -e
cd "$(dirname "$0")/.."
K=.venv/Scripts/kaggle.exe; [ -x "$K" ] || K=kaggle
for d in alexchilton/santa-2026-rudolph-programs asteriospanoras/santa-2026-elf-programs-and-experiments \
         goodpjw2008/santa-2026-dag-lns-programs rishabhbafnaiiitd/santa-2026-optimal-programs \
         konbu17/santa-2026-elf-calculator-programs royalacecat/santa-2026-replay-133860-lean \
         sacuscreed/santa-2026 hanifnoerrofiq/santa-2026-elf-calculator-peephole-programs \
         coltondowning/the-elf-calculator-atlas nevska/santa-2026-explorer-files "$@"; do
  n=$(echo "$d" | tr '/' '__'); rm -rf "ext/$n"; mkdir -p "ext/$n"; $K datasets download "$d" -p "ext/$n" --unzip -q || echo "failed $d"
done
