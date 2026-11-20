#!/usr/bin/env bash
# Download public program datasets used for the per-puzzle merge baseline into ext/ (gitignored).
set -e
cd "$(dirname "$0")/.."
K=.venv/Scripts/kaggle.exe; [ -x "$K" ] || K=kaggle
for d in alexchilton/santa-2026-rudolph-programs asteriospanoras/santa-2026-elf-programs-and-experiments \
         goodpjw2008/santa-2026-dag-lns-programs rishabhbafnaiiitd/santa-2026-optimal-programs \
         konbu17/santa-2026-elf-calculator-programs royalacecat/santa-2026-retained-replay-inputs-133928; do
  n=$(echo "$d" | tr '/' '__'); mkdir -p "ext/$n"; $K datasets download "$d" -p "ext/$n" --unzip -q
done
