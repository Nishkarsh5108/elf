#!/usr/bin/env bash
# Download public program datasets and notebook CSV outputs into ext/ (gitignored).
# Never execute downloaded code; only merge verified CSV program files as data.
# Re-run periodically: public sources improve daily. Then: python src/merge.py out/public_merge.csv ext data/sample_submission.csv
set -e
cd "$(dirname "$0")/.."
K=.venv/Scripts/kaggle.exe; [ -x "$K" ] || K=kaggle
for d in alexchilton/santa-2026-rudolph-programs \
         asteriospanoras/santa-2026-elf-programs-and-experiments \
         goodpjw2008/santa-2026-dag-lns-programs \
         rishabhbafnaiiitd/santa-2026-optimal-programs \
         konbu17/santa-2026-elf-calculator-programs \
         royalacecat/santa-2026-replay-133780-lean \
         royalacecat/santa-2026-replay-133772-lean \
         kindasomethin/santa-2026-exact-portfolio-checkpoints \
         ozertuu/santa-2026-verified-programs \
         sacuscreed/santa-2026 \
         hanifnoerrofiq/santa-2026-elf-calculator-peephole-programs \
         coltondowning/the-elf-calculator-atlas \
         nevska/santa-2026-explorer-files "$@"; do
  n=$(echo "$d" | tr '/' '__'); rm -rf "ext/$n"; mkdir -p "ext/$n"; $K datasets download "$d" -p "ext/$n" --unzip -q || echo "failed $d"
done
for nb in royalacecat/santa-2026-public-frontier-score-133-772 \
          goodpjw2008/santa-2026-lns-cp-sat-recombine-lb-133-778 \
          asteriospanoras/santa-2026-best-public-score-a-gift-per-run \
          alexchilton/rudolph-the-red-nosed-register-allocator \
          amanatar/santa-2026-fast-validator-eda-sota-baseline \
          kindasomethin/santa-2026-exact-portfolio-frontier-133813 \
          kmmmvs/santa-2026-verified-best-of-merge; do
  n="nb_$(echo "$nb" | tr '/' '_')"; rm -rf "ext/$n"; mkdir -p "ext/$n"; $K kernels output "$nb" -p "ext/$n" -q || echo "failed nb $nb"
done
