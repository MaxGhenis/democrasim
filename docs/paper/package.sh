#!/usr/bin/env bash
# Build the submission package: arXiv source tarball + SSRN PDF + metadata.
# Run from docs/paper/: ./package.sh
set -euo pipefail
cd "$(dirname "$0")"

latexmk -pdf -interaction=nonstopmode -quiet democrasim.tex >/dev/null

OUT=submission
rm -rf "$OUT" && mkdir -p "$OUT"

# arXiv wants the source; it compiles the tex itself (figures included).
tar czf "$OUT/arxiv-democrasim.tar.gz" democrasim.tex figures/*.png
cp democrasim.pdf "$OUT/democrasim.pdf"   # SSRN takes the PDF directly

pdftotext -f 1 -l 1 democrasim.pdf - 2>/dev/null |
  sed -n '/^Does plurality/,/survey estimates\.$/p' |
  tr '\n' ' ' | sed 's/  */ /g' > "$OUT/abstract.txt"

cat > "$OUT/metadata.md" <<'EOF'
# Submission metadata

- Title: Elections on measured stakes: plurality welfare tracking with
  microsimulated household impacts
- Author: Max Ghenis (max@maxghenis.com)
- arXiv primary category: econ.GN (General Economics); cross-list
  candidate: cs.MA (Multiagent Systems)
- SSRN classifications: Political Economy; Public Economics; Computational
  Social Science
- Keywords: voting, welfare aggregation, misperception, Condorcet jury
  theorem, probabilistic voting, microsimulation, tax policy
- Code and data: https://github.com/MaxGhenis/democrasim
- Note for arXiv: first-time econ.GN submitters may need an endorsement;
  SSRN has no gate. Abstract in abstract.txt.
EOF

echo "wrote $OUT/: arxiv-democrasim.tar.gz, democrasim.pdf, abstract.txt, metadata.md"
