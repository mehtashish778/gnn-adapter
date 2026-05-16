#!/usr/bin/env bash
# Build docs/academic_report.{tex,pdf} from docs/academic_report.md
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MD="${ROOT}/docs/academic_report.md"
TMP="$(mktemp "${TMPDIR:-/tmp}/academic_pandocXXXX.md")"
cleanup() { rm -f "$TMP"; }
trap cleanup EXIT

python3 "${ROOT}/scripts/md_tex_math_delims_for_pandoc.py" < "$MD" > "$TMP"

COMMON=( -s
  -V "geometry:margin=2.54cm"
  -V documentclass=article
  -V fontsize=11pt
  -V colorlinks=true
  # Broader Unicode coverage for ≈, τ, ∈ in body and verbatim (fallback if fonts missing locally).
  -V mainfont="DejaVu Serif"
  -V monofont="DejaVu Sans Mono"
)

pandoc "$TMP" "${COMMON[@]}" -o "${ROOT}/docs/academic_report.tex"

PDF_ENGINE="${PDF_ENGINE:-xelatex}"
pandoc "$TMP" "${COMMON[@]}" --pdf-engine="$PDF_ENGINE" \
  --pdf-engine-opt=-interaction=nonstopmode \
  -o "${ROOT}/docs/academic_report.pdf"

echo "Wrote ${ROOT}/docs/academic_report.tex and ${ROOT}/docs/academic_report.pdf (engine=$PDF_ENGINE)"
