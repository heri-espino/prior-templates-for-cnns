#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build
TMLR_STYLE_DIR="$PWD/tmlr/tmlr-style-file-main"
export TEXINPUTS=".:$TMLR_STYLE_DIR:${TEXINPUTS:-}"
export BSTINPUTS="$TMLR_STYLE_DIR:${BSTINPUTS:-}"
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
sed -e 's@../literature/references@../../literature/references@'  build/main.aux > build/bibliography.aux
(cd build && bibtex bibliography)
cp build/bibliography.bbl build/main.bbl
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
cp build/main.pdf main.pdf
printf 'Paper compiled: papers/main.pdf\n'
