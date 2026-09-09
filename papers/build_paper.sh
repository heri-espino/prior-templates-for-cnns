#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build
lualatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
sed 's@../literature/references@../../literature/references@' build/main.aux > build/bibliography.aux
(cd build && bibtex bibliography)
cp build/bibliography.bbl build/main.bbl
lualatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
lualatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
cp build/main.pdf main.pdf
printf 'Paper compiled: papers/main.pdf\n'
