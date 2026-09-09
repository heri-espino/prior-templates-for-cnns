# Plain article manuscript

`main.tex` is the editable LaTeX article; `main.pdf` is its compiled preview. It includes the existing methods, results, figures, limitations and 24 cited references. It is an editorial manuscript, not certification that the deferred experimental audit has passed. Author names and affiliations remain to be supplied.

Run from the repository root:

```sh
bash papers/build_paper.sh
```

Requires a TeX distribution with LuaLaTeX and BibTeX. No Python, Pandoc or experiments are required to compile. The source uses standard article typography, tables, equations, figures and author–year citations with `plainnat`; no journal template is imposed. If TMLR is selected for submission, switch to its official template and bibliography style then.

Edit `main.tex` directly. `main.md` is the earlier Markdown counterpart; edits are not synchronized automatically. Figures are read from the existing analysis folders; bibliography from `literature/references.bib`. Keep both versions scientifically consistent until the Markdown counterpart is retired. Supplementary results remain in `supplementary_results.md`.
