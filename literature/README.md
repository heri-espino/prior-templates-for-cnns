# Literature and BibTeX

- [Corpus index](INDEX.md): 39 supplied PDF/Docling pairs.
- [Reusable bibliography](references.bib): 43 works, including the three technically reviewed additions.
- [Review and expansion recommendation](BIBLIOGRAPHY_REVIEW.md): metadata corrections, version choices and three targeted additions.
- [Candidate bibliography](candidates.bib): redirects to references.bib; the three candidates are now integrated.
- [Main-paper citation keys](../papers/citation_keys.md): conversion map for the current numbered Markdown references.

From a LaTeX document in `papers/`, use the venue's bibliography style and:

```latex
\cite{molaei_2020_gabor-filter-structure,linse_2024_predefined-filters}
\bibliography{../literature/references}
```

For the current TMLR target, follow the [official template](https://github.com/JmlrOrg/tmlr-style-file) and use `\bibliographystyle{tmlr}` with its citation commands. Do not replace the template bibliography system with biblatex. Only cited entries appear unless `\nocite{*}` is used. Load only `references.bib`; the candidates have been integrated.

Keys preserve existing corpus names so links remain stable; the `year` field describes the cited version. Some files are older preprints despite a later year in their filename. `file` paths are relative to the repository root and are catalog metadata, not required by LaTeX. The `.bib` protects title capitalization and escapes author accents for classic BibTeX. Original PDFs and Docling outputs remain unchanged.

[Technical reading and experimental implications](TECHNICAL_COMPARISON.md).
