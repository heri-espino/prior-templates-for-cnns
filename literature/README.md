# Literature and BibTeX

- [Corpus index](INDEX.md): 39 supplied PDF/Docling pairs.
- [Reusable bibliography](references.bib): those 39 works plus PCFNet, already cited in the paper.
- [Review and expansion recommendation](BIBLIOGRAPHY_REVIEW.md): metadata corrections, version choices and three targeted additions.
- [Candidate bibliography](candidates.bib): separate from the existing collection and main citations.
- [Main-paper citation keys](../papers/citation_keys.md): conversion map for the current numbered Markdown references.

From a LaTeX document in `papers/`, use the venue's bibliography style and:

```latex
\cite{molaei_2020_gabor-filter-structure,linse_2024_predefined-filters}
\bibliography{../literature/references}
```

For biblatex, use `\addbibresource{../literature/references.bib}` instead. Only cited entries appear unless `\nocite{*}` is used. Include `candidates.bib` only when integrating the corresponding papers.

Keys preserve existing corpus names so links remain stable; the `year` field describes the cited version. Some files are older preprints despite a later year in their filename. `file` paths are relative to the repository root and are catalog metadata, not required by LaTeX. The `.bib` protects title capitalization and escapes author accents for classic BibTeX. Original PDFs and Docling outputs remain unchanged.
