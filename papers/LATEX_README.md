# TMLR review manuscript

`main.tex` is the editable TMLR manuscript. It uses the unmodified supplied `tmlr.sty` in anonymous review mode and `tmlr.bst` for author–year citations. The class is `article`, 10pt; the official style controls margins, headings and spacing. Do not add `accepted` or `preprint` for an initial submission.

Run from the repository root:

```sh
bash papers/build_paper.sh
```

Requires pdfLaTeX and BibTeX (a standard TeX distribution). No Pandoc, Python, or experiment execution is required for the paper build. The build script locates the supplied style files under `tmlr/tmlr-style-file-main`, prioritizes the local manuscript over the template example, and writes auxiliary files under `build/`.

The appendix structure is now:

- `confirmation_robustness_results.tex`: fresh Stage-D prospective confirmation, Stage-E 1,200-model robustness map, and independent checkpoint audit;
- `supplementary_results.tex`: historical Stage A–C results explaining the transition from relative `U` to selected fidelity;
- `kernel_matching_results.tex`: exploratory one-to-one kernel-matching follow-up.

Edit LaTeX directly. Markdown counterparts are historical editorial sources and are not automatically synchronized exports.

The default manuscript is anonymous: no author/affiliation block, acknowledgments, or identifying project URL. The current public working repository should not be linked in an anonymous submission because it identifies the authors.

Scientific status as of 10 September 2026: the frozen 20-block Stage-D primary selected-fidelity budget contrast is confirmed; the separate Stage-E robustness study completed all 1,200 models; and the independent 32-checkpoint audit passed its frozen tolerances. See `MANUSCRIPT_STATUS.md` for the exact inferential hierarchy and remaining submission-packaging work.
