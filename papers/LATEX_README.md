# TMLR review manuscript

`main.tex` is the editable TMLR manuscript and `main.pdf` its compiled preview. It uses the unmodified supplied `tmlr.sty` in anonymous review mode and `tmlr.bst` for author–year citations. The class is `article`, 10pt; the official style controls margins, headings and spacing. Do not add `accepted` or `preprint` for an initial submission.

Run from the repository root:

```sh
bash papers/build_paper.sh
```

Requires pdfLaTeX and BibTeX (a standard TeX distribution). No Pandoc, Python or experiment execution is required. The build script locates the supplied style files under `tmlr/tmlr-style-file-main`, prioritizes our local manuscript over the template's example `main.tex`, and writes auxiliary files under `build/`.

The appendix is `supplementary_results.tex`, included after the references. It transcribes all existing supplementary tables; no new statistical tests were performed. Figures are loaded from existing analysis folders and citations from `../literature/references.bib`. Compile from the repository rather than uploading main.tex alone to an online editor.

Edit LaTeX directly. The Markdown counterparts remain historical editorial sources, not automatically synchronized exports. The default PDF is anonymous: no author/affiliation block, acknowledgments or identifying project URL. The template's “Under review” header denotes formatting mode; it does not mean the paper has been submitted.

Formatting adaptation is complete; scientific submission sign-off is separate. See `TMLR_ADAPTATION.md` for the changes, verification and remaining author actions. In particular, the independent checkpoint audit remains deferred and an anonymous code/checkpoint supplementary ZIP has not been assembled.

Kernel follow-up: main.tex contains the exploratory matching methods/results and gallery; kernel_matching_results.tex adds all matching endpoint means and figures after the existing supplementary appendix. Regenerate these derived figures with the root run_kernel_analysis script before compiling if they are absent.
