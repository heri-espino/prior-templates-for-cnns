# Main manuscript status

Current paper: [main.md](main.md). Complete numerical tables: [supplementary_results.md](supplementary_results.md). Literature evidence: [literature_comparison.md](literature_comparison.md).

## Completed in this revision

1. Integrated the 400-run alignment experiment, 200-run retention-release experiment and 80-checkpoint GPU robustness analysis as sequential evidence in one paper. Robustness is in the abstract, methods, results, figure, discussion and conclusion, not an appended correction.
2. Compared the specific contribution with the closest structured-filter, concept and activation-patching work in the provided corpus. Original PDFs were consulted where extraction/metadata was incomplete. The paper states the novelty boundary and avoids an unsupported first-of-its-kind claim.
3. Distinguished four locally prospective Holm-adjusted tests from later exploratory comparisons. The manuscript/supplement retain all tasks, conditions, rankings and channel sizes. Plots label SD bands versus marginal paired intervals; no exploratory interval is promoted to a corrected confirmation.

## Explicitly deferred by the owner

The final independent reproducibility review is **not complete or claimed complete**. It should recompute a representative subset, investigate discrepancies, and reconcile submission code/artifacts. Preliminary work from an interrupted earlier pass is not a submission sign-off and is excluded from this manuscript revision.

## Remaining submission preparation

Choose venue and author information; finish venue-specific reference metadata/formatting; complete the deferred review and human scientific review. The present manuscript has a focused comparison against reviewed sources, not an exhaustive proof of priority. No additional training is scheduled by this revision.

Bibliography follow-up: integrated three technically reviewed additions, with final-version corrections and explicit reading limits in literature/TECHNICAL_COMPARISON.md. This does not complete the deferred experimental review.

Plain article LaTeX version started on 9 September 2026: main.tex, compiled main.pdf, build_paper.sh and LATEX_README.md. Uses plainnat provisionally at owner request for simple article formatting; journal-specific style is deferred. Orozco-Solis (2024) added after reading the supplied PDF/text.

TMLR formatting supersedes the provisional plain article: main.tex uses the official anonymous review style and bibliography; all supplementary comparisons are included after references. See TMLR_ADAPTATION.md. This is not an experimental audit or a submission/acceptance claim.

Kernel morphology follow-up: existing Stage B checkpoints now support nearest and one-to-one comparisons, fixed-block galleries and complete endpoint matching appendix. 200 models/2000 checkpoints; saved alignment reproduced exactly. Exploratory derived evidence only; audit remains pending. Scripts and reading notes: analysis/kernel_similarity/.
