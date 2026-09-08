# Bibliography review and targeted expansion

Reviewed 8 September 2026. This is a bibliography/coverage review, not the deferred experimental reproducibility review or an exhaustive systematic literature review.

## Existing collection and the earlier recommendation

The user's supplied earlier recommendation explains the collection: the 40 recommended works correspond to 39 local PDF/Docling pairs plus PCFNet, which is cited in the main paper but has no local PDF. The Jorgenson file has the extra `jorgenson26a` prefix; it is the intended paper, not a different work. No source files were renamed or edited.

The earlier recommendation was a useful discovery list, not a verified bibliography or a proof of novelty. In particular, phrases suggesting that known templates plus an inductive bias establish causal identifiability were too strong. The present experiments measure finite intervention outcomes, not a general identification theorem. The original use of “ground-truth templates” also needs care: the renderer supplies ground-truth object annotations, but bank resemblance does not independently establish the semantics of a learned feature. The current main manuscript already uses the narrower interpretation.

## Deliverables and verification level

- [references.bib](references.bib): all 39 supplied works plus PCFNet, with stable corpus keys, full author lists, protected titles, LaTeX accents, and DOI/URL where verified.
- [bibliography_metadata.json](bibliography_metadata.json): readable metadata snapshot and per-entry sources. The `.bib` is the file to use in LaTeX; keep this snapshot consistent when updating versions.
- [citation key map](../papers/citation_keys.md): all 20 current numbered manuscript references mapped to BibTeX keys.
- [candidates.bib](candidates.bib): three separately identified additions. They are not silently added to the manuscript or represented as already downloaded/Docling-parsed papers.

Title pages were inspected as extracted PDF text for all 39 local papers and compared with Docling headings/author blocks. Publisher/PMLR/JMLR/arXiv records were checked selectively for publication details and discrepancies. This is not a new full technical reading of every paper. Metadata matching rejected unrelated title-search hits; incomplete Crossref search results were not used to populate entries. Specific DOI lookups and supplied primary pages were used instead.

All 43 entries were processed together by classic BibTeX (`plain.bst`) with no errors or warnings; all local `file` fields exist, all 20 main references resolve, and keys are unique. This checks bibliography syntax and coverage, not every publication's latest version. Some entries deliberately cite the exact supplied arXiv version because a final publication record has not been verified. They are not asserted to be unpublished today.

## Corrections and version distinctions

| Item | Finding / decision |
|---|---|
| Linse | Cite IJCNN **2023**, with DOI `10.1109/IJCNN54540.2023.10191449`; local arXiv version is **2024**. The existing key remains `linse_2024_predefined-filters`. [Author arXiv record](https://arxiv.org/abs/2411.18388). |
| Jorgenson | Confirmed PMLR **321:166–175, 2026**, proceedings of TAG-DS 2025. Correct author order comes from the [proceedings record](https://proceedings.mlr.press/v321/jorgenson26a.html), including Timothy Doster. |
| Geiger | **JMLR**, not TMLR: 26(83):1–64, 2025. Publisher pagination differs from the supplied preprint's 1–63 header; use the [journal record](https://www.jmlr.org/papers/v26/23-0058.html). |
| Barton | **CVPR Workshops 2026**, not main-track CVPR; 3986–3995. [CVF record](https://openaccess.thecvf.com/content/CVPR2026W/XAI4CV/html/Barton_Multi-Granularity_Concept_Whitening_for_Neural_Network_Interpretability_CVPRW_2026_paper.html). |
| Özbulak | Verified SIU **2018**, pp. 1–4 via Crossref's publisher-deposited DOI record `10.1109/SIU.2018.8404757`; accents recovered from supplied author manuscript. |
| PCFNet | Publisher record confirms Neurocomputing 382:32–39, 2020; full names Yangling Ma, Yixin Luo, Zhouwang Yang. Local full text remains missing. [Publisher](https://www.sciencedirect.com/science/article/pii/S0925231219316789). |
| ProtoPNet, GlanceNets, Margeloiu, CEM | Docling's multi-column reading order omits or interleaves authors. PDF text recovers the author lists, including Chaofan Tao/Cynthia Rudin, Andrea Passerini, and Umang Bhatt/Adrian Weller respectively. Do not generate author fields from the first Markdown block alone. |
| Bruna | Local file named 2013 contains arXiv **1203.1513v2 (2012)**. This `.bib` currently cites that specific preprint; final journal version metadata can be substituted after verification. |
| Goyal | Key contains 2019, but the supplied **1907.07165v2 is dated 2020**. The entry cites that version, without inventing a conference venue. |
| Wang | Key contains 2024, but supplied **2308.05202v1 is dated 2023**. Entry cites that preprint. |
| Poeta | Key contains 2025, but supplied **2312.12936v1 is dated 2023**. Entry cites that preprint. The arXiv landing page links a later DOI (`10.1145/3774643`); final metadata and version differences remain to be checked before substituting it. |
| FaCT | Supplied version updated in April 2026 says **accepted to NeurIPS 2025**. Conference year is 2025; versioned arXiv identifier records the supplied revision. |
| Zhang/Nanda | Confirmed **ICLR 2024**, rather than treating the 2023 initial arXiv upload as the conference year. |

## Does the bibliography need to grow?

A targeted extension is justified; a large increase or a TMLR citation quota is not. The corpus already covers structured/fixed filters, first-layer statistics, concept supervision and leakage, causal concepts, patching, and identifiability. The main paper cites 20 of these works. Corpus size and manuscript citation count serve different purposes.

Before adding more broad surveys, make fuller use of existing direct precedents: Zhang/Wu/Zhu (2018) and Varshneya et al. (2021) for regularization that promotes semantic filters; GlanceNets and Margeloiu for semantic alignment versus intended behavior. These are already available in `references.bib`. More references should sharpen an explicit comparison, not imply that existing results have gained external validation.

### Three additions with a specific purpose

1. **Miller, Chughtai and Saunders (COLM 2024), _Transformer Circuit Faithfulness Metrics Are Not Robust_.** Highest priority. The primary paper directly studies faithfulness-score sensitivity to ablation methodology. Add beside Zhang/Nanda in related work and the measurement limitations. Our extension is a prior-retention treatment comparison across channel budgets in small CNNs, not the discovery that faithfulness depends on intervention design. [Paper](https://openreview.net/pdf?id=zSf8PJyQb2). Reviewed abstract and introductory methodological framing; full experiment-by-experiment comparison remains to be done.

2. **Nicolson, Schut, Noble and Gal (TMLR 2025), _Explaining Explainability: Recommendations for Effective Use of Concept Activation Vectors_.** High priority. Studies layer inconsistency, concept entanglement and spatial dependency, including the controlled Elements dataset. It supports careful definition of a concept measurement; our single-channel AUROC and map patching are not TCAV/CAV evaluation. Add beside the independent concept measurements, without implying we evaluated their method. [Paper](https://openreview.net/pdf?id=7CUluLpLxV). Reviewed abstract/introduction and publication header; use a deeper methods comparison before detailed claims.

3. **Sharma and Le (TMLR 2026), _Encoding Without Influence: Dissociating Demographic Representation from Causal Effect in Large Language Models_.** Relevant recent comparison. Separates detected representation from functional influence using interventions in language models. Add a short distinction in related work: our manipulation is a visual filter prior, with learning trajectories and channel-budget sensitivity. This makes an unrestricted claim that “representation differs from causal use” is new especially untenable. [Paper](https://openreview.net/pdf?id=TQbXHsI3Lm). Reviewed abstract/introduction/publication header; do not transplant its causal claims into our CNN experiment. Repository and PDF disagree on publication month, so the BibTeX records the agreed year 2026 only.

These three are recommended on scientific relevance. Two happen to be TMLR articles; the closest missing measurement precedent is from COLM. None independently confirms our numerical result. Their inclusion would improve positioning, not resolve the synthetic-task limitations or replace the deferred reproducibility review.

## Remaining bibliography work before submission

Verify the final publication versions for entries currently represented as exact arXiv preprints; obtain PCFNet full text if available; perform the deeper technical comparison of the three candidates; integrate only claims supported by that reading. Final venue-specific citation style should follow the selected LaTeX template. This bibliography task does not complete the experimental submission audit.
