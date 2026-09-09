# Bibliography review and targeted expansion

Reviewed 8 September 2026. This is a bibliography/coverage review, not the deferred experimental reproducibility review or an exhaustive systematic literature review.

## Existing collection and the earlier recommendation

The user's supplied earlier recommendation explains the collection: the 40 recommended works correspond to 39 local PDF/Docling pairs plus PCFNet, which is cited in the main paper but has no local PDF. The Jorgenson file has the extra `jorgenson26a` prefix; it is the intended paper, not a different work. No source files were renamed or edited.

The earlier recommendation was a useful discovery list, not a verified bibliography or a proof of novelty. In particular, phrases suggesting that known templates plus an inductive bias establish causal identifiability were too strong. The present experiments measure finite intervention outcomes, not a general identification theorem. The original use of “ground-truth templates” also needs care: the renderer supplies ground-truth object annotations, but bank resemblance does not independently establish the semantics of a learned feature. The current main manuscript already uses the narrower interpretation.

## Current verification status

`references.bib` now contains 43 works and all 23 main-paper references. The three additions are integrated after technical reading; see [the comparison and reading boundaries](TECHNICAL_COMPARISON.md). `candidates.bib` redirects to the main bibliography to avoid duplicate keys. The original 39 PDFs remain unchanged; the latest user commit adds three full extracted texts and assets but not their referenced PDFs.

The final-version check covered the ten entries previously typed as miscellaneous/preprints. Four now cite verified journal records: Bruna (TPAMI 2013, DOI 10.1109/TPAMI.2012.230), ExplainFix (WIREs 13(2):e1483, 2023, DOI 10.1002/widm.1483; online first 2022), Wang (TGRS 62:1–9, 2024, DOI 10.1109/TGRS.2024.3365910), and Poeta (ACM Computing Surveys, online 2025, DOI 10.1145/3774643; volume/pages not verified and omitted). Publisher-deposited DOI records and, for Wang, the [author institution's record](https://repository.kaust.edu.sa/items/b81d1467-c920-4f02-afdd-8750041c3908) support these changes. Local PDFs remain older supplied versions; metadata verification is not a full comparison of every textual revision.

No final archival record was verified for Debot, Gavrikov, Goyal, Heimersheim or Prisma; retain their exact arXiv versions. Margeloiu remains an ICLR workshop paper, not a main-conference acceptance. An under-review OpenReview submission (Gavrikov) and an invited workshop talk (Prisma) do not establish proceedings publication. These are bounded search outcomes, not assertions that no later publication exists.

PCFNet full text was not located in publisher/repository searches; the [OpenAlex DOI record](https://api.openalex.org/works/https://doi.org/10.1016/j.neucom.2019.11.075) reported closed access and no repository full text. The new upload does not contain it. Its existing publisher-supported claims remain limited accordingly.

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
| Bruna | Local file named 2013 contains arXiv **1203.1513v2 (2012)**. The bibliography now cites the verified 2013 TPAMI publication; the local PDF is unchanged. |
| Goyal | Key contains 2019, but the supplied **1907.07165v2 is dated 2020**. The entry cites that version, without inventing a conference venue. |
| Wang | Key contains 2024, but supplied **2308.05202v1 is dated 2023**. The bibliography now cites the verified final journal record. |
| Poeta | Key contains 2025, but supplied **2312.12936v1 is dated 2023**. The bibliography now cites the verified final journal record. The DOI record verifies an online ACM Computing Surveys publication in 2025; full-text version equivalence is not claimed. |
| FaCT | Supplied version updated in April 2026 says **accepted to NeurIPS 2025**. Conference year is 2025; versioned arXiv identifier records the supplied revision. |
| Zhang/Nanda | Confirmed **ICLR 2024**, rather than treating the 2023 initial arXiv upload as the conference year. |

## Does the bibliography need to grow?

A targeted extension is justified; a large increase or a TMLR citation quota is not. The corpus already covers structured/fixed filters, first-layer statistics, concept supervision and leakage, causal concepts, patching, and identifiability. The main paper cites 20 of these works. Corpus size and manuscript citation count serve different purposes.

Before adding more broad surveys, make fuller use of existing direct precedents: Zhang/Wu/Zhu (2018) and Varshneya et al. (2021) for regularization that promotes semantic filters; GlanceNets and Margeloiu for semantic alignment versus intended behavior. These are already available in `references.bib`. More references should sharpen an explicit comparison, not imply that existing results have gained external validation.

## Technical additions and remaining work

Miller (COLM 2024), Nicolson (TMLR 2025), and Sharma (TMLR 2026) are now integrated as references 21–23. The [technical comparison](TECHNICAL_COMPARISON.md) records inspected sections, distinctions between estimands, and proposed controls. They were selected for relevance, not a TMLR citation quota.

Use the selected LaTeX template for final citation formatting. Current target: official TMLR template and `tmlr.bst`. Full manuscript LaTeX conversion and the independent experimental submission audit remain outstanding. PCFNet full-text reading also remains outstanding.

Validation: all 43 entries processed with official `tmlr.bst` using classic BibTeX without errors or warnings; 23 manuscript reference anchors and existing local bibliography file paths checked. This is a bibliography check, not a compiled submission manuscript or experiment audit.
