# Literature comparison and contribution boundary

This document supports [the main manuscript](main.md), not a separate paper. Source corpus: Git commit `ca45a5e` (the owner's Docling Markdown and original PDFs). The index contains 39 parsed papers. The closest sources below were read selectively by abstract/introduction and relevant methods; this is a focused comparison, not a systematic review or an assertion that every page of the corpus was read.

## Precise claim

Within the tested small CNNs and rendering distributions, retaining or releasing a weight-space template prior changes prediction, kernel alignment and independently measured concept properties. The relative intervention-fidelity advantage of release over constant retention can reverse with the number of patched channels, and depends on ranking and architecture. The study connects these training manipulations and measurement choices in a controlled empirical setting.

It does **not** claim invention of structured initialization, trainable Gabor parameterization, fixed convolutional features, concept alignment, activation patching, granularity sensitivity, or the distinction between accuracy and interpretability. It does not establish that no prior publication has made the same narrower empirical observation. No matched performance benchmark against the reviewed architectures was run.

## Structured-filter and optimization precedents

| Source (manuscript reference) | Inspected evidence | Overlap and contribution boundary |
|---|---|---|
| Özbulak & Ekenel [1] | Abstract; proposed initialization and CNN sections | First-layer Gabor initialization precedes us. Our question concerns independent concept/intervention outcomes under continued retention and release. |
| Molaei & Shiri Ahmad Abadi [2] | Abstract; §2.4; original PDF p.7 | Parameter updates preserve the Gabor family. Docling equations contain layout artifacts; the PDF confirms parameter-space updates. Our soft penalty on free kernel weights is different, and we do not claim the first structure-preserving training scheme. |
| Ma, Luo & Yang / PCFNet [3] | Publisher abstract/introduction read in the preceding review; DOI 10.1016/j.neucom.2019.11.075 | Learnable predefined first-layer image filters already connect priors and data efficiency. Not in the uploaded corpus; no claim of a full-paper re-review from Docling. |
| Wang & Alkhalifah [4] | Abstract; learnable/constrained first-layer formulation | Task-related Gabor parameterization and restrictions already target noisy, limited-data generalization. Our artificial shapes and edge/corner/ring anchors are not a replication of their seismic application. |
| Linse, Barth & Martinetz / PFCNN [5] | Abstract; Introduction; predefined-filter modules | Fixed spatial filters plus learned pointwise mixtures can support recognition. Our shallow frozen deficit is not evidence against this architecture. Corpus filename says2024; arXiv lists an IJCNN2023 journal reference, so the filename is not used as publication metadata. |
| Gaudio et al. / ExplainFix [6] | Abstract; §§2–3; §4.1 and §4.2 | Fixed spatial filters, steering, pruning and computational benefits are established. We test a different intervention/retention question; our code has no evidence of superior training speed versus ExplainFix. |
| Gavrikov & Keuper [7] | Abstract; Introduction; pointwise-combination motivation | Trainable linear combinations of random filters can be expressive. This motivates keeping downstream optimization distinct from the initial bank's appearance. |
| Chowers & Weiss [8] | Abstract; Introduction; energy-profile explanation | First-layer spectral patterns can reflect input statistics even without meaningful labels. Our spectrum controls narrow this confound but do not establish that semantic geometry alone causes any effect. |
| Jorgenson et al. [9] | Abstract; static/dynamic-access framing; §5.1; PDF first page | Global data properties can leave filter-weight signatures. This is not object-level causal usefulness. The subsequent bibliography review confirms PMLR 321:166–175 (2026), TAG-DS 2025 proceedings; see ../literature/BIBLIOGRAPHY_REVIEW.md. |

## Concept and intervention precedents

| Source | Inspected evidence | Overlap and contribution boundary |
|---|---|---|
| Bau et al. / Network Dissection [10] | Abstract; Introduction | Unit–concept measurement is established. Our localization is an adapted, separately selected renderer-mask measure, not a new general dissection algorithm. |
| Bau et al. / Causal Units [11] | Opening summary; classifier/generator intervention motivation | Relating concept units to causal behavior in vision is already studied. We manipulate the training prior and compare intervention budgets, rather than claim first causal unit analysis. |
| Chen, Bei & Rudin / Concept Whitening [12] | Abstract; Introduction | Aligning activation axes to known concepts differs from our kernel-weight similarity. We do not use concept whitening or claim equivalent semantic guarantees. |
| Debot & Marra [13] | Abstract; Introduction; representation versus functional distinction | Accuracy/interpretability distinctions and explicit control of sidechannel reliance already exist. Our CNN has no concept bottleneck or sidechannel mechanism; comparisons are conceptual rather than performance baselines. |
| Parchami-Araghi et al. / FaCT [14] | Abstract; Introduction | Faithful model-inherent concept traces and concept consistency are existing goals. Our sampled intervention scores neither recover such traces nor certify equivalent faithfulness. |
| Goyal et al. / CaCE [15] | Abstract; Introduction; causal concept definition | Concept changes and correlations are distinct. Our renderer constructs matched changes, but U measures internal reproduction of model responses, not the CaCE estimand. |
| Zhang & Nanda [16] | Abstract; Introduction; §§5–6 | Corruption, metric and joint-patching/window-size choices can change localization. This is the closest methodological precedent; our contribution is the concrete prior-training comparison and channel-budget reversal in CNNs, not the claim that granularity can matter. |
| Heimersheim & Nanda [17] | Abstract; §§2.1–2.3 and §3 | Exploratory versus verification evidence, intervention granularity, and necessity/sufficiency distinctions constrain interpretation. Our same-layer swaps are not circuit verification. |
| Geiger et al. [18] | Abstract; framework scope and contents | Causal abstraction formalizes relations among interventions and higher-level descriptions. We do not establish a full abstraction map or identification theorem. |
| Méloux et al. [19] | Abstract; identification question | Non-unique circuits/interpretations in enumerable small systems warn against uniqueness claims. Our size-dependent score is not a proof of their formal non-identifiability result. |
| Sutter et al. [20] | Abstract; Introduction | Expressive alignment maps can make abstraction evidence uninformative. Our restricted channel copying is a different intervention family; neither confirms nor refutes their theorem. |

All local references resolve from the [main manuscript bibliography](main.md#references) to both parsed text and original PDF. Direct quotations and corrupted extracted equations were not copied into the paper. Citation metadata is intentionally conservative where the supplied document does not establish a publication venue/date. Remaining venue-specific bibliography formatting is editorial work.

## Research positioning

The most defensible claim is a controlled empirical characterization at the intersection of two established lines of work. The first-stage four corrected tests are inconclusive; the later release and ranking/size results are exploratory. The complete negative, positive and uncertain contrasts are retained. This is a candidate contribution, not a certificate of global novelty or publication acceptance.

## Additional full-text comparison

References 21–23 (Miller, Nicolson, Sharma) are integrated in the main paper. See [technical reading notes](../literature/TECHNICAL_COMPARISON.md) for inspected sections, protocol/estimand differences and remaining controls. The new sources constrain novelty; they do not validate our numerical outcomes.
