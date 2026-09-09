# Repository citation and archival DOI

Checked against primary guidance on 9 September 2026.

## Anonymous review version

TMLR requires anonymized submissions and supplementary material; the latter may be PDF/ZIP up to 100MB. The current identifying repository should therefore not be linked from the anonymous PDF. Prepare a separate review archive or genuinely anonymized access point, inspect its contents/metadata, and only then replace the current availability statement with the actual location. The package has not been assembled in this editorial pass. [TMLR author guidelines](https://jmlr.org/tmlr/author-guide.html).

## Public archival version

Recommend keeping GitHub for development and archiving the exact release associated with the paper on Zenodo. The repository is https://github.com/heri-espino/prior-templates-for-cnns. A DOI provides a persistent citation to an archived record; it does not certify reproducibility. Cite the archived version with its actual authors, title, version, date and DOI, and retain GitHub as the development link.

Zenodo permits reserving a DOI in a draft before publication; DOI registration occurs when the record is published. A reservation alone is not a public, retrievable artifact. No DOI has been reserved or published for this work in this task. [Zenodo DOI documentation](https://help.zenodo.org/docs/deposit/describe-records/reserve-doi/).

Concrete release preparation:

1. Finish the independent audit and freeze the exact source/result revision used by the manuscript. Document dependencies, commands, expected outputs and hashes.
2. Package author-created code, renderer configuration, relevant checkpoints, original result tables, analysis scripts and the required figures. Do not automatically archive the entire development repository: exclude third-party literature PDFs/extractions, private handoff material, credentials and transient environments. Preserve applicable software licenses; the authors must confirm their own release license and creator metadata.
3. Create the Zenodo draft with the verified author list, version and rights. Reserve a DOI if it is needed inside the release files, inspect the archive, and publish only when the snapshot is final.
4. Check that the DOI resolves to the correct files and that a fresh environment can execute the documented commands. Add the real software/data citation to the public manuscript and a CITATION.cff to the released source. Do not insert a made-up DOI or unverified authors now.

A named Zenodo record or GitHub release can reveal authorship. Keep any identifying citation out of the anonymous review PDF; include it in the public/camera-ready version when appropriate. The archival DOI is recommended but is not a substitute for the unfinished experimental review or an anonymous review package.
