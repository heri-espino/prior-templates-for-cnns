# Exploratory diagnostic added after partial outcomes

After examining a partial table of 189 runs, the frozen shallow models showed substantial accuracy recovery when their classifier alone was refitted. To explore whether this optimization change also alters intervention fidelity, evaluate the already validation-selected alternative head under the **same** test pairs, concept channel rankings, random rankings and patching metrics.

Scope: template_init, template_retention_1, frozen_templates, frozen_spectrum and frozen_random_unitnorm, across both tasks, both architectures and ten blocks. There is no new fitting, no new head selection rule, no change to the main model, and no modification to any prospective primary test. These are 200 additional evaluations, not 200 additional training runs.

Keep all metrics in separate files. Report descriptive changes in counterfactual accuracy and U after replacing the classifier. This diagnostic is explicitly exploratory, was motivated by partial outcomes, and cannot be used as a new confirmatory claim. Head refitting also changes probability calibration, so a fidelity difference need not indicate a changed first-layer representation.

Additional descriptive tables report alignment of the already validation-selected channels themselves and mean squared input probability effect. These do not affect channel selection or the primary estimand.
