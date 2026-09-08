# Next actions

GPU robustness completed (80/80), uploaded at results/patch_robustness_gpu_001. See analysis/patch_robustness_gpu_001/REPORT.md. Original contrast/k4 GPU vs CPU max ΔU≈1.29e−7, accuracy identical.

Conditional replication is deferred: on two_concepts/TinyCNN, release ΔU positive at k4/8 but negative at k1/2 for all three rankings. Deeper results also change with ranking/budget. Thus broad architecture claim is measurement-dependent; do not say robustness universally passed. This is an assessment, not a preregistered pass/fail rule.

New review launcher: bash review_robustness.sh. No additional GPU run needed for already completed default80. If pursuing fresh-task training, first agree/fix a narrower budget-dependence hypothesis, full-k-curve contrast and absolute counterfactual accuracy. Do not silently run confirmation of only favorable k4. Current paper can incorporate the measurement-sensitivity finding without new training.

Main paper revision integrates literature and robustness; see papers/MANUSCRIPT_STATUS.md. Owner explicitly deferred final independent reproducibility review. Preliminary uncommitted analysis/submission_review/ exists locally from the interrupted earlier pass: numerical outputs reproduced, but strict full-ranking comparison differed below tested cutoffs in one cell. Do not count this as final validation; investigate later.
