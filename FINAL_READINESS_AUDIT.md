# Final Readiness Audit — Mars and Moon CNN/QCNN Experiments

**Workspace:** `D:\Mars-QCNN`  
**Scope:** Read-only review of existing experiment records and a read-only consistency check of saved Mars prediction pairs. No training was launched. The only new file from this audit is this report.

## Verdict

**READY AFTER DOCUMENTATION FIXES.** The saved artifacts support a careful descriptive coursework report on the experiments as run. Before submission, make the reporting corrections in Section 3: clearly designate the repaired 15-run Mars evaluation as the primary Mars result; use the corrected Moon CNN parameter count; and describe the different training regimes and non-comparable latency measurements. No new experiment is required if the claims remain descriptive and do not attribute gains solely to a quantum circuit. A stronger architecture-only or site-generalization claim would require additional controlled data/experiments.

## 1. Ready for reporting

### Mars

The primary Mars evidence is the repaired controlled all-site run in `runs\pipeline_repair_audit\retraining_seedset`. Its `progress.json` marks all 5 leave-one-site-out folds × seeds 42, 43, and 44 complete. I verified that there are 15 test prediction files for each model and, for all 15 fold/seed pairs, CNN and CNN→QCNN files have identical ordered tile IDs, safe labels, and three terrain targets (45,699 paired rows total).

The task predicts three **terrain proxy targets** from real HiRISE orthoimage/DTM patches: 5 m mean slope (degrees), roughness (meters), and hazard fraction. “Safe” is a derived proxy label: mean slope <8° and hazard fraction <0.05. It is not an observed landing outcome or validated safety assessment. The dataset has five sites: 15,233 coverage-eligible 2 m/px patches overall; leave-one-site-out folds use three sites for training, one for validation, and one for test.

The CNN is a randomly initialized ResNet-18 trained end-to-end (11,242,563 trainable parameters). The hybrid uses the trained CNN representation frozen and fits an 8-dimensional bottleneck plus an 8-qubit QCNN head (4,152 trainable parameters total in the head, 48 in its final head component). Thus the real-data experiment compares a full end-to-end CNN with a small head on a supervised frozen representation; it is a pipeline comparison, not an architecture-only or equal-capacity comparison.

In the repaired three-seed/five-site aggregate, the QCNN has lower equal-site regression RMSE and MAE for all three targets:

| Target | CNN RMSE / MAE | CNN→QCNN RMSE / MAE | Numerically lower |
|---|---:|---:|---|
| 5 m mean slope (°) | 10.250 / 8.712 | 8.190 / 6.837 | QCNN |
| Roughness (m) | 1.566 / 1.172 | 1.364 / 1.027 | QCNN |
| Hazard fraction | 0.453 / 0.392 | 0.377 / 0.314 | QCNN |

The safety trade-off depends on the decision rule:

- At the fixed risk threshold, CNN mean safe recall / unsafe FPR / predicted-safe coverage are **5.56% / 5.27% / 5.67%**; QCNN gives **2.77% / 2.78% / 3.02%**. QCNN is more conservative at this operating point but has lower safe recall and coverage.
- At validation-selected thresholds targeting validation FPR ≤5%, test-site macro safe recall / unsafe FPR / coverage are **43.33% / 39.65% / 41.35%** for CNN and **30.62% / 33.94% / 33.86%** for QCNN. The validation FPR target did not transfer to test sites.

Numerically, the QCNN reduces the reported regression errors and false-positive rate, but the CNN has higher safe recall and coverage. Paired, equal-site 95% t intervals for QCNN-minus-CNN cross zero for each reported regression and safety comparison. Example regression intervals are slope RMSE [-8.11, 3.99], roughness RMSE [-0.847, 0.444], and hazard RMSE [-0.274, 0.121]. These results therefore do not identify a clear overall winner.

### Moon

The completed LROCNet experiment uses the supplied train/validation/test split and fixed folder mapping `ejecta=0`, `oldcrater=1`, `none=2`. It has 3,608 train, 613 validation, and 779 test images. The images are grayscale content encoded as RGB. The independent saved-results audit reports identical ordered test sample IDs and labels across the three seeds and verifies that saved predictions match logits and saved metrics/confusion matrices.

Mean ± sample SD across seeds 42, 43, and 44 on the same test partition:

| Model | Accuracy | Balanced accuracy | Macro-F1 |
|---|---:|---:|---:|
| CNN | 0.530 ± 0.035 | 0.558 ± 0.053 | 0.465 ± 0.024 |
| Hybrid CNN–QCNN | 0.529 ± 0.039 | 0.608 ± 0.006 | 0.497 ± 0.026 |
| Frozen-embedding MLP | 0.451 ± 0.037 | 0.587 ± 0.021 | 0.440 ± 0.028 |
| Train-majority baseline | 0.684 ± 0.000 | 0.333 ± 0.000 | 0.271 ± 0.000 |

The hybrid’s mean balanced accuracy and macro-F1 are higher than the CNN’s here, while accuracy is nearly the same. Three seeds on one supplied split are descriptive evidence, not proof of a general advantage. The matched frozen-embedding MLP is the more relevant classical head comparison for the QCNN component, but it is still one test partition and only three seeds.

## 2. Limitations that must be disclosed

### Mars

- Five sites are the geographic replication units. Tiles within a site are not independent site-level observations. Three random seeds do not create more independent sites.
- The aggregate 95% t intervals are based on five site-level values (site means incorporating the three seeds), not on independent tile-level sampling. They are descriptive and unstable at n=5. Several interval bounds fall outside valid probability/count ranges; do not present them as bounded probability intervals or decisive significance tests. Metrics with undefined denominators are averaged only over the sites where they are defined.
- Site distributions differ substantially and the fixed/validation-calibrated safety operating points have low recall or high held-out FPR. These are terrain-proxy results, not evidence for operational landing safety.
- QCNN consumes a frozen supervised CNN representation and has far fewer trainable parameters. No real-data matched classical-head control is present in this primary Mars comparison, so numerical differences cannot be attributed specifically to quantum computation.
- The older `runs\controlled_evaluation\protocol_report.md` is a one-seed seed-42 result. It should not be mixed into the repaired three-seed aggregate. The old `runs\real_v1` / `runs\real_v1_qcnn` fold-0 outputs are also a separate earlier comparison with documented augmentation, resume, and preprocessing confounds in `runs\post_training_audit\audit_report.md`.

### Moon

- Training has 73.61% `none` images and test has 68.42% `none` images. The majority baseline’s 68.42% test accuracy is correct but predicts no minority classes (balanced accuracy 0.333, macro-F1 0.271). Accuracy alone is therefore misleading.
- The CNN is end-to-end trained (59,347 trainable parameters); the hybrid and frozen-embedding MLP use a frozen CNN encoder and train small heads. Their input representation and optimization regime differ from the CNN. The hybrid head has 568 trainable parameters versus 547 for the MLP control. This can support a frozen-embedding head comparison, not a clean end-to-end CNN-versus-QCNN attribution.
- CNN timing includes image-encoder inference, while recorded QCNN timing is for the head on already-computed embeddings. It excludes the encoder and is not an apples-to-apples end-to-end latency comparison. Peak VRAM was not recorded.
- The completed run was generated before test dataset/embedding access was moved to the final evaluation block in the trainer. Test embeddings were computed before fitting the QCNN/MLP heads, but were not used for gradient updates, validation selection, or threshold selection; the saved audit confirms final metrics came from saved predictions. State this accurately rather than claiming test inputs were untouched until all model fitting was complete.
- The provenance audit did not verify class annotation criteria, original split-generation methodology, source-image identity, or geographic/site identity. Filename prefix disjointness is only an unverified heuristic. The results do not establish source-group or site-independent generalization.
- The three-seed mean ± SD is not a confidence interval. The same fixed test partition is reused across seeds, so seeds reflect initialization/training variation, not independent test samples.

## 3. Genuine issues requiring correction before submission

These are reporting corrections; existing experiment files should remain unchanged.

1. **Use the correct Moon CNN parameter count.** The saved per-seed `runs\summary.json` records CNN `trainable=195` because that count was taken after freezing the encoder for the hybrid stage. The independent audit correctly establishes **59,347 trainable parameters during CNN fitting**, with the 195-parameter classifier remaining trainable after encoder freeze. Quote the corrected training count and explain the distinction. Do not copy the misleading field without its context.
2. **Identify the primary Mars run explicitly.** Use repaired outputs in `runs\pipeline_repair_audit\retraining_seedset\metrics` for the final three-seed/five-site results. Label the earlier one-seed `runs\controlled_evaluation` report and fold-0 `runs\real_v1*` results as historical, separate analyses. Their metrics and preprocessing status differ.
3. **Describe Mars uncertainty and safety performance faithfully.** State that the intervals are nominal five-site t intervals, cross zero for paired model differences, and are not strong inference. Report recall together with FPR/coverage; do not frame low FPR alone as safety success.
4. **Do not compare Moon inference times as end-to-end.** Report them as separately measured CNN and cached-embedding QCNN-head timings, or omit direct latency ranking. Do not state a quantum speedup.
5. **Qualify the Moon test-set timing.** The completed run did not use test labels for training or model selection, but it computed test embeddings before fitting the hybrid heads. The current trainer was changed afterward to defer test access. Make the coursework narrative refer to the actual completed run, not only the current source behavior.

No new experiment is necessary for a descriptive report if these constraints are followed. If the coursework intends a claim that the quantum component itself improves performance, the smallest appropriate next experiment is a predeclared, repeated-seed comparison of QCNN and parameter-matched classical heads on identical frozen embeddings, splits, optimizer, stopping rule, and evaluation; the lunar MLP control is a start, but three seeds and one provenance-uncertain split are not enough for a generalization claim. For Mars, add an analogous matched classical head on the same fold-specific embeddings before making any quantum-specific claim. Neither experiment was run in this audit.

## Evidence index

- Repaired Mars run completion and seeds: `runs\pipeline_repair_audit\retraining_seedset\progress.json`.
- Repaired Mars protocol/config and metrics: `runs\pipeline_repair_audit\retraining_config.yaml`, per-fold/seed `runs\...\config_snapshot.yaml`, and `runs\pipeline_repair_audit\retraining_seedset\metrics\{fixed_threshold_aggregate.csv,validation_calibrated_test_aggregate.csv,fixed_threshold_paired_site_differences.csv,validation_calibrated_test_paired_site_differences.csv}`.
- Repaired Mars model and preprocessing details: `runs\pipeline_repair_audit\retraining_seedset\runs\model1_resnet18\fold_0_seed_42\param_counts.json`, `runs\pipeline_repair_audit\retraining_seedset\runs\model2_qcnn\fold_0_seed_42\param_counts.json`, `runs\pipeline_repair_audit\repair_report.md`, and `runs\pipeline_repair_audit\retraining_preflight.md`.
- Earlier Mars protocol and metric definitions: `runs\controlled_evaluation\protocol_report.md`; prior fold-0 audit: `runs\post_training_audit\audit_report.md`.
- Moon protocol/results: `data\raw\moon\lrocnet\experiment_lunar_cnn_qcnn\README.md`, `config.yaml`, `runs\summary.json`, and per-seed `runs\logs\seed_*\results.json`.
- Moon independent audit and parameter-count correction: `data\raw\moon\lrocnet\experiment_lunar_cnn_qcnn\runs\results_verification\results_audit.md`, `results_audit.json`, and `recomputed_metrics.csv`.
- Moon source-split uncertainty: `data\raw\moon\lrocnet_provenance_report.txt`.
- Read-only Mars pair check performed for this audit: all 15 fold/seed prediction pairs had identical ordered tile IDs, safe labels, and terrain targets; 45,699 rows checked.

**Final verdict: READY AFTER DOCUMENTATION FIXES.**
