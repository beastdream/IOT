# Experiment B baseline runtime argument correction

The failure compared different lifecycle stages. A_baseline/args.yaml is written before optimizer construction and contains warmup_bias_lr=0.1. A_baseline/framework_resolved_args.yaml and experiment.json.resolved_training_arguments agree on the actual post-optimizer value, 0.0. The previous B smoke runtime metadata also records 0.0. There was no actual warmup discrepancy between the two runs.

Installed Ultralytics 8.4.128 engine/trainer.py, build_optimizer (line 1110), assigns warmup_bias_lr=0.0 for optimizer=auto. The project previously inherited and compared against the earlier args.yaml. The custom trainer was not independently replacing arguments with defaults. No installed framework files or baseline artifacts were edited.

B now inherits the complete runtime argument dictionary, requiring agreement between the two baseline runtime records and complete key coverage against args.yaml. No warmup value is hard-coded. The post-optimizer check compares every field, including optimizer, model, device and warmup arguments. Only output identity paths, the independently validated runtime dataset YAML path, and the exactly-two-epoch smoke override may differ. Sampling configuration stays outside the framework argument dictionary. Unexpected or missing arguments are rejected. Optimizer class, all initial parameter-group settings, and parameter count remain separately checked.

Readiness requires a passing GPU smoke with matching implementation, baseline runtime arguments and optimizer provenance, verified argument report, sampler diagnostics and checkpoint hashes. It rechecks runtime arguments and optimizer metadata rather than trusting only a status flag. Explicit --overwrite archives the old smoke directory through the existing lifecycle.

Baseline actual runtime arguments:

| Field | Value |
| --- | --- |
| optimizer | auto (actual AdamW) |
| lr0 | 0.01 (auto optimizer initial group LR: 0.001667) |
| lrf | 0.01 |
| momentum | 0.937 (actual AdamW betas: 0.9, 0.999) |
| warmup_epochs | 3.0 |
| warmup_momentum | 0.8 |
| warmup_bias_lr | 0.0 |
| weight_decay | 0.0005 (weight group; bn/bias groups: 0.0) |
| close_mosaic | 10 |
| mosaic | 1.0 |
| mixup | 0.0 |
| fliplr | 0.5 |
| hsv_h / hsv_s / hsv_v | 0.015 / 0.7 / 0.4 |
| imgsz / batch / workers | 416 / 4 / 0 |
| seed / patience | 42 / 20 |

The smoke output baseline_argument_comparison.json records every runtime argument and the actual optimizer comparison. The smoke preserves all 1158 TRAIN draws per epoch and full VALID coverage. Full Experiment B must not be launched as part of this correction.

Verification completed: compileall passed; pytest passed (214 tests); failure review, sampling and dry run passed. The two-epoch GPU smoke passed on NVIDIA GeForce GTX 1650. All 116 argument fields were checked with no unexpected drift; actual optimizer groups matched. Both epochs used the weighted TRAIN sampler, validation covered all 127 images, VALID/TEST leakage was zero, and nonempty best.pt and last.pt were produced. Dataset and baseline immutability checks passed. The previous failed smoke was archived by the explicit overwrite lifecycle. No full training was executed.
