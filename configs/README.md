# S182 configuration map

The original submission reads configuration relative to `src/final_pipeline/`. This folder is a human-readable index; it does not replace those runtime files.

| Stage | Runtime configuration | Selected behavior |
|---|---|---|
| S1 | `src/final_pipeline/model/stage1_dino/runtime_config.json`, `stage1_dino_base/runtime_config.json` | Small and Base DINOv2 heads; the S163 cue is retained but made no official score difference |
| S2 collision | `src/final_pipeline/model/stage2/s118/rule.json`, `stage2/s172/config.json` | Motion candidate re-ranking, S160 localizer, S172 visual refinement; bounded runtime |
| S2 entry and side | `src/final_pipeline/model/stage2/s161/params.json` plus `s182_chain.py` | S161 crossing, S174 coverage, S176 bucket, S178 side, S180 shift, S181 fallback |
| S3 | `src/final_pipeline/model/stage3/s109/selection.json`, `stage3/s118/rule3.json` | Motion and video acceleration mixture, S141 steering |

The base S2 rule still names S161 because the final S175–S182 changes are chained in Python after that rule. Learned parameter JSON and all model weights were excluded from this repository.
