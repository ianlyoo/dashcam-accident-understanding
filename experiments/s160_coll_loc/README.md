# S160 collision localizer



Ready: `$DATA_DIR\releases\S160_loc.zip` (1,221,717,798 bytes).



SHA256: `042e49a6dd9f1091a158becbb8107caf8ca1bb74e06d620df81f555cd262db6b`



Exact base: `$DATA_DIR\releases\S156_casc.zip`; SHA256 `e7fb70a66ad251d47c180861e2e001da7b270259c2e12ca394c6de3f1c510d70`.



## Selected export and evidence



exported HGB(126,1) + Conv(0..4)/MLP(0..4), 25 epochs, temp=1, w_nn=1, half=9.



The finalize log and exported configuration agree. The handoff description of an `nn_soft1_ep40 halves ensemble` was inaccurate: halves 0/3/6/9 were four alternative decoders. Those alternatives scored 412/411/419/423 of 750 versus S156 393. The existing final export is retained.



The fixed available-OOF fusion proxy scores 458/750 versus 393/750. It includes five Conv seeds and only three MLP seeds, while the final model has five of each. Both final and OOF AdamW weight decay are .001. This is not exact final-ensemble OOF. Prior architecture/hyperparameter/decoder searches used the same 750 public clips; selection bias remains. No new search or training was performed for this release. See selection.json for every saved summary.



Existing full export parity: 750/750, zero row/logit differences; mean localizer time 0.9358 s/clip (p95 1.1288 s). The 659/750 final-fit hit count is in-sample and is not evidence of transfer.



## Integration and public CUDA QA



Only model/stage2/s144/predict.py changed; four localizer files were added. All 93 other S156 members, including Stage1, Stage3, inference.py, requirements, and S156 cascade weights are byte-identical. Only collision_frame and entry_frame=min(S109 entry, collision) may differ.



Fresh WSL Hermes-Ubuntu CUDA runs passed on five official public sample clips and three public Nexar long clips. Unchanged Stage1/Stage3 passing QA was reused by exact member identity. The independent integration_s160.py detector/embedding/sklearn/PyTorch reference agreed on all three long collision outputs and the S156 starting frames. Offline audit blocked network calls; zero attempts.



Three injected localizer failures (weight loading, runtime exception, invalid index/nonfinite output) returned the complete exact S156 row on a public long clip. Localizer exceptions retain the already computed S156 collision. Existing base-model errors retain the inherited S156 behavior.



| Panel | Clips | S156 seconds | S160 seconds | Added seconds/clip |

|---|---:|---:|---:|---:|

| public | 5 | 20.221 | 16.355 | -0.773 |

| long | 3 | 284.176 | 154.945 | -43.077 |



These paired end-to-end timings include process-local cold loads and host/GPU scheduling noise; direct localizer row-construction and scoring/decode timings (excluding package loading) are in release.json. Whole official inference under 60 minutes is unmeasured.



Observed added localizer row/scoring work on long clips: 0.490 s/clip. The negative whole-pipeline elapsed delta is confounded by shared-machine load and cache state; it is not evidence that adding S160 speeds up S156.






## Validation and reproduction






From repository root, run `.venv/Scripts/python.exe -B candidates/s160_coll_loc/s160_stage.py S160` in a fresh staging directory, then `candidates/s160_coll_loc/launch_qa.ps1`; after QA, run `.venv/Scripts/python.exe -B candidates/s160_coll_loc/release.py`. Existing stage/ZIP paths are intentionally not overwritten. release.json binds the retained QA and artifact hashes.



Stacking note: S164 build.batch_patch applies cleanly to this S160 predict.py and the combined source compiles. Both arms modify that file, so carry both changes when stacking. Combined exported runtime still needs its own check.



Reference harness recovery: the first attempt used descriptive ensemble name E10_base as the checkpoint prefix and failed with FileNotFoundError. The saved models use E10b. The corrected reference passed on the unchanged artifact; first-attempt logs are retained under $DATA_DIR/s160_coll_loc/logs/qa_integration_attempt1.*.

