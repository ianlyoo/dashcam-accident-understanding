# S167 stack, phase 1



Release: `$DATA_DIR/releases/S167_stk.zip` (1,217,087,211 bytes).

SHA256: `057ac49a7e0f1aeb571b2f6973e290813c77172a80f7cc89fae254f8ff9481cb`.



Exact S156 carrier + S163 one-way Stage1 recapture cue + S164 runtime patch.

S160 collision and S162 acceleration are not included in phase 1.

Member-by-member decompressed SHA256 comparison finds 92 original members

unchanged, two changed (`inference.py`, `model/stage2/s144/predict.py`), and

three added (`model/stage1/s163/cue.py`, `model/stage1/s163/model.json`,

`model/runtime_s164/runtime.py`). All inherited weights and requirements match.

`member_manifest.json` records every base-arm and composed member identity.



The only cross-arm overlap is `inference.py`. Its exact bytes are S156 followed

by S163's original append and then S164's original append. S164 wraps the final

Stage2/3 functions; S163 wraps only Stage1. Final dispatch ASTs match their arms.

The S144 detector-loop change and all new files are copied byte-for-byte from

their arm ZIPs. Detector batching stays at its default 1.



Real offline WSL CUDA QA imported the staged export through an unregistered

dynamic loader. Stage1 matches S163 on 10 public clips plus all 3 prior cue

cases (the repaired recapture, known CCD false positive and safe original).

Stage2 matches S156 on 5 public and 7 existing Nexar clips; Stage3 matches S156

on all 5 full public clips / 5,992 rows. Every CSV matches reference bytes.

No network attempts, runtime prefetch errors, lazy-scene errors or Stage3 visual

fallbacks occurred. Public/reused training panels are integration evidence,

not a fresh accuracy holdout. S163's known false positive remains a score risk.



| Same panel | Rows | S156 seconds | S164 seconds | S167 first-pass seconds |

|---|---:|---:|---:|---:|

| stage1_public | 10 | 37.20 | 37.29 | 229.66 |

| stage2_public | 5 | 45.66 | 36.21 | 17.16 |

| stage2_long | 3 | 282.55 | 152.31 | 98.04 |

| stage2_cascade | 4 | 375.33 | 197.04 | 118.26 |

| stage3_public | 5992 | 184.03 | 126.48 | 105.37 |



Matched-panel total: 924.77s -> 568.48s

(38.5% reduction). S164's prior total was

549.33s. Scaling the requested

S156 reported ~49m40s by this measured ratio estimates **30m32s**.

This is a conditional projection, not measured server runtime. Baseline timings

are reused S164-worker measurements on the identical panels; current shared

load, hidden workload, storage and six-process server execution differ. QA

uses two processes and serial DataLoader loading in both reference and stack.

Stage2/3 runtime code is byte-identical to S164. Apparent extra gains relative

to its historical run are run-condition variation, not a new optimization.

Timing excludes framework import and the extra three Stage1 diagnostic cases.

The first Stage1 pass took 229.66s

during an overlapping DeepVoice 14 GB smoke lease (near-full physical GPU use).

It passed parity and is preserved in `original_qa_reports` and the table.

A separate Stage1 timing repeat was completed: True.

That repeat took 31.64s. Substituting it gives

a diagnostic panel total of 370.46s and ratio-scaled

server time of 19.90 minutes.

The first complete pass remains the primary comparison; the repeat does not

replace it or establish an isolated server-speed estimate.

Any repeat uses the same source and public inputs; no prediction was selected

or changed based on timing. `qa_timing` holds the separate repeat evidence.

The supplied ~49m40s is an inherited assumption, not independently verified by

this integration task. S164's more conservative motion-only projection is

~43m30s before S163's small workload-dependent CPU cue overhead.





RSS peaked at 2.746 GiB

(sampled every .25s), maximum 2 processes;

CUDA reserved peaked at 1.432 GiB.

The Torch allocator cap was 1.75 GiB; CUDA reserved excludes driver/context.

Static submission validation, full ZIP CRC/member hash checks and exact

staged-tree identity passed. No private evaluation data, upload or commit.



Reproduction from repo root, using `.venv/Scripts/python.exe -B`:



```text

python -B candidates/s167_stack/build_stack.py --action stage --arms $DATA_DIR/releases/S163_s1.zip $DATA_DIR/releases/S164_fast.zip

powershell -NoProfile -File candidates/s167_stack/launch_qa.ps1

powershell -NoProfile -File candidates/s167_stack/launch_qa.ps1 -RepeatStage1

python -B candidates/s167_stack/build_stack.py --action package --arms $DATA_DIR/releases/S163_s1.zip $DATA_DIR/releases/S164_fast.zip

python -B tools/validate_submission.py $DATA_DIR/releases/S167_stk.zip

python -B candidates/s167_stack/finalize.py

```



Run long operations detached with logs as in `launch_qa.ps1`. Builder accepts

`--arms` as a list and always orders S164 last. Phase 2 uses a new `--out` and

`--zip` to preserve phase-1 evidence, adding the completed S160/S162 ZIPs.

All arm differences are computed against S156, with disjoint textual edits

merged automatically; incompatible overlapping edits stop for explicit review.

Phase 2 must update QA reference expectations for its changed Stage2/3 outputs

and review S164 motion-prefetch compatibility with those new arms.

Existing trees and release ZIPs are never overwritten. ZIP entries have fixed

timestamps, sorted names and fixed compression settings. Builder works from

the supplied release ZIPs, not mutable worker source files.

