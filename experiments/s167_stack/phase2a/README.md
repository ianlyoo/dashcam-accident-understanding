# S169 stack: phase 2a



Ready 2026-09-27T21:13:03.220069+09:00. ZIP `$DATA_DIR/releases/S169_stk.zip`,

1,217,088,326 bytes, SHA256 `94b963cd8949102ee1a0e53da9b9c57c74252219b863a5e15fbe3f6f937ba3ef`.



Exact S156 + S163 Stage1 cue + S164 runtime + S162 conservative acceleration

decoder. S160 collision is not included. Compared with S167, exactly one of

97 members changes: `model/stage3/s141/predict.py`. It is byte-identical to the pinned S162 release;

the other 96 members, including inference.py, S164 runtime, weights and

requirements, are unchanged. `member_manifest.json` records every arm hash

and base-relative change. The original multi-arm builder is reused unchanged.



S164 wraps final Stage3 dispatch and prefetches only original CPU motion arrays.

S141/S162 still runs in the parent, inside its original lock, and temporarily

replaces S109's acceleration combiner. S162 does not modify motion extraction

or the frame observer. Normal CUDA QA prefetched all 5 clips and replayed all

2,998 frames through the original visual observer; no legacy fallback occurred.

S162's steering function ASTs match S156. All scoped hooks are restored.



Actual offline WSL/CUDA checks passed:



- Stage1: all 10 public and 3 diagnostic cases equal S163 CSV bytes.

- Stage2: all 5 public and 7 existing Nexar clips equal S156 CSV bytes.

- Stage3: all 2,998 rows equal S162, with exactly 68 acceleration changes

  versus S156; steering, IDs, sample indices and STOPPED remain identical.

- Fault injection: two clips enter S164 prefetch and replay (1,200 frames).

  OPEN_004's decoder mutates its logits and raises once; all its columns match

  S156, observably restoring 58 acceleration rows. OPEN_002 retains its normal

  S162 output. No visual-model fallback occurs. A subsequent two-clip call

  returns normal S162 outputs, and decoder/combiner/extractor hooks restore.



The fault targets a worker-prefetched clip with real S162 label differences,

so it exercises the combined cached-feature path, not a single-file serial

shortcut. Fault and recovered CSVs, diagnostics and hashes are retained under

`$DATA_DIR/s167_stack/phase2a/qa`.



| Same panel | Rows | S156 seconds | S169 seconds |

|---|---:|---:|---:|

| stage1_public | 10 | 37.20 | 31.06 |

| stage2_public | 5 | 45.66 | 30.62 |

| stage2_long | 3 | 282.55 | 135.29 |

| stage2_cascade | 4 | 375.33 | 168.78 |

| stage3_public | 2998 | 102.32 | 81.77 |



Same-panel total 843.06s -> 447.54s

(46.9% reduction). Conditional scaling of the

inherited 49m40s server reference gives **26m22s**.

These are historical baseline comparisons under changing shared load, not a

controlled speedup or measured server result. Stage3 uses S162's existing

10Hz public panel (2,998 rows); S167/S164's earlier 184.03 -> 126.48 seconds used

5,992 native rows and is not directly comparable. S162 alone previously took

53.26s on the same 10Hz panel. Framework imports,

three Stage1 diagnostic clips, fault injection and recovery are excluded from

the matched total. S164's more conservative inherited estimate remains about

43m30s plus small, workload-dependent cue/decoder overhead, also unverified.



Peak aggregate RSS 2.766 GiB,

maximum 2 processes; CUDA reserved

1.432 GiB. QA used a 1.75 GiB Torch


DataLoader loading and S164_WORKERS=2. RSS sampled every .25s; CUDA reserved

excludes driver/context. No network attempts or cap violations were observed.

All ZIP member hashes/CRCs and tools/validate_submission.py passed.

The initial idle waiter exited when another worker removed its lease during

enumeration, before CUDA began. Lease scanning was corrected to tolerate that

deletion and to defer admission on extant unreadable leases. The original

launcher.err is preserved; the replacement waiter reuses completed evidence.



S162 remains exploratory: its reused proxy evidence does not establish hidden

accuracy, and its sparse public acceleration labels regressed 40/50 -> 39/50.

S163's known false-positive risk remains. No new accuracy improvement, private

evaluation access, upload, commit or official score is claimed.



Reproduction from repo root, Windows Python `.venv/Scripts/python.exe -B`:



```text

python -B candidates/s167_stack/prepare_phase2a.py

powershell -NoProfile -File candidates/s167_stack/launch_phase2a.ps1

powershell -NoProfile -File candidates/s167_stack/finish_phase2a.ps1

```



Preparation calls the unchanged build_stack.py with arms S163_s1.zip,

S162_acc.zip and S164_fast.zip; output is s167_stack/phase2a/candidate.

Existing trees and ZIPs are preserved. Run long jobs detached with logs.

Phase 2a stops here; phase 2b requires the later explicit S160 prompt.

