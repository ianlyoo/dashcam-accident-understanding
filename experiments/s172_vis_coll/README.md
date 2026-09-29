# S172 visual collision arm

Exact carrier: `$DATA_DIR/releases/S171_ent.zip`, SHA256
`f2f3b55617d80fdee1afabe930d426bc92c99ceb7f48f2953e6504efda280315`.
Validated revision3 release: `$DATA_DIR/releases/S172_vis.zip`,
SHA256 `2969891d7472a8fb8bcc6195fee97bf4f55185163149bf71869b5c46fdced549`.
S172 has no official score or upload in this worker task.

The new decision runs after the entire S171 Stage2 prediction, including S161's
entry reconstruction anchored to S160's original collision. It changes only
`collision_frame` and `entry_frame=min(S171 entry, new collision)`. Entry side,
evasion, and all other fields pass through. A missing prior, load/decode/model
failure, unavailable CUDA, or invalid prediction preserves the exact S171 row.
There is no cross-file adaptation or cumulative inference-time cutoff.

Frozen Apache-2.0 DINOv2-Large uses the already pinned local checkpoint
`399fba97a95f22c36834418bc69373364a99af3a1153da1c0fb31db567c92e23`.
Revision2 stores that checkpoint in FP16, already the inference dtype. The
converted SHA256 is `f0dceb8e216082a9f6e0dc4b503831f01031b88bbd32bdf9a5cae241f785faf2`.
All 439 tensors exactly equal the source cast to FP16; descriptors from public
clip 00000 frames 0 and 600 match bit for bit. `precision_conversion.json`
records this check and the conversion notice is included in the model license
notice and safetensors metadata. This reduces checkpoint storage from 1.22 GB
to 609 MB without retraining or changing inference parameters.
At 4 Hz, 224-square full-frame images produce 2048 features: CLS and mean patch
tokens. A 64-unit temporal head uses three kernel-3 convolutions with dilations
1/2/4, a 15-sample (3.5-second) receptive field, and no position/length inputs.
The two descriptor halves are independently normalized. Training uses only the
same 750 public Nexar positive clips and original five 150-clip S160 folds.

| Head loss | Epochs | Best selected OOF hit@0.3 | Fusion | Repairs / regressions |
|---|---:|---:|---|---:|
| Gaussian cross entropy | 15 | 481/750 | product .5, half-width 9 | 41 / 18 |
| Gaussian cross entropy | 30 | 484/750 | mixture .5, half-width 6 | 52 / 26 |
| Gaussian plus hit-window mass | 15 | 484/750 | product .5, half-width 9 | 41 / 15 |
| Gaussian plus hit-window mass | 30 | 485/750 | product .5, half-width 9 | 47 / 20 |

Every head averages seeds 0/1. Each checkpoint compares 33 decoder settings;
there are 132 coarse comparisons. The selected 30-epoch mass-loss head has
fold hits 102/93/89/98/103 versus S160's 99/89/83/95/92. Its best visual-only
decoder scores 341/750. Fusion is therefore essential: the selected posterior
is proportional to S160 probability times the square root of visual probability,
decoded over S160-supported frames with +/-9-frame probability mass.

These are selected, reused OOF results, not independent confirmation. S160's
458/750 reference itself uses a previously selected proxy ensemble with only
three MLP-seed OOF predictions, whereas deployed S160 has five MLP seeds. The
new arm uses the deployed S160 posterior at inference; this inherited proxy
mismatch and all prior S160 selection bias remain. No hidden gain is measured.

10 Hz refinement was evaluated around two label-blind leading fused peaks,
separated by 1.5 seconds, within +/-0.5 seconds. Actual additional DINO frames
were extracted. Each dense target preserved the trained 4 Hz context spacing.
The fixed decoder fell to 481/750. A further 33-setting decoder grid, holding
the sampling schedule fixed, reached only 483/750 (60 repairs, 35 regressions).
Its replay matched all 750 original fine predictions exactly. Refinement is
therefore disabled in the release: this explicitly revises the proposed recipe
in favor of the best measured OOF result. There are 165 distinct decoder
comparisons total; none is fresh confirmation.

`oof_audit.json` verifies all 40 saved fold/seed/epoch checkpoints: exact original
fold membership, disjoint train/held-out IDs, matching source hashes, finite
predictions, and bit-exact reconstruction of each saved two-seed OOF ensemble.
The selected mass-loss, 30-epoch CUDA heads have also been refitted on all 750
public clips with the same architecture, loss, seeds and training implementation.

`feature_manifest.json` binds all 750 atomic caches (115,472 sampled frames).
Windows MP4 extraction averaged 2.876 seconds per clip (p95 3.359, max 4.838);
RSS peak 1.53 GiB, minimum free commit 43.18 GiB, peak CUDA reservation
0.633 GiB. These measurements exclude the trained head/refinement and are not
exported folder-runtime measurements. The task's added-runtime allowance is
480 seconds for the user-specified scenario of 137 long clips.

Reproduction components: `inspect_inputs.py`, `prepare_baseline.py`,
`supervise.py`/`extract.py`, `train.py` (CPU) or `gpu_train.py` (leased CUDA),
`refine_oof.py`, `build.py`, `qa_supervise.py`/`qa.py`, and `release.py`.
Argument files retain the selected CUDA OOF, refinement and full-fit commands.
All large caches, training checkpoints, logs and staging trees live under
`$DATA_DIR/s172_vis_coll`; selected small export weights will also
be retained here. `run_bounded.py` runs noninteractive PowerShell inspections
with explicit timeouts. Job supervisors are detached with bounded children.

`cpu_checks.json` records S160 decoder parity on all 750 clips, temporal spacing
parity, field/clamp isolation, three fallback modes and CPU RNG preservation.
2 GiB lease. Public and long base rows matched exact S171; protected fields,
injected load/runtime/invalid faults, same-process recovery and all 2,998
subsequent Stage3 rows passed. Peak aggregate RSS was 3.28 GiB, peak CUDA
reservation 1.43 GiB, minimum host free commit 47.44 GiB. Stage1 implementation,
weights and its existing passing evidence are reused by exact identity.
The revision2 pending ZIP passed the validator and every streamed CRC/member
hash check. Release promotion still requires acceptable runtime evidence.

The initial artifact passed public QA (five clips, no visual prior available),
but long QA stopped at 4,359,606,272 aggregate RSS bytes, above the 4 GiB cap.
Its pending ZIP, logs and failure record remain in the original work directory.
Revision2 addresses checkpoint-loading overhead with the exact FP16 storage
conversion above. Its fresh QA uses S171's allocator trimming pattern before
model loads and retains the original caps. Set `S172_REVISION=revision2` when
running `qa_supervise.py`, `pack_pending.py`, or `release.py`; the revised tree,
QA and pending archive are isolated in `$DATA_DIR/s172_vis_coll/revision2`.

The integration QA's three visual timings were 298.61, 5.89 and 4.03 seconds.
The first coincided with a 14 GiB DeepVoice smoke lease and a 15,909 MiB / 100%
host GPU snapshot. A separate timing-only WSL benchmark is queued using the
Its cached OOF priors are for runtime only, not accuracy validation. Neither
this pending measurement nor the shared-load result establishes L40S timing.
See WORKER_REPORT.md for the live supervisor and handoff limitation.

## Revision3 runtime arm (2026-09-28)

S171_ent's official Stage2 score is 0.490828857 with a 33m08s official
scored 485/750 on reused five-fold public OOF. Exact 2 Hz cache subsetting
scored 474/750; exact 2.5 Hz subsetting scored 482/750, with fold hits
103/93/88/98/100. The chosen 2.5 Hz pattern retains sample indices
0,2,3,5,6 modulo 8 from the original 4 Hz grid. It keeps the same frozen
weights, two trained heads, product fusion weight 0.5 and nine-frame decoder.
These rate comparisons reuse the selected OOF data; they are not fresh evidence
of hidden-set improvement. The original DINO features were FP16, not FP32.

Revision3 batches up to 16 frames with FP16 autocast. It checks both a
600-second visual-stage timer and 45-minute process timer before each clip,
during the visual frame batches, and before accepting a prediction. On breach,
it retains S171 collision and entry for that clip and all remaining clips,
logging visual coverage and the number of S171 fallback rows. A mocked-clock
unit test passed both timers and protected-field isolation. Actual exported
WSL QA passed public and three-long-clip panels, three injected failures,
same-process recovery, an injected process-time guard and all 2,998 Stage3
rows. Long-clip visual times were 2.384/2.972/2.761 seconds; peak aggregate
RSS was 3.30 GiB and peak CUDA reservation 1.43 GiB.

Fresh timing-only WSL benchmark on those three public long clips with cached
OOF S160 priors measured 4.175/3.134/2.626 seconds, a 3.109-second fixed
overhead, and 456.828 seconds (7.614 minutes) projected for 137 clips.
The shared RTX 4070 Ti SUPER is not an L40S server measurement. The fixed
150-clip original fold-0 check found that batch-16 FP16 autocast changed
descriptors numerically (mean absolute difference 0.00166 from the earlier
FP16 batch-4 cache), while all 150 decoded picks and 103/150 hits matched the
cached 2.5 Hz arm. A separate actual FP32 DINO run on those 150 public clips
also scored 103/150, with no changed picks versus exported FP16; FP32 is an
extra control, not the original production dtype. This checks the changed
feature path on 150 clips, not all 750. The 482/750 five-fold score is the
cache-based OOF estimate. The release ZIP passed the validator, every member
hash and the gated release receipt in `release.json`; no official S172 gain
has been measured.

No private evaluation data, upload, commit, push, correspondence or purchase.
