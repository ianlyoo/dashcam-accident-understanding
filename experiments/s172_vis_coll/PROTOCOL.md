# S172 visual collision arm

Base: exact S171_ent.zip, SHA256 f2f3b55617d80fdee1afabe930d426bc92c99ceb7f48f2953e6504efda280315.

Public-only population: exact 750 S160 Nexar IDs and existing five folds of
150 each (inputs.json). Frozen local Apache-2.0 DINOv2-Large revision
47b73eefe95e8d44ec3623f8890bd894b6ea2d6c; FP16 224-square full-frame inputs;
4 Hz global CLS and mean-patch features. Features checkpoint atomically per clip.
Temporal head uses event labels on training folds only. No absolute clip-position
features. Refine leading predicted peaks at 10 Hz, then evaluate fusion or
reranking with S160 on identical folds. Report every tried setting and selection
bias; S160's 458/750 is a selected incomplete-final-ensemble OOF proxy.

Only final collision and min(S171 entry, new collision) may change. New visual
decision must run after S171 entry tracking, preserving its original anchor.
Every load/decode/head/nonfinite failure returns the exact S171 row for that clip.
No cross-file updating, no private evaluation access.

>=12 GiB. Per-process CUDA allocator capped at 1.65 GiB to leave context overhead.
member identities, offline loading and the existing submission validator.

User decision gate: by ~15:00 KST Sep 28, stop if measured OOF improvement is
<=10/750 over the 458 baseline. Do not package without that evidence. Release
deadline 19:00 KST. Added runtime allowance 480 seconds / 137 long clips =
3.50365 seconds per clip; measure local cold and warm timings and state server

## Pre-result head comparison

Two 64-unit, three-layer temporal convolution fits use seeds 0/1 and checkpoints
at epochs 15/30. `v1` uses Gaussian-target cross entropy on CPU; `v2mass` adds
negative log probability mass within the +/-0.3-second hit window and runs on
CUDA under the same S172 2 GiB lease. Each checkpoint compares 33 fixed decoders
(visual alone, posterior product, or mixture; three smoothing widths), for 132
coarse decoder comparisons total. This is exploratory selection on reused OOF
data. The CUDA wrapper checks aggregate RSS with the CPU job below 4 GiB.

## Measured refinement revision

The selected coarse model scores 485/750. Actual 10 Hz refinement with the
same decoder scores 481/750. A subsequent 33-setting decoder grid on the
cached fine predictions (keeping the original label-blind coarse schedule
fixed) reaches only 483/750. Its fixed-decoder replay matches all 750 previous
fine picks exactly. Total distinct decoder comparisons: 132 coarse +33 fine.
The release therefore disables refinement and retains the 485/750 coarse
posterior product. This is an explicit method revision, not an unchanged
original recipe or a fresh validation result. Both failed refinements remain
recorded; the full 30-epoch, two-seed CUDA refit has completed.

## Memory correction after first exported QA

Initial public WSL QA passed. Initial long WSL QA stopped at an observed
4,359,606,272 aggregate RSS bytes (4.06 GiB), exceeding the unchanged 4 GiB
cap during DINO loading. That failed run and the original pending archive
remain under the original work directory. The next candidate is staged in
`revision2`, with DINO checkpoint storage changed from FP32 to the FP16 dtype
already used by extraction and exported CUDA inference. `half_revision.py`
requires every tensor to equal the original tensor cast to its target dtype
and requires bit-identical descriptors on specified public frames before
producing a new build record. No model refitting or OOF reselection occurs.
The revised QA uses the original S171 harness's allocator trimming pattern
before model loads, logs phase/RSS milestones, and retains all resource caps.
Use `S172_REVISION=revision2` for QA, pending packaging, and final release.

## Runtime measurement under sharing

Revision2 passed both exported WSL panels, injected faults, same-process
recovery, and exact Stage3 continuation. Long QA recorded 298.61, 5.89 and
4.03 seconds for its three visual clip updates. During that run, a host
snapshot showed 15,909 MiB occupied and 100% GPU utilization while DeepVoice
held a 14 GiB smoke lease. That timing is preserved, not discarded.
The remaining runtime question is tested with one separate WSL run of the
unchanged exported visual arm on the same three public frame folders, under
Cached public S160 OOF priors are used for this timing-only run; it makes no
accuracy claim. All visual computations, frame I/O, model load and cleanup
remain real. The projection includes measured fixed overhead and the actual
integration run's S160-prior capture overhead. Neither run measures L40S.

exception, pause at the next checkpoint and poll every 30 seconds. Updated
benchmark source checks before CUDA work and before each clip, exits 75 to
release its lease, and resumes through the updated supervisor. The already
waiting supervisor uses the earlier 15-second polling and does not resume
exit 75 automatically. Two guarded replacement attempts made no changes:
first detected a console helper, then the request disappeared. The worker


Official S171 scored Stage2 0.490828857 in 33m08s; it remains the champion
until an official S172 result exists. Exact four-Hz cache-subset comparisons on
the original five OOF folds gave 474/750 at 2 Hz and 482/750 at 2.5 Hz,
versus 485/750 at 4 Hz. The 2.5 Hz arm retains sample indices 0,2,3,5,6
modulo eight, runs the same frozen DINOv2-L in batches of 16 under explicit
FP16 autocast, and keeps the same two temporal heads and decoder.

`runtime_fast.py` records a 600-second cumulative visual-stage timer and
2700-second process timer, checked before clips, across visual batches and
before accepting each pick. It returns exact S171 collision/entry for the
current and all remaining files after a breach, logging coverage and count.
`test_guard.py` passed with a mocked clock; actual exported WSL QA passed
public/long panels, protected fields, three injected faults, same-process
recovery, an injected process-clock breach and 2,998 Stage3 rows. Peak RSS
3.30 GiB, CUDA reservation 1.43 GiB. Fresh WSL timing gave 456.828 seconds
(7.614 local minutes) projected over 137 clips, including model load/cleanup.
This is an RTX 4070 Ti SUPER shared-GPU measurement, not L40S. The 150-clip
actual batch-16/autocast FP16 check retained all 150 picks and 103/150 hits;
descriptors differ bitwise from the original FP16 batch-4 cache. A separate
FP32 control on the same 150 clips scored 103/150, with no changed picks
versus exported FP16 batch-16. Cached FP16, exported FP16 batch-16 and FP32
all scored 103/150 there; descriptor mean absolute differences were 0.00166
between the two FP16 paths and 0.00201 between FP32 and exported FP16.
The 482/750 score remains a cache-based five-fold estimate, because the
exported descriptors are not bit-identical. Selection and subset scoring reuse
public OOF and are not fresh confirmation. Release SHA256
2969891d7472a8fb8bcc6195fee97bf4f55185163149bf71869b5c46fdced549;
