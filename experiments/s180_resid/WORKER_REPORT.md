# S180 residual entry timing

RESULT done. Release `$DATA_DIR\releases\S180_resid.zip`, SHA256 `d5033adfce1d40fb5fc98289196fb791b274ad4529d111c3d2fbacaa9623a84b`.

## Evidence and decision

The base is exact S178 (`S178_side.zip`, SHA256
`bab785c758b82f7524bdaa1153220c3aa61a7d8b120ae5cac14c049ec83ec047`).
The only inference change is +0.2 s for accepted S174-only geometric
crossings, clamped to collision. All other entry sources and every
collision, side, evasion, and Stage3 output remain S178.

The study covers 211 lawful annotations: CCD consensus 79, CCD A3-low 45,
confident Nexar 13, and DKB human 74. They are annotation counts rather
than disjoint clips; six IDs overlap DKB and CCD. Signed error is predicted
time minus the annotated interval midpoint in seconds. Hit@0.3 means
the prediction lies within the full interval extended by 0.3 s.

| Decision source | n | S178 hits | Mean signed s | Median signed s | Early / late misses |
|---|---:|---:|---:|---:|---:|
| S161 crossing | 45 | 26 | +0.210 | +0.000 | 12 / 7 |
| S174 crossing | 25 | 8 | +0.834 | -0.100 | 9 / 8 |
| S176 first frame | 40 | 25 | -1.480 | +0.000 | 15 / 0 |
| S176 first observation minus 0.2 s | 10 | 2 | +8.850 | +3.000 | 0 / 8 |
| S109 fallback | 91 | 23 | +0.584 | +0.100 | 32 / 36 |

Means are sensitive to long Nexar outliers. The S174 error is
heterogeneous rather than a global offset: its near-boundary early
cases benefit from a small delay, while distant late misses remain.

| Panel | S161 | S174 | S176 first | S176 obs-0.2 | S109 | Total S178 -> S180 |
|---|---:|---:|---:|---:|---:|---:|
| CCD consensus 79 | 8/14 | 1/6 | 22/24 | 0/4 | 9/31 | 40 -> 40 |
| CCD A3-low 45 | 8/13 | 2/7 | 1/5 | 1/1 | 5/19 | 17 -> 20 |
| Nexar confident 13 | 1/2 | 0/1 | 2/3 | 0/2 | 1/5 | 4 -> 4 |
| DKB human 74 | 9/16 | 5/11 | 0/8 | 1/3 | 8/36 | 23 -> 25 |

S174 +0.2 s repairs five labels (CCD `001115`, `001273`,
`001427`; DKB `000021`, `000091`) and breaks none. +0.1 s repairs
three; +0.3 s repairs six, both without paired losses. +0.2 s is
the smaller gain on two panels and the shift selected when DKB
is held out. A tracked vehicle rectangle can touch the corridor
edge a few frames before its wheel/tire reaches it, so a two-frame
delay at 10 fps is physically plausible. This remains a proxy.

## Leave-one-panel-out and rejected shifts

Each fit excludes every training label whose clip ID occurs in
the held-out panel. S174 selects +0.3 s holding out CCD consensus,
CCD A3-low, or Nexar, and +0.2 s holding out DKB. Held-out S174
hits are 1->1, 2->5, 0->0, 5->7 respectively. The shipped fixed
+0.2 s has the same held-out hits on all four panels; no fold loses.
This is limited evidence: only one confident Nexar annotation is
an S174 crossing, and the CCD and Nexar labels are AI-made.

S161 has median error zero; nonzero shifts tie with both repairs
and breaks or lose pooled hits. S176 first-frame +0.1 s gains one
pooled label but no held-out hit and delays a known already-inside
frame-zero case. S176 first-observation-minus-0.2 s gains nothing
within +/-0.3 s; eight misses are late, often by seconds. S109
fallback has 68/91 misses with balanced early/late errors; small
shifts have zero or negative pooled gain. These remain S178.

The largest remaining source is S109 fallback (68 misses). Its
largest identified reason is `s174_no_supported_crossing` (34
misses / 48 rows); 15 of those misses have original S161 reason
`short_track`. A concrete next fix is collision-anchored wheel
tracking through dense nearby frames to recover a measured
outside-to-inside transition when coarse tracking lacks two
observations on each side; abstain without a measured crossing.

## Verification and limits

Exported offline CUDA QA passed on 9 public CCD clips:
5 entry changes, no protected-field changes,
correct source-frame IDs and collision clamp, per-clip failure
fallback and same-process recovery. Dependency install took
3.48 s; peak RSS was
2.60 GiB, CUDA reserved
0.95 GiB, and minimum free commit
49.76 GiB. No network attempt.
The nine QA clips include five predicted S174 repairs and controls
for the unchanged S161, S176, and S109 paths. S178 exported QA
provided 84 additional clip outputs for analysis; three missing
DKB clips were exported on exact S178. Eight overlaps with cached
tracker replay agree in entry and source. Most of the 243-clip
labeled replay is cached rather than newly exported. No private
evaluation data and no official/hidden gain are claimed.

Full rows, per-panel scans and overlap exclusions are in
`rows.jsonl` and `analysis.json`; fallback reasons are in
`fallback_buckets.json`. The clean patch is `S180_on_S178.patch`.

ZIP validator passed (0); member CRC/SHA256
verification passed for 108 members. Release SHA256:
`d5033adfce1d40fb5fc98289196fb791b274ad4529d111c3d2fbacaa9623a84b`. The ZIP is a fallback candidate; the
integrator must validate its own stacked whole artifact.

RISKS: Local labels are partly AI-made and the S174 sample is
small. A +0.2 s shift has no demonstrated hidden-set improvement.

NEXT: Integrator may apply the clean S178-based patch to the stacked
