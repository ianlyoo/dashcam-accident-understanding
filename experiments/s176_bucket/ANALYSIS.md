# S176 bucket analysis

All measurements use public labels and exact exported S171 entry/collision rows.
S174 candidate entries come from its cached-detector 243-clip replay.
The S161 trace determines the abstention bucket and candidate first-inside
observation. The primary label is selected once per clip using the fixed
source precedence recorded in `measure.json`; source views overlap and
are reported separately below. These reused labels are proxies, not
official hidden-set results.

| S161 reason | Total | S174 abstentions eligible |
|---|---:|---:|
| `inside_from_start` | 51 | 44 |
| `inside_when_first_seen` | 13 | 12 |
| `far_inside_when_first_seen` | 10 | 10 |
| `not_in_lane_at_collision` | 22 | 19 |

## Eligible abstentions: one primary label per clip

The table scores the clips S176 can actually change. Repairs and breaks
are relative to S174. The `first clip frame` experiment outside
`inside_from_start` is shown for completeness but is not considered
definition-faithful and was not eligible for selection. A missing
first-inside trace keeps S174 unchanged.

| Reason | Rule | n | S171 | S174 | Alternative | Repairs | Breaks |
|---|---|---:|---:|---:|---:|---:|---:|
| `inside_from_start` | first clip frame | 44 | 6 | 6 | 28 | 26 | 4 |
| `inside_from_start` | first observed inside | 44 | 6 | 6 | 25 | 23 | 4 |
| `inside_from_start` | first inside minus 0.2s | 44 | 6 | 6 | 25 | 23 | 4 |
| `inside_from_start` | first inside minus 0.5s | 44 | 6 | 6 | 26 | 24 | 4 |
| `inside_from_start` | backward lateral extrapolation | 44 | 6 | 6 | 25 | 23 | 4 |
| `inside_when_first_seen` | first clip frame | 12 | 0 | 0 | 7 | 7 | 0 |
| `inside_when_first_seen` | first observed inside | 12 | 0 | 0 | 0 | 0 | 0 |
| `inside_when_first_seen` | first inside minus 0.2s | 12 | 0 | 0 | 1 | 1 | 0 |
| `inside_when_first_seen` | first inside minus 0.5s | 12 | 0 | 0 | 1 | 1 | 0 |
| `inside_when_first_seen` | backward lateral extrapolation | 12 | 0 | 0 | 1 | 1 | 0 |
| `far_inside_when_first_seen` | first clip frame | 10 | 1 | 1 | 2 | 2 | 1 |
| `far_inside_when_first_seen` | first observed inside | 10 | 1 | 1 | 1 | 1 | 1 |
| `far_inside_when_first_seen` | first inside minus 0.2s | 10 | 1 | 1 | 0 | 0 | 1 |
| `far_inside_when_first_seen` | first inside minus 0.5s | 10 | 1 | 1 | 0 | 0 | 1 |
| `far_inside_when_first_seen` | backward lateral extrapolation | 10 | 1 | 1 | 1 | 1 | 1 |
| `not_in_lane_at_collision` | first clip frame | 19 | 6 | 6 | 2 | 2 | 6 |
| `not_in_lane_at_collision` | first observed inside | 19 | 6 | 6 | 5 | 0 | 1 |
| `not_in_lane_at_collision` | first inside minus 0.2s | 19 | 6 | 6 | 5 | 0 | 1 |
| `not_in_lane_at_collision` | first inside minus 0.5s | 19 | 6 | 6 | 5 | 0 | 1 |
| `not_in_lane_at_collision` | backward lateral extrapolation | 19 | 6 | 6 | 5 | 0 | 1 |

## All clips in each S161 bucket

This second view includes S174-accepted clips too. Applying an
alternative to those clips is hypothetical; the shipping S176
implementation preserves them. The eligible table above is the
selection evidence.

| Reason | Rule | n | S171 | S174 | Hypothetical alternative |
|---|---|---:|---:|---:|---:|
| `inside_from_start` | first clip frame | 51 | 6 | 7 | 32 |
| `inside_from_start` | first observed inside | 51 | 6 | 7 | 29 |
| `inside_from_start` | first inside minus 0.2s | 51 | 6 | 7 | 29 |
| `inside_from_start` | first inside minus 0.5s | 51 | 6 | 7 | 30 |
| `inside_from_start` | backward lateral extrapolation | 51 | 6 | 7 | 29 |
| `inside_when_first_seen` | first clip frame | 13 | 0 | 0 | 8 |
| `inside_when_first_seen` | first observed inside | 13 | 0 | 0 | 0 |
| `inside_when_first_seen` | first inside minus 0.2s | 13 | 0 | 0 | 1 |
| `inside_when_first_seen` | first inside minus 0.5s | 13 | 0 | 0 | 1 |
| `inside_when_first_seen` | backward lateral extrapolation | 13 | 0 | 0 | 1 |
| `far_inside_when_first_seen` | first clip frame | 10 | 1 | 1 | 2 |
| `far_inside_when_first_seen` | first observed inside | 10 | 1 | 1 | 1 |
| `far_inside_when_first_seen` | first inside minus 0.2s | 10 | 1 | 1 | 0 |
| `far_inside_when_first_seen` | first inside minus 0.5s | 10 | 1 | 1 | 0 |
| `far_inside_when_first_seen` | backward lateral extrapolation | 10 | 1 | 1 | 1 |
| `not_in_lane_at_collision` | first clip frame | 22 | 7 | 7 | 2 |
| `not_in_lane_at_collision` | first observed inside | 22 | 7 | 7 | 6 |
| `not_in_lane_at_collision` | first inside minus 0.2s | 22 | 7 | 7 | 6 |
| `not_in_lane_at_collision` | first inside minus 0.5s | 22 | 7 | 7 | 6 |
| `not_in_lane_at_collision` | backward lateral extrapolation | 22 | 7 | 7 | 6 |

S176 selects first clip frame only for `inside_from_start` and
first observed inside minus 0.2s only for `inside_when_first_seen`.
On the latter bucket the sole repair is a low-confidence CCD A3
label; no confident-label gain was measured. First-observed-inside
and extrapolation give no net gain for `far_inside_when_first_seen`;
the same alternatives lose one primary hit for
`not_in_lane_at_collision`. Those buckets retain S174.

The extrapolator fits the first three inside penetration samples
against sampled frame position and projects a positive inward slope
back to zero penetration. If there is no positive trend, it returns
the first observed inside frame. Thus it does not fabricate an
outside observation. Fixed offsets tested were 0.2s and 0.5s.

## Paired labels on all 243 clips

The selected S176 rules change 50/243 entries (39 from
`inside_from_start`, 11 from `inside_when_first_seen`) in the trace
replay. Accepted S161/S174 crossings are preserved.

| Label view | n | S171 | S174 | S176 | Repairs / breaks vs S174 |
|---|---:|---:|---:|---:|---:|
| ccd_labels_A1_low | 53 | 13 | 14 | 14 | 2 / 2 |
| ccd_labels_A2_low | 57 | 17 | 18 | 16 | 0 / 2 |
| ccd_labels_A3_low | 45 | 15 | 16 | 17 | 2 / 1 |
| ccd_labels_A2_confident | 21 | 3 | 3 | 8 | 5 / 0 |
| ccd_consensus | 79 | 19 | 20 | 40 | 20 / 0 |
| ccd_labels_A3_confident | 34 | 11 | 12 | 22 | 10 / 0 |
| ccd_labels_A1_confident | 27 | 5 | 5 | 13 | 8 / 0 |
| nexar_blind_confident | 10 | 1 | 1 | 3 | 2 / 0 |
| nexar_blind_low | 15 | 3 | 3 | 4 | 1 / 0 |
| s117_interval | 3 | 1 | 1 | 1 | 0 / 0 |

CCD consensus: 20→40/79;
CCD A3 low: 16→17/45;
13 distinct confident Nexar/S117 clips: 2→4/13.
Point labels use absolute error ≤0.3s; S117 interval endpoints
are expanded by 0.3s. Source views overlap and cannot be summed
as independent clips.

## Exported package verification

S174 ZIP base SHA256: `d2af1b1c63625035aa99199b0cebbb4d342eba7425190f75b9f2a6c49e182bdc`. The S176 patch
[`S176_on_S174.patch`](S176_on_S174.patch) has SHA256 `f85af16ac2991bd7312c66681bd1f3cfac3df8a753b8084937279ddec4ad4c11`.
The patch line endings were normalized to LF and the S176 notice
was included in the integrator patch after exported QA;
the QA and final build receipts verify that every staged package
member is byte-identical before and after that metadata-only fix.
Only `model/stage2/s161/predict.py` changes among S174 members;
S176 adds one notice and no dependency or weight. On S174 accepted
results it returns the exact previous result; on an S174 fault or
S176 rule fault it returns the S174 result for that clip.

Offline exported WSL CUDA Stage2 QA used 9 public clips:
5 natural entry changes, identical protected fields,
source frame IDs, collision clamp, single/batch parity, injected
fallback fault, same-process recovery, and forced crossing/bucket
frame mapping. No network attempts.
S174 511.9s; S176 521.9s;
dependency installation 2.99s.
Peak RSS 3.197 GiB; peak CUDA
reserved 1.465 GiB; minimum free
commit 52.57 GiB.
The validated ZIP is `$DATA_DIR\releases\S176_bucket.zip` with SHA256
`fc5c3da13c1c0f00b61a73377d55c4d548d12c26031d0837aa042c3f61abda68`. Member SHA/CRC and the submission
validator passed. No S176 hidden score or official server runtime
has been measured. First-frame transfer remains the main risk:
S145 applied it indiscriminately and lost 24 hidden entry hits;
S176 confines it to S161 `inside_from_start` after S174 abstains.
The one known S161 cached-replay/export disagreement in S174
remains a limitation of the paired proxy.
