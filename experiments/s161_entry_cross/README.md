# S161: collision-anchored entry crossing, Arm C

Status: **release ready for manual review**. WSL CUDA QA, export parity and ZIP validation passed.
The first QA was deliberately interrupted to correct evidence-output placement and

Base: exact `$DATA_DIR/releases/S156_casc.zip`, SHA256
`e7fb70a66ad251d47c180861e2e001da7b270259c2e12ca394c6de3f1c510d70`.
Work: `$DATA_DIR/s161_entry_cross/`.

Queue handoff recorded 2026-09-27 13:21 KST: launcher PID 42004, Python supervisor
Do not start a duplicate while this supervisor is alive. Inspect
`logs/qa_v2_supervisor.{out,err}.log`, `logs/qa_v2.log`, and `logs/integration.log`.
The queued chain automatically runs `finish.py` after passing integration; it then
creates the ZIP, validates it, writes `release.json`, and updates this README.
These PIDs are a handoff snapshot, not a claim that the process is still running.

## Change

Run the S132 participant tracker around the **final S156 collision**, selecting
the most impact-like nearby vehicle and tracking it backward. Accept only an
observed outside-to-inside crossing: at least two inside and two outside samples,
at least four track observations, a bracket at most 0.5 s wide, and entry no later
than collision. Everything else retains S109 entry with S156's existing clamp.
Already-inside/first-seen tracks never trigger frame-zero replacement. Side and
evasion remain S156. Detector and all other model weights are unchanged.

Lane contact is approximated by the inset lower box edge crossing a flat-road lane
trapezoid; this is not actual wheel keypoint or lane-marking detection. The horizon
is estimated per clip. Sampling uses an absolute grid, batch two, 16 sparse samples,
and at most 64 detector frames per file. There is no cross-file state or cumulative
time fallback. The legacy `budget_seconds: 240` field is schema compatibility only.

## Evidence

Entry hit means distance <=0.3 s from a point label, or from an S117 interval edge.
All annotations are reused AI proxies, not official/human truth or fresh holdouts.
No S161 fitting or parameter sweep was performed. Nexar collision anchors are the
S156 K=2/delta=.2 **OOF** predictions, with the baseline entry clamped identically;
short CCD clips retain the unchanged S109 collision. Parameters were fixed before
this evaluation, but the source method and annotations have been used previously.
Of 26 Nexar baselines, 24 come from existing exported S109 rows; 00586 and 00917
use the existing identical-motion-function replay. Cached base and tracker JPEG
encodings differ (older Q88/Q92 preparations; newly needed tracker folders Q95).
This is an OOF proxy estimate, not an exact full-fit export replay of all 26 clips.
The separate three-clip integration compares exact S156/S161 exports on identical
JPEGs; it tests deployment behavior, not out-of-fold accuracy.

| Source | n | S156 hits | S161 hits | Repairs / breaks |
|---|---:|---:|---:|---:|
| CCD consensus | 79 | 16 | 19 | 5 / 2 |
| Nexar blind, medium/high | 10 | 1 | 1 | 0 / 0 |
| Nexar blind, low | 15 | 4 | 4 | 1 / 1 |
| S117 intervals | 3 | 0 | 0 | 0 / 0 |
| CCD A1, medium/high | 27 | 4 | 5 | 2 / 1 |
| CCD A1, low | 53 | 14 | 13 | 1 / 2 |
| CCD A2, medium/high | 21 | 3 | 3 | 1 / 1 |
| CCD A2, low | 57 | 19 | 17 | 3 / 5 |
| CCD A3, medium/high | 34 | 9 | 11 | 2 / 0 |
| CCD A3, low | 45 | 13 | 15 | 4 / 2 |

Coverage: **217 CCD + 26 Nexar = 243 distinct clips**, all locally available numeric
entry annotations. Source views overlap and are not pooled. Official public Stage2
labels contain `t_entry=-1` on all five clips, so they provide execution checks only.
S117's other three clips have unknown entry and are not scored. Older Nexar contact
reviews annotate collision/contact, not numeric lane-entry timing. The S114 lane
training data explicitly contain no accident-entry labels.

CCD consensus gives +5.20 hits per 137-equivalent, or +0.00532 weighted total-score
proxy. Its paired bootstrap 95% delta-rate interval is [-0.0253, +0.1013]. The combined
13 distinct confident Nexar/S117 entries remain 1/13. **No measured hidden gain** is
claimed. Transfer is plausible because actor crossing addresses the official entry
definition and first-frame behavior is removed, but the long-clip evidence does
not support a positive effect yet. Treat this as an exploratory, weak-evidence arm.

S117 failures expose anchor/geometry limits: 00900's OOF collision is 17.016 s, earlier
than entry [19.3,19.7], so the clamp prevents a repair. 00635 anchors late (26.2 s)
and the participant is not in lane; 00647 has an already-inside track and abstains.

## Runtime and validation

September 27 mode repair: `qa_v3` is the final passing run. The original
mode failure is retained as `logs/qa_v2_mode_conflict.log`; the first repair
rerun stopped because `qa_v2` already existed, then the fresh-directory rerun
passed. Its three-stage harness recorded 3,395,182,592 bytes (3.162 GiB)
lifetime host RSS peak, above this worker's assigned 3 GiB cap. This is a
resource-budget breach, not a compliant peak. The separate long integration
ran subsequently; future entry work uses Stage2-only QA and reuses unchanged
Stage1/Stage3 identity evidence. No hidden-score gain is claimed.

On 26 native-rate Nexar clips, uncached GPU tracker cost averaged **1.6406 s**
(maximum 3.3649 s): **224.8 s / 137 clips**, excluding detector load. This is a
conditional workload estimate, not an official workload count or runtime guarantee.
CCD's extra 138 low-confidence clips reuse public detector caches, filling missing
detections on CPU; these replay timings are not used for the runtime estimate.
Peak measured allocator usage: 538,978,816 bytes. CPU cache replay peak commit:
2,899,755,008 bytes, within 3 GiB. Device/shared-load differences remain a risk.

Policy checks pass for observed crossings, all first-seen abstentions, short/uncertain
tracks, protected outputs and input-order independence. The staged diff from S156
contains only two modified members and three additions, listed below. Full WSL CUDA QA, three long-clip exported parity checks and the ZIP validator passed.
See `release.json` for hashes, runtime and CSV identities.
AST comparison confirms only `_predict_track` and `validate_rule` differ;
the other 30 functions are identical (`source_diff.json`). The September 27
repair permits an entry-only tracker with `preserve_s109_fields` while rejecting
`side:true`; the final S156 collision still supplies the tracker anchor.

QA uses the real shipped inference and weights with a test-only DataLoader shim
(`num_workers=0`) to respect the worker's process limit. This shim is not packaged.
CSV comparisons and runtime evidence are under `qa_v3/ipc/` and `integration.json`.

## Stack / reproduce

Stack by copying `model/stage2/s161/{predict.py,params.json}` and the notice, adding
the `entry_frame` track rule with `side:false`, and applying the small
`_predict_track` changes in the local `adapter.py`: replace the cumulative budget
check with per-file execution and retain `entry=min(entry,collision)`. This composes
after the existing collision branch, so another collision arm can supply the anchor.
Do not overwrite another arm's adapter wholesale. No shared `tools/` or `src/` edits.

Modified: `model/stage2/s118/{adapter.py,rule.json}`.
Added: `model/stage2/s161/{predict.py,params.json}`, `model/licenses/S161-NOTICE.txt`.
Weights are reused from `model/stage2/s144/detector.pth`.

From repository root, Windows Python is `.venv/Scripts/python.exe -B`:

1. `candidates/s161_entry_cross/check_policy.py`
2. `candidates/s161_entry_cross/run_eval.py` (checks commit and takes a 2 GiB lease;
   append-only/resumable). For cache replay: `evaluate.py --device cpu --use-cache`.
3. `candidates/s161_entry_cross/summarize.py` writes the versionable complete
   `metrics.json` and `paired_predictions.jsonl`, including source hashes.
4. `candidates/s161_entry_cross/stage.py` verifies S156 and stages the exact arm.
5. `candidates/s161_entry_cross/run_qa.py` waits for shared QA/GPU availability,
   runs the local WSL QA harness, then `integration.py` and gated `finish.py`;
   logs stay under the work dir. Existing QA directories are preserved, so reuse
   their bound passing evidence rather than launching this command twice.
6. The automatic packaging step uses the unchanged S118 builder with
   `--out $DATA_DIR/s161_entry_cross/stage`,
   `--qa $DATA_DIR/s161_entry_cross/qa_v2/qa.json`,
   `--zip-name S161_entry.zip`, then run `tools/validate_submission.py`.

All outputs are public-data-only. No private evaluation files, training, uploads,
source deletions or modifications to other workers' paths were performed.

## Release identity

ZIP: `$DATA_DIR/releases/S161_entry.zip`

SHA256: `7bcabece62a8c82907f997a431cc1673ba36c81270b92133fdfcd0eb738f9e44`

Size: 1308005749 bytes. Validator exit 0.

Exact non-entry CSV parity passed on the public three-stage QA set and
three native-rate Nexar clips; 2 long-clip entries changed.
Long-clip tracker time: 4.312 s / 3 clips; detector load: 0.949 s.
Matched whole-call wall delta: -67.721 s (includes uncontrolled cache/load effects).
The integration script packages only after every prerequisite passes; finish.py
is safe to rerun and verifies any existing ZIP instead of overwriting it.
