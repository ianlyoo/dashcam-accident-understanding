# S174 coverage arm - delivery





`d2af1b1c63625035aa99199b0cebbb4d342eba7425190f75b9f2a6c49e182bdc`. Bytes: 1,222,046,221; 105 members;

official ZIP/unpacked size limits and static validator passed. Exact base

S171 SHA256: `f2f3b55617d80fdee1afabe930d426bc92c99ceb7f48f2953e6504efda280315`. Only S161 predict.py changes among

the original 104 members; one S174 notice is added.



VERIFIED: all 243 labeled and all 750 public Nexar activation audits;

paired hit@0.3 against exact exported S171 rows; actual offline WSL CUDA

Stage2 QA; preserved accepted S161 decisions, protected columns, source frame

IDs, per-clip failure/recovery, native crossing mapping, ZIP member SHA/CRC.

Dependency install 2.39s. Passing QA RSS

3.447 GiB, CUDA reserved

1.465 GiB, free commit minimum

52.68 GiB. S174 released its lease and


resumed after the request cleared.



Measured S161 activation: 52/243 labeled and

208/750 public Nexar. S174 added 24 accepted

fallback crossings and changed 24 entries on labeled clips.

CCD consensus 19 ->

20/79; CCD A3 low

15 ->

16/45; confident

2 -> 2/13. Detailed reasons and limitations are in ANALYSIS.md.



RISKS: cached-detector candidate replay is not a full exported 243-clip S174

run; one cached S161 accepted output disagrees with exported S171. The local

137-long-clip added-time projection is 74.6s,

not an official runtime guarantee. Actor identity and absent road markings

remain unresolved; no S174 hidden-score gain is claimed.




until another official whole-artifact result warrants promotion.



---



# Retained progress history



# S174 coverage arm - progress, 2026-09-28 09:00 KST



RESULT partial. Exact S171 SHA256 is

`f2f3b55617d80fdee1afabe930d426bc92c99ceb7f48f2953e6504efda280315`.

S173 is retained as a fallback release and is not the S174 base. The official


official results; they supersede the earlier weak local transfer concern.



The S161 source uses a COCO vehicle detector, backward association to the

collision time, and a fixed perspective ego-corridor; it accepts only a

bracketed outside-to-inside actor crossing with at least two samples on each

side. The most promising deployment extension is a second pass only on clips

where that acceptance fails. The source can retain every S161 firing exactly

while testing multiple collision-time actor candidates and corridor estimates.

No candidate changes, activation counts, paired gains, runtime or QA are

claimed yet. Public labeled inventory is 243 distinct clips; the 750 public

Nexar source and existing detector/collision caches are being inventoried.

Next: exact S171-anchor activation audit, then fixed fallback implementation,

paired replay, offline CUDA QA and ZIP validation. No private data or upload.



## 09:35 KST checkpoint (before the 13:00 requested progress deadline)



The 750 public Nexar final-fit S160 collision anchors were reconstructed from

the retained per-clip rows and exported localizer; the hit count is 659/750,

matching S160's recorded package parity. These anchors are in-sample, so this

is an activation audit, not a new entry-accuracy claim. The CUDA activation





The exact-S171-anchor labeled replay has reached 238/243: 52 accepted S161

crossings; 39 no anchors, 43 short tracks, 21 not in lane at collision,

71 first-seen/already-inside categories, and 12 observed crossings rejected

by the conservative support gate. These are preliminary counts until all

243 complete. One cached-detector replay gives a crossing at CCD 001118 that

differs from the exact exported S171 entry; candidate scoring will retain

the exact S171 row and disclose this cache-replay limitation.



S174 is now staged on SHA-verified exact S171. The only original member

modified is `model/stage2/s161/predict.py`; 103/104 originals remain identical.

The extension tries up to three collision-time actors, corridor-width/horizon

alternatives and native-frame refinement only after S161 abstains. Its import

is optional: an extension initialization fault leaves original S161 loaded.

Three focused CPU policy tests passed. A five-clip CPU replay exceeded its

90-second smoke timeout after one row due to uncached Faster R-CNN work;

GPU paired replay is queued after the 750-clip audit. No output mismatch was

established by that timeout. Full paired metrics and exported QA are pending.



## 09:55 KST activation audit complete



The exact-S171-anchor labeled audit completed all243 clips: S161 accepted

52/243 (45/217 CCD, 7/26 labeled Nexar). Algorithmic reasons were 65

`crossing` before the strict support gate, 51 `inside_from_start`, 43

`short_track`, 39 `no_anchor`, 22 `not_in_lane_at_collision`, 13

`inside_when_first_seen`, and 10 `far_inside_when_first_seen`. The one

cached-replay/export disagreement at CCD001118 remains disclosed.



The separate public 750 Nexar audit completed with 208 accepted S161

crossings. There were 284 observed crossing reasons before the strict gate;

other reasons were 177 `inside_from_start`, 96 `not_in_lane_at_collision`,

81 `inside_when_first_seen`, 60 `far_inside_when_first_seen`, 47 `short_track`

and 5 `no_anchor`. No audit errors. It used final-fit S160 anchors and MP4

decoding for missing cached detections, so it is an activation measure, not a



request disappeared. At completion sampled RSS was1.702 GiB and CUDA reserved

0.768 GiB. The remaining five labeled clips were then completed on CUDA.




## 10:05 KST paired entry evidence



The full 243-clip CUDA cached-detector replay completed against exact exported

S171 entry and collision rows. S174 added 24 accepted fallback crossings and

changed 24 entry frames; all 52 S161 accepted results stayed unchanged. CCD

consensus hit@0.3 was 19->20/79, CCD A3 low 15->16/45, and the 13 distinct

confident Nexar/S117 labels 2->2/13. The measured added fallback work averaged

0.545 s across 26 native-rate labeled Nexar clips, projecting 74.6 s/137

long clips. This is local evidence, not a hidden-score claim. Exported offline


and candidate phase is running.

