# S174 S161 activation and coverage analysis




Stage2 score .490828857, with 18 more entry hits on 137 hidden clips than S170.

This is the official result for S171, not a measured S174 score.



The public labeled audit used all 243 distinct CCD/Nexar clips and each clip's

exact exported S171 final collision row from S173's paired evaluation. It ran

the exact S171 S161 source and parameters using public cached Faster R-CNN

detections, filling misses with the same detector. It accepted

**52/243** (45/217 CCD,

7/26 labeled Nexar).

One cached replay (`ccd/001118`) accepted a crossing at a different index from

the exported S171 entry. Exact exported S171 rows remain the paired baseline;

cached-detector replay is a diagnostic proxy, and this discrepancy is retained.



The 750 public Nexar positives were audited separately with final-fit S160

collision anchors reconstructed from retained per-clip feature rows and the

exported localizer. The 659/750 in-sample collision hit count exactly matches

S160's package parity receipt. S161 accepted **208/750**. The

26 labeled Nexar clips overlap this 750; counts must not be added together.

The 750 audit used retained FRCNN boxes where available and decoded the public

MP4 for other sampled frames. Final-fit S160 anchors are in-sample and this is

not a fully exported S171 replay or a new entry-accuracy estimate.



| S161 algorithmic reason, labeled 243 | Clips |

|---|---:|

| crossing | 65 |

| inside_from_start | 51 |

| short_track | 43 |

| no_anchor | 39 |

| not_in_lane_at_collision | 22 |

| inside_when_first_seen | 13 |

| far_inside_when_first_seen | 10 |



| S161 algorithmic reason, public Nexar 750 | Clips |

|---|---:|

| crossing | 284 |

| inside_from_start | 177 |

| not_in_lane_at_collision | 96 |

| inside_when_first_seen | 81 |

| far_inside_when_first_seen | 60 |

| short_track | 47 |

| no_anchor | 5 |



The `crossing` reason counts observed box/corridor crossings before the strict

`decide` gate; acceptance also requires a short bracket, at least two outside

and two inside observations, a four-frame track, and entry before collision.

S161 has no explicit `no_lane` reason: it already uses a camera-perspective

ego-corridor instead of painted-line detection. 45 labeled clips

with a recorded horizon estimate had fewer than five supporting vehicle boxes

and used the default horizon. `no_anchor`, `short_track`, and

`not_in_lane_at_collision` can reflect a wrong actor, occlusion, or detector

failure; the public labels do not provide actor identity, so a wrong-actor

count cannot be established honestly. First-seen-inside categories cannot

prove a crossing and remain conservative abstentions in S161.



S174 runs only after S161's decision abstains. It considers up to three

distinct vehicle seeds close to the final S160 collision, uses the same

appearance/IoU association with a wider anchor search, tests fixed plausible

camera-corridor width/horizon alternatives, and inspects native frames around

a supported outside-to-inside transition. It never overrides a S161 accepted

result. The optional source append also leaves the original S161 module loaded

if extension initialization fails. It adds no models or dependencies.



S174 accepted an extra **24/243** fallback crossings and

changed **24/243** entry frames in the paired cached-detector

replay. Added activations by original fallback reason: {'no_anchor': 2, 'inside_from_start': 7, 'short_track': 6, 'crossing': 5, 'not_in_lane_at_collision': 3, 'inside_when_first_seen': 1}.

Actual entry changes by original fallback reason: {'no_anchor': 2, 'inside_from_start': 7, 'short_track': 6, 'crossing': 5, 'not_in_lane_at_collision': 3, 'inside_when_first_seen': 1}.

The replay used exact exported S171 rows as baseline, but S174's candidate

geometry was run directly with cached detector inputs. Exported CUDA QA on

a smaller public panel independently checked protected fields, original frame

IDs, S161 preservation, failure recovery and the shipping archive.



| Label view | n | S171 hit@0.3 | S174 hit@0.3 | Repairs / breaks |

|---|---:|---:|---:|---:|

| ccd_labels_A1_low | 53 | 13 | 14 | 2 / 1 |

| ccd_labels_A2_low | 57 | 17 | 18 | 2 / 1 |

| ccd_labels_A3_low | 45 | 15 | 16 | 2 / 1 |

| ccd_labels_A2_confident | 21 | 3 | 3 | 0 / 0 |

| ccd_consensus | 79 | 19 | 20 | 1 / 0 |

| ccd_labels_A3_confident | 34 | 11 | 12 | 1 / 0 |

| ccd_labels_A1_confident | 27 | 5 | 5 | 0 / 0 |

| nexar_blind_confident | 10 | 1 | 1 | 0 / 0 |

| nexar_blind_low | 15 | 3 | 3 | 0 / 0 |

| s117_interval | 3 | 1 | 1 | 0 / 0 |



CCD consensus is 19 ->

20/79; low-confidence CCD A3 is

15 ->

16/45; the 13 distinct

confident Nexar/S117 clips are 2 -> 2/13. Point-label hit@0.3

uses absolute time error <=0.3s; S117 intervals use their endpoints expanded

by 0.3s. Source views overlap and are not pooled as extra clips. CCD and Nexar

entry labels are reused proxies, not official hidden truth.



Added fallback time on 26 native-rate labeled Nexar clips averaged

0.545s/clip; the local 137-clip projection

is 74.6s (1.24min),

excluding the original S171 workload. This is below the assigned 6-minute

allowance on the measured local mix, not an official server upper bound.

Exported Stage2 paired QA took 168.9s S171 versus

191.1s S174 on its public panel. Offline dependency

installation took 2.39s. The final ZIP is

bound by SHA256 to the staged tree, paired metrics, activation audits and QA.



Limits: fixed corridor geometry can misplace unseen lane markings, a larger

actor search can select a different nearby vehicle, detector boxes approximate

vehicle boundaries rather than tire contact, and cached public detections may

differ from an exported JPEG decode. No hidden S174 gain is claimed.

