# S178 geometric side arm — delivery



RESULT done. Release `$DATA_DIR/releases/S178_side.zip`, SHA256

`bab785c758b82f7524bdaa1153220c3aa61a7d8b120ae5cac14c049ec83ec047`, 1,222,049,042 bytes. Exact S176 base SHA256

`fc5c3da13c1c0f00b61a73377d55c4d548d12c26031d0837aa042c3f61abda68`. Clean integrator patch

`candidates/s178_side/S178_on_S176.patch`, SHA256 `0b380dc599a2a68d9a3a7118e43e0f3296b7c106f8ce295fff2be133cdf31c50`.




CHANGED: only `entry_side` may change. A supported original S161 crossing uses

the last observed exterior ego-corridor boundary side. The two accepted S176

already-inside buckets use the collision actor's horizontal center only if it

is <=0.35 or >=0.65 of image width. S174-only crossings and every ambiguous

case retain S176 side. `collision_frame`, `entry_frame`, `evasion_space`, and

all other output fields remain S176; no new model weights or dependencies.



VERIFIED: actual offline exported WSL CUDA QA on 84 public clips,

including all 76 previously uncovered human-labeled CCD clips and a paired

S176/S178 natural-change panel. 10 side outputs changed. The

paired export preserved every other output field; original source-frame IDs,

entry/collision clamp, a per-clip fault and same-process recovery passed.

Dependency install 3.60s, peak RSS

3.347 GiB, CUDA reserved

1.596 GiB, minimum free commit

50.94 GiB. No network inference attempt.

The 107-member ZIP passed static validator, member SHA256/CRC, size limits and




Official side means screen-relative LEFT/RIGHT approach of the other vehicle,

not road bearing; `src/videohackathon/evaluation.py` scores fixed-class

LEFT/RIGHT macro-F1. Official clarification:

https://dacon.io/competitions/official/236753/talkboard/417186 and

https://dacon.io/competitions/official/236753/talkboard/417202.



## Paired side F1 by labeled source and class



Values are incumbent S172 side -> S178 side. The CCD and Nexar primary panels

are AI-made, not ground truth. Individual CCD annotator views overlap the CCD

consensus and each other. DKB and Jungmin are independent human annotations

that also overlap each other. DKB 54/74 and Jungmin

27/36 use actual exported S178 outputs; their

remaining cases use the cached S161/S174 trace replay against exact stored

incumbent sides. Selected exported/cached parity differences: 

0; details in `human.json`.



| Labeled panel | n | LEFT F1 | RIGHT F1 | Macro-F1 | Repairs / breaks |

|---|---:|---:|---:|---:|---:|

| CCD consensus | 152 | 0.577->0.609 | 0.554->0.614 | 0.565->0.612 | 10 / 3 |

| CCD A1 confident | 54 | 0.509->0.545 | 0.491->0.528 | 0.500->0.537 | 3 / 1 |

| CCD A1 low | 10 | 0.250->0.250 | 0.500->0.500 | 0.375->0.375 | 0 / 0 |

| CCD A2 confident | 54 | 0.618->0.690 | 0.604->0.640 | 0.611->0.665 | 3 / 0 |

| CCD A2 low | 24 | 0.381->0.400 | 0.519->0.571 | 0.450->0.486 | 2 / 1 |

| CCD A3 confident | 54 | 0.571->0.571 | 0.538->0.644 | 0.555->0.608 | 5 / 2 |

| CCD A3 low | 7 | 0.500->0.500 | 0.333->0.333 | 0.417->0.417 | 0 / 0 |

| Nexar confident | 23 | 0.235->0.421 | 0.552->0.593 | 0.394->0.507 | 3 / 1 |

| Nexar low | 0 | - | - | - | - |

| DKB human | 74 | 0.682->0.700 | 0.571->0.647 | 0.627->0.674 | 4 / 1 |

| Jungmin human | 36 | 0.714->0.762 | 0.600->0.667 | 0.657->0.714 | 2 / 0 |



The five official sample clips have no usable side truth (`-1`). SAMA provides

lane geometry rather than a crash entrant's screen-relative side. Hangi's

public Nexar review drafts mark themselves evaluation-ineligible and do not

promote their provisional side observations to ground truth; none is silently

counted as a labeled side panel.



The selected S161+bucket policy has 52

original S161 accepted crossings among 243 public labeled clips. S174 adds

24 crossings, but using their side lowered CCD

consensus F1 from 0.599

to 0.585 and

CCD A2-confident from 0.629

to 0.574;

S174 side stays incumbent. The bucket anchor-center rule adds only when

unambiguous: CCD consensus improves from

0.599 to

0.612;

Nexar confident is unchanged by that addition.



## Evasion signal checked, not shipped



The older tracker clearance prediction was tested on accepted S161 crossings

and aligned S176 buckets. Macro-F1 results (incumbent -> proxy) are:



| Evasion panel | n | Incumbent -> clearance proxy macro-F1 | Repairs / breaks |

|---|---:|---:|---:|

| CCD consensus | 82 | 0.590->0.598 | 6 / 7 |

| CCD A1 confident | 21 | 0.475->0.475 | 1 / 1 |

| CCD A2 confident | 37 | 0.608->0.619 | 4 / 4 |

| CCD A3 confident | 27 | 0.616->0.629 | 2 / 2 |

| Nexar confident | 6 | 0.625->0.625 | 0 / 0 |

| DKB human overlap | 22 | 0.377->0.398 | 1 / 1 |

| Jungmin human overlap | 9 | 0.585->0.585 | 0 / 0 |



CCD consensus has six repairs and seven breaks despite

a small class-balance F1 gain; confident Nexar has no changed outputs.

Evasion is a physical usable-passage judgment at collision, which the single

actor trace does not settle. S178 leaves `evasion_space` untouched.



RISKS: no S178 hidden score is claimed. The main 243-clip comparison is a

cached-detector replay; human extra clips were actually exported, while other

human rows retain that replay limitation. S172's visual collision changes the

anchor on long clips when the integrator stacks this patch, so the stacked

artifact needs its own output-isolation QA. A prior broad tracker side arm

lost hidden score even after local AI-label gains; the new rule is narrower

and grounded in S161's crossing, but hidden transfer remains unproven.



NEXT: integrator can apply the clean S176-based patch to its S172/S175 stack,


champion until a better official whole-artifact result is obtained.



---



# Retained progress history



## 11:25 KST setup



Official DACON Stage2 definition: `entry_side` is the victim/other vehicle's

approach from the dashcam image's LEFT or RIGHT, not road bearing or heading.

Only these two classes are used. `evasion_space` is whether usable room to

continue or avoid exists at collision, considering lanes, boundaries,

vehicles and barriers; a tracked actor's lateral trace alone does not prove

it. Sources: https://dacon.io/competitions/official/236753/talkboard/417186

and https://dacon.io/competitions/official/236753/talkboard/417202.



The exact S176 release will be the patch base if a lawful labeled side-panel

comparison supports a change. S172 changes collision only; its side/evasion

head is the S109 side/evasion head retained by S171/S174/S176. CCD AI-made

consensus, individual CCD AI annotators and frozen Nexar contact-sheet labels

will be measured separately using fixed LEFT/RIGHT macro-F1. No private data.



## 12:25 KST geometric study and exported QA in progress



The SHA-verified S176 base is `fc5c3da13c1c0f00b61a73377d55c4d548d12c26031d0837aa042c3f61abda68`.

Its S161 tracker has 52 accepted public labeled crossings; S174 adds 24.

I replayed the 24 S174 sides explicitly because its earlier audit omitted the

side field. S161 crossing sides alone raise CCD consensus side macro-F1 from

0.56549 to 0.59867 (six repairs, one break), and confident Nexar from

0.39351 to 0.50682 (three repairs, one break). Adding all S174 sides reduces

CCD consensus to 0.58508 and hurts one confident CCD annotator panel, so the

candidate abstains on S174-only crossings. For the two S176 already-inside

buckets, adding a collision-anchor center at x<=0.35 or x>=0.65 raises CCD

consensus to 0.61183 (ten repairs, three breaks total), with the same Nexar

0.50682. These are cached-trace paired diagnostics, not exported or official

scores. `study.json` contains per-class F1 and paired IDs for every panel.



Evasion clearance replay yields CCD consensus 0.58967->0.59804 but has six

repairs and seven breaks; confident Nexar is unchanged at 0.625, with mixed

individual CCD panels. Evasion remains S176. The side-only patch has been

staged with an exact-base member audit and reverse `git apply --check`. WSL


side changes plus 76 additional human-labeled CCD clips lacking prior S109

outputs, so the full DKB and Jungmin panels can be scored before release.

