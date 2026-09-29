# S185: no-S176 stack on S175



Release: `$DATA_DIR/releases/S185_nob.zip`. SHA256 `ee3c8fa804b4cebf8000b044c4da10d194d04500ba2e7d857226778a6b99b569`.




S185 starts from exact S175, which already applies S174 entry after S172's

final visual collision. A final tracker pass on that same collision anchor

retains **S178 side only for accepted original S161 crossings**. S178 never

changed side on S174-only crossings; those keep S175 side, and S180 shifts

their accepted entry +0.2 s, clamped to collision. S181 applies collision

minus 0.85 s only after S174 completed and abstained, for original S161

`short_track`, `not_in_lane_at_collision`, and

`far_inside_when_first_seen`. Per-clip errors retain exact S175 rows.



S176 source and both bucket entry decisions are absent. The S178 side changes

from S176 bucket traces are also absent. In particular,

`inside_from_start` and `inside_when_first_seen` retain S175/S109 entry;

S181 does not cover them. `no_anchor`, `short`, `far_crossing`,

`frame_budget`, and rejected crossings also retain S175/S109 unless S174

itself accepts a crossing. The S181 gate further requires a completed S174

fallback; if it faults, S175/S109 remains. These are the pieces cleanly

separated from the original S176-based patch chain; no S178 S174-only side

rule was dropped because S178 had already abstained there.



Observed S109-source reasons on the five QA panels (panel rows may overlap):



| Original S161 reason | Rows | S174 completed | S181 gap |

|---|---:|---:|---:|

| `far_inside_when_first_seen` | 3 | 3 | 3 |

| `inside_from_start` | 7 | 7 | 0 |

| `inside_when_first_seen` | 1 | 1 | 0 |

| `not_in_lane_at_collision` | 1 | 1 | 1 |

| `short_track` | 3 | 3 | 3 |



Five real WSL CUDA panels passed exact S175 parity for Stage2 collision and

evasion, independent S174-ZIP tracker replay of entry and side on final S172

anchors, a per-clip fault and same-process recovery. Paired S182 QA confirms

S185 equals S175 on 8 S176-bucket panel

rows and equals S182 on every other panel row, isolating S176's effect.

Stage1/3 code, model files and inference prefix are byte-identical to S175.

ZIP member SHA256/CRC and submission validator passed. Peak aggregate RSS

3.196 GiB, peak CUDA reservation

1.432 GiB, minimum host free commit

48.23 GiB.



Conditional server runtime estimate: **47m36s** = S175

estimate 2423.2s plus 432.8s isolated S185 tracker time.

S172

official runtime was 34m18s versus approximately 41m local projection;

hardware and hidden stage mix differ. The official whole-inference limit is

60 minutes. No S185 official score is claimed.

