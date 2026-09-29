# S181 S109 fallback entry audit and candidate

RESULT done. Release `$DATA_DIR\releases\S181_fallback.zip`, SHA256 `547323b4f5a5c34ee273c67ac2cba91694844ea179bda248aa870c4b89548b77`.

## Abstention and timing audit

The base is exact S180 (`S180_resid.zip`, SHA256
`d5033adfce1d40fb5fc98289196fb791b274ad4529d111c3d2fbacaa9623a84b`).
On the four lawful entry panels, 91 annotations (89 unique clips) use
S109 after S161/S174/S176 abstain; 68 annotations miss hit@0.3.
CCD consensus has 31 such rows, CCD A3-low 19, confident Nexar 5,
and DKB human 36. The panels overlap by clip ID.

| Final S161 reason | Rows | S109 hits | Misses | Why no geometric entry |
|---|---:|---:|---:|---|
| `short_track` | 34 | 9 | 25 | Track lacks enough actor observations. |
| `no_anchor` | 32 | 9 | 23 | No usable collision actor was selected. |
| `not_in_lane_at_collision` | 20 | 4 | 16 | Tracked actor is not inside the modeled ego corridor at its collision-end observation. |
| `far_inside_when_first_seen` | 3 | 0 | 3 | Actor was already well inside when first seen; crossing unobserved. |
| `crossing` | 2 | 1 | 1 | Observed crossing failed the strict acceptance gate. |

Among rows with detailed S174 replay, 48 end with
`s174_no_supported_crossing` (34 misses) and 16 with
`s174_no_anchor` (11 misses). The other 27 have exported final
reasons but no retained S174 candidate-reason replay, so their
specific S174 failure cannot be assigned. S176 handles only
`inside_from_start` and `inside_when_first_seen` after S174
abstains; none of these 91 final fallback rows met that rule.
Of the two S161 `crossing` rows, one has only one outside
observation; the other has an overly wide bracket and one
outside observation. Their strict S161 gate correctly abstains.

The original S161 actor trace is missing in 78/91 fallback
annotations, all outside in six, mixed in five, and all inside
in two. Missing traces usually follow no anchor or short track;
they do not prove that the real entrant never touched the lane.
Public labels do not identify the collision actor, so an actor
association error cannot be counted reliably. In five rows the
collision prediction precedes even the allowed labeled entry
interval: these are collision/entry inconsistencies, potentially
an anchor error or annotation mismatch rather than a timing shift.

The labeled entry-to-collision gap on these fallback rows has
median 1.10 s,
IQR 0.50–1.80 s;
among misses it is 1.10 s
(IQR 0.50–2.00 s).
Signed here means collision time minus labeled entry midpoint;
negative values place the labeled entry after predicted collision.

## Paired alternatives and selection

The median crossing-to-collision gap of 52 *accepted* S161
public clips is 0.85 s (IQR 0.50–1.70 s). This geometric trace
quantity does not use fallback entry labels. Applying collision
minus that gap to every S109 fallback row gives 23->37/91 hits,
with 22 repairs and eight breaks. Restricting it to final
`short_track`, `not_in_lane_at_collision`, and
`far_inside_when_first_seen` retains the same 23->37/91 score,
with 17 repairs, three breaks and 53/91 rows changed. `no_anchor`
has five repairs and five breaks; rejected `crossing` has no net
gain. Both keep S180. This is a collision-relative proxy for the
unseen wheel crossing, not a measured tire contact in each clip.

The cheaper original-trace first-touch rule changes only four
fallback rows: 23->22/91, zero repairs and one break. It is
rejected. Most fallback rows lack a usable traced boundary
transition, which limits this alternative.

| Panel | S109 fallback S180 -> S181 hits | Fixed gated gain |
|---|---:|---:|
| CCD consensus 79 | 9 -> 12 / 31 | +3 |
| CCD A3-low 45 | 5 -> 8 / 19 | +3 |
| Confident Nexar 13 | 1 -> 2 / 5 | +1 |
| DKB human 74 | 8 -> 15 / 36 | +7 |

Leave-one-panel-out removes **all** training labels for a
held-out clip ID. The 0.85 s S161 gap remains the median in all
four folds after excluding held-out IDs. Selecting favorable
S161 reason buckets on the other three panels independently
yields held-out gains +3 CCD consensus, +3 CCD A3-low, +1
Nexar, and +7 DKB. The fixed three-reason policy has identical
held-out gains. S161 gap fitting is label-independent, but
bucket selection uses these same public panels; the small Nexar
sample and mixed AI-made labels limit transfer confidence.

## Scope, verification, and limits

The patch changes only Stage2 `entry_frame`, then inherited
collision clamp. It runs after S174/S176 complete and abstain;
S161/S174/S176 accepted decisions and S180 timing remain intact.
No weights, dependencies, or detector passes are added. The
staged tree differs from S180 only in `model/stage2/s161/predict.py`
plus `model/licenses/S181-NOTICE.txt`; `S181_on_S180.patch`
reverse-applies cleanly.

Auxiliary retained AI-label sensitivity spans 111
cached fallback clips: 8 label views improve, 2 tie,
and 0 decline (`sensitivity.json`). These views overlap
the primary panels and each other; they are not extra
independent clips or an additional pooled score.

Offline exported CUDA paired QA passed on 10 public clips:
4 entry changes and no protected-field changes.
Source frame IDs, entry<=collision, per-clip failure fallback,
and same-process recovery passed. Dependency install took
3.93 s; peak RSS
2.57 GiB; peak CUDA reservation
1.23 GiB; minimum free commit
50.88 GiB. No network attempts.

ZIP validator passed on `$DATA_DIR\releases\S181_fallback.zip`. All
109 member SHA256s/CRC were checked; ZIP SHA256
`547323b4f5a5c34ee273c67ac2cba91694844ea179bda248aa870c4b89548b77`.

RISKS: The collision-gap rule is an exploratory proxy when the
actor crossing is unobserved. `not_in_lane_at_collision` can
reflect a wrong actor or geometry; public labels cannot separate
them. No hidden or official gain is claimed. CCD and Nexar
entry labels are AI-made, DKB labels human. Most 243-clip
predictions are cached detector replay; exported CUDA QA checks
a smaller selected public set.

NEXT: Apply the clean S180-based patch to the stack and run
