# S185 without S176: rule separation

human panel while its observed gains came from AI-made CCD consensus labels;
results, not new measurements from S185 QA.

S185 uses exact S175/S174 as its base. It adds a separate pass on S172's final
collision frame; there is no S176 module, first-frame bucket, or
first-observation-minus-0.2-second bucket in the staged artifact.

The original S178 patch changed side for **accepted original S161 crossings**
and two S176 buckets. It deliberately abstained on S174-only crossings after
its side comparison worsened on a labeled panel. S185 retains only the
accepted-S161-crossing side change. The bucket side changes are dropped;
S174-only accepted crossings keep S175 side. This is the original S178
crossing policy, not an added S174 side policy.

S180's +0.2-second entry shift is applied only when S174 accepts a crossing
after S161 abstains. S181's collision-minus-0.85-second gap is applied only
when S174's fallback completed and abstained and the original S161 reason is
`short_track`, `not_in_lane_at_collision`, or
`far_inside_when_first_seen`. The gap does not depend on S176's state; S185
captures S174 completion directly.

Original S161 `inside_from_start` and `inside_when_first_seen` now keep their
S175/S109 entry if S174 abstains. `no_anchor`, `short`, `far_crossing`,
`frame_budget`, and unsupported/rejected crossings also retain S175/S109.
S174 accepted crossings instead follow the S180 shift. Any per-clip error
retains the exact S175 row. The final README will include the observed reason
counts from exported QA and the S182/S185 paired isolation check.
