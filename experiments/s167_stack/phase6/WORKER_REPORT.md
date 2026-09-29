# S182 chain integration — completed

## Final status, 15:58 KST

RESULT done. `$DATA_DIR/releases/S182_full.zip` SHA256
`9307c6001253a07e334f8eff881ec24c87c3aef8eeaa23983efd0f11fd2a2918`.
Five CUDA panels, exact S177 protected-field parity, independent S178/S180/S181
entry/side replay on S172 collision anchors, per-clip fallback/recovery,
4 GiB RSS and 2 GiB CUDA caps, Stage1/3 member identity, ZIP member SHA256/CRC
and submission validator passed. Conditional server runtime estimate 49m00s,

and fixed it without changing candidate model code. The corrected public
receipt and all remaining panels passed; see `DIAGNOSIS.md` and `release.json`.

## Superseded stop record

RESULT partial. `$DATA_DIR/s167_stack/phase6/candidate` was staged
from exact S177 plus SHA-pinned S178→S180→S181 source chain. The original
S177 members are unchanged except the appended Stage2 dispatcher; the chain
uses S172's final collision anchor. No release ZIP was packaged. S179 was not
started because the requested build order puts it after S182.

CPU rule fixture passed for S178 original-crossing position preservation,
S180 +0.2 s shift and collision clamp, and S181 0.85 s gap/abstention.
The public WSL CUDA export passed Stage2 collision/evasion parity against
S177 and an independent S181-ZIP source replay of entry and side on S172
anchors. In a per-clip injected frame-enumeration fault, the affected clip
retained its exact S177 row, but at least one unaffected clip differed from
the normal exported run. The initial failure used strict DataFrame equality;
a second attempt compared output values and reproduced the mismatch. It may
be baseline drift in the harness or fault-dependent tracker state; this is
not yet isolated. Stopping at the developer-mandated two-attempt limit.

No Stage1/3 CUDA QA, remaining Stage2 panels, validator, packaging, runtime
absent, the launcher has exited, and neither S182 nor S179 release ZIP exists.

NEXT: capture the actual S177 base rows from the same exported run, persist
both faulted and recovered outputs, and distinguish a harness baseline
mismatch from a real cross-clip fault effect. If resolved, resume the remaining
S182 checks, then build and QA S179.
