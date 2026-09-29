# S182 full chain on S177



Release: `$DATA_DIR/releases/S182_full.zip`. SHA256 `9307c6001253a07e334f8eff881ec24c87c3aef8eeaa23983efd0f11fd2a2918`.




Exact S177 carries S172 visual collision, S161/S174 entry and S176 bucket entry.

The final pass replays the SHA-pinned S178 side, S180 +0.2 s S174-only entry

shift and S181 0.85 s fallback-gap rules in that order on **S172's final

collision frame**. Accepted S178 crossing/bucket side replaces only side;

accepted S180 or S181 entry replaces only entry, always clamped to collision.

All other fields remain exact S177. A per-clip error retains that clip's exact

S177 row; a load/close error retains the whole exact S177 output. Original

S178 adapter source is included as provenance; its side-consumption semantics

are applied by the final postprocessor to avoid changing S177's earlier pass.



The S178, S180, S181 ZIP/source chain and exact S177 carrier were hash-checked.

Five real WSL CUDA panels (public, long, cascade, entry, bucket) passed Stage2

collision/evasion parity and independent chained entry/side replay on the

same S172 anchors; a per-clip injected fault and same-process recovery passed.

Stage1/3 code, model files and dispatcher prefix are byte-identical to S177.

Validator and ZIP member SHA256/CRC passed. Peak aggregate RSS

3.163 GiB, peak CUDA reservation

1.432 GiB, minimum host free commit

65.69 GiB.



Conditional server runtime estimate: **49m00s** = S177

estimate 2753.3s plus 186.5s isolated S182 tracker time.

The

official S172 run was 34m18s versus approximately 41m local projection,

so the estimate is conservative and hardware/stage mix may differ. Official

whole-inference limit remains 60 minutes. No S182 official score is claimed.

