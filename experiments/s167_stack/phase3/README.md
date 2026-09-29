# S171 entry stack



ZIP: $DATA_DIR/releases/S171_ent.zip

SHA256: f2f3b55617d80fdee1afabe930d426bc92c99ceb7f48f2953e6504efda280315

Bytes: 1222043478; members: 104. Ready: 2026-09-28T04:15:22.348836+09:00.



Exact S170 (uploaded row 105253) plus S161. Only S118 adapter.py and rule.json

change; three S161 files are added. The other 99 S170 members are identical.

All added/modified files are byte-identical to the pinned mode-repaired S161 arm.

The existing build_stack.py reproduces the five-arm stack from S156; S164 wraps

the final dispatch. The prior reviewed S144 CRLF merge remains unchanged.



Final entry rule: compute S160 collision first, retaining S170's current entry.

Run S161's observed-crossing tracker anchored to that final S160 collision.

If S161 accepts a crossing, select its frame number; otherwise retain S170 entry.

Then clamp entry = min(selected entry, S160 collision). A tracker load failure

returns the supplied S170 rows; a per-clip error retains that S170 entry. Side

and evasion are untouched. S161's fixed validator permits entry-only tracking

with preserve_s109_fields, while rejecting track side=true. No cumulative

cross-file timing budget is used.



Independent pinned-S161 tracker replay verifies the intended entry rule on every

public/long/cascade and three-clip entry diagnostic row, using actual S170

collisions as anchors. Standalone S161 CSVs use S156 collision anchors and are

not assumed to be the correct S171 entries. Protected Stage2 fields equal S170.

Real injected tracker errors retain complete S170 rows; the next same-process

call recovers S171 with observable entry changes and restored runtime hooks.

CPU fixtures also verify load fallback, clipping, recovery and the mode guard.

Stage1 (13 clips) and Stage3 (2998 rows) match S170 CSV bytes. CRC/member SHA256

and tools/validate_submission.py pass. No private evaluation data was accessed.



Matched inference total 501.33s; conditional scaling from S156's 49m40s gives

29m32s. Shared load and server stage mix remain

uncertain; this is not an official runtime. Independent replay/fault/diagnostic

work and framework imports are excluded. Per-panel S170/S171 times are in release.json.

RSS peak 3.096 GiB under 4 GiB;

CUDA reserved 1.465 GiB under the 2 GB lease;

minimum sampled free commit 47.49 GiB;


between panels. PowerShell is NonInteractive; command/QA waits are bounded.

The earlier 3 GiB limit was a local machine-sharing guard, not a competition


12 GiB free commit. The measured peak above records that approved exception.



S161 remains exploratory: observed entry proxies do not establish hidden-score

gain, and anchor/geometry limits remain. Other arms retain their recorded risks.





with five retries at 0.5 seconds on PermissionError. Failed writes and stale

samples warn and skip; they never stop inference. Actual observed RSS/process

or fresh free-commit breaches still enforce resource caps. Passed baseline,

public and long evidence was retained against preserved historical source

snapshots; only remaining checks were run. The earlier cascade stale-monitor

abort was a harness failure, with RSS below 1 GiB and free commit over 70 GiB.

