# S179 side-only stack

RESULT done. `$DATA_DIR/releases/S179_side.zip`, SHA256
`7f4b5ed48116846fc3e16210907720c72fc28461cc95b4a04c9d754c084c1558`.
It is exact S177 plus the SHA-pinned S178 crossing/bucket side rule, replayed
after S177 on S172's final collision anchor. S180 and S181 are absent.

VERIFIED: Five exported WSL CUDA panels (public, long, cascade, entry, bucket)
preserved Stage2 collision, entry and evasion exactly against S177. Side
matched independent S178-ZIP source replay on the same anchors; the long,
entry and bucket panels exercised side changes. Per-clip injected fault kept
the affected exact S177 row, and same-process recovery matched normal S179.
Stage1/3 members and inference prefix are byte-identical to S177. ZIP member
SHA256/CRC and submission validator passed. Peak aggregate RSS 3.214 GiB,
CUDA reservation 1.432 GiB, minimum free commit 65.66 GiB; lock and lease
released. Conditional server runtime estimate 48m49s, below the 50-minute

NEXT: Stop after the requested two builds. See `README.md` and `release.json`.
