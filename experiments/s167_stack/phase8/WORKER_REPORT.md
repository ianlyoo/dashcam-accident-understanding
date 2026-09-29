RESULT done (S185 no-bucket stack, 2026-09-28 17:58 KST)

CHANGED: `$DATA_DIR/releases/S185_nob.zip` is exact S175 plus
S178 side on accepted original S161 crossings, S180 +0.2 s on accepted
S174-only crossings, and S181 collision-gap fallback after completed S174
abstention. S176 first-frame/first-observation entry buckets and their side
changes are absent. SHA256:
`ee3c8fa804b4cebf8000b044c4da10d194d04500ba2e7d857226778a6b99b569`.

VERIFIED: Five real WSL CUDA panels passed protected-field parity with S175,
independent replay on final S172 collision anchors, per-clip S175 fault
fallback and same-process recovery. Paired S182 isolation: eight S176-bucket
rows equal S175; every other tested row equals S182. ZIP member hashes/CRC
and submission validator pass. Peak aggregate RSS 3.195 GiB, CUDA reservation
1.432 GiB, minimum free commit 48.23 GiB. No owned QA process, lock or GPU
lease remains. Full evidence: `README.md`, `release.json`, QA receipts.

RISKS: Conditional server runtime estimate 47m36s, below the 50-minute
projection flag; server hardware and hidden stage mix may differ. No official
S185 score is claimed.

