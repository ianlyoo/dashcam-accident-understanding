RESULT done (S185 no-bucket stack, 2026-09-28 17:58 KST)

CHANGED: `$DATA_DIR/releases/S185_nob.zip`, SHA256
`ee3c8fa804b4cebf8000b044c4da10d194d04500ba2e7d857226778a6b99b569`.
Exact S175 carrier plus S178 accepted-S161 side, S180 S174-only shift, and
S181 gated collision-gap fallback. S176 bucket entry and side are absent.

VERIFIED: Five WSL CUDA panels, independent rule replay, protected-field
parity, per-clip fault fallback/recovery, S182 isolation, member hashes/CRC,
and submission validator passed. Eight S176-bucket rows equal S175; every
other tested row equals S182. Peak RSS 3.195 GiB, CUDA 1.432 GiB, minimum

RISKS: Conditional server runtime estimate 47m36s. No official S185 score.

`phase8/release.json`, and `phase8/WORKER_REPORT.md`.

---

RESULT done (S182 phase 6 and S179 phase 7, 2026-09-28 16:21 KST)

CHANGED: `$DATA_DIR/releases/S182_full.zip` = exact S177 plus
ordered S178→S180→S181 side/entry rules, SHA256
`9307c6001253a07e334f8eff881ec24c87c3aef8eeaa23983efd0f11fd2a2918`;
`$DATA_DIR/releases/S179_side.zip` = exact S177 plus S178 side
only, SHA256
`7f4b5ed48116846fc3e16210907720c72fc28461cc95b4a04c9d754c084c1558`.

VERIFIED: Each passed five real WSL CUDA panels, independent rule replay on
S172 final collision anchors, protected-field parity, per-clip fault fallback
and same-process recovery. Stage1/3 members/inference prefix are exact S177.
ZIP member SHA256/CRC and submission validator passed. Peak aggregate RSS was
3.163 GiB for S182 and 3.214 GiB for S179; CUDA reservation stayed below 2 GiB.

RISKS: Conditional server runtime estimates are 49m00s (S182) and 48m49s
(S179), both below the 50-minute flag threshold; hardware and hidden stage

NEXT: Stop after both requested builds. Details in `phase6/README.md`,
`phase6/release.json`, `phase7/README.md`, and `phase7/release.json`.

---

RESULT done (S177 phase 5, 2026-09-28 14:21 KST)

CHANGED: $DATA_DIR/releases/S177_bkt.zip, 1,788,791,743 bytes,
121 members; exact S175 carrier plus pinned S176 bucket-only entry pass.
SHA256: b30e56829cfbbad98bd3121b1c7ced6760fb8bb1da97f73aba006d45cb26adea.

VERIFIED: All non-entry outputs match S175 on public, bucket, long, cascade and
entry panels; independent S176-from-ZIP replay on final S172 collision anchors
matches every entry. Nine injected errors retain exact S175 rows, and
same-process recovery restores S177. CRC/member hashes and submission validator
pass. Peak RSS 3.207 GiB, CUDA reserved 1.432 GiB, two processes, minimum free
commit 42.96 GiB. Full evidence: phase5/README.md, release.json,
WORKER_REPORT.md.

RISKS: Conditional server runtime 45m53s; no official S177 result. First-frame
transfer remains unproven.


---

RESULT done (S175 phase 4, 2026-09-28 12:58 KST)

CHANGED: $DATA_DIR/releases/S175_stk.zip, 1,788,778,606 bytes,
118 members; exact S172 carrier plus pinned S174 entry module and S175
postprocessor. SHA256: 3827273f061a7222e22d693207d1f5897bcefe114b20802adf48c971ff1e98be.

VERIFIED: Stage1/3 and Stage2 collision, side and evasion match S172; independent
S174 replay matches entry on final S172 collision anchors, followed by clamp.
Injected tracker error retains exact S172 entry and same-process recovery works.
CRC/member hashes and submission validator pass. Peak RSS 3.111 GiB, CUDA
reserved 1.432 GiB, two processes, minimum free commit 50.365 GiB. Full
evidence: phase4/README.md, release.json, WORKER_REPORT.md.

RISKS: Conditional server runtime 40m23s; no official S175 result. Entry proxy
transfer remains unproven.


---

RESULT done (S171 phase 3, 2026-09-28 04:15 KST)

CHANGED: $DATA_DIR/releases/S171_ent.zip, 1,222,043,478 bytes,
104 members; exact S170 + S161, with 99 S170 members unchanged.
SHA256: f2f3b55617d80fdee1afabe930d426bc92c99ceb7f48f2953e6504efda280315

VERIFIED: All Stage2 protected fields match S170; S161 entry on final S160
collision anchors, then clamp, matches independent replay. Tracker errors retain
exact S170 rows; same-process recovery restores S171. Stage1 (13 clips) and
Stage3 (2998 rows) match S170 byte-for-byte. Public/long QA retained, not rerun.
CPU fixture, atomic monitor failure checks, CRC/member SHA256 and validator pass.
Peak RSS 3.096 GiB, CUDA reserved 1.465 GiB, two processes; minimum free commit
47.487 GiB. Full evidence: phase3/README.md, release.json, WORKER_REPORT.md.

RISKS: Conditional server estimate 29m32s from matched-panel scaling; shared
load/stage mix limit precision. Entry proxy transfer is unproven.


---

Historical interrupted phase3 report:

RESULT blocked (S171 phase 3, 2026-09-28 03:25 KST)

CHANGED: Exact S170 + S161 staged; corrected CPU fixture and absolute
interpreters verified. Details: candidates/s167_stack/phase3/WORKER_REPORT.md.

VERIFIED: CPU clamp/fallback/recovery, identities, S170 diagnostic baseline,
S171 public/long Stage2 CUDA parity and independent entry replay.

RISKS: Two further harness failures (Windows sharing violations on supervisor
log, then host_commit.json). No candidate mismatch; no S171 package/hash.

NEXT: Stopped per worker two-failed-attempt rule. Repair commit writer/child
cleanup, then remaining CUDA QA/validator/package. No owned QA processes,

---

Previous completed phase:

RESULT done


CHANGED: $DATA_DIR/releases/S170_all.zip; 1,222,034,227 bytes, 101 members.

SHA256: 66d0be5c0efbe5256055a1eb6662b3106aa0b4b4813a61d29b35314fe02a94f4



VERIFIED: Stage1=S163 (13 clips); Stage2=S160 (12 clips), protected fields=S156;

Stage3=S162/S169 (2998 rows, 68 accel changes), steering/STOPPED=S156.

S164 prefetch/lazy path active. Real localizer failure returns exact S156 rows,

preserves the other clip, restores hooks and recovers S160 on the next call.

All CUDA evidence, member CRC/SHA256 and tools/validate_submission.py passed.



RISKS: Conditional server estimate 31m52s from 49m40s; matched panels improve

843.06->540.92s (35.8%). Shared-load estimate, not an official server measurement.


free commit 52.61 GiB, CUDA reserved 1.465 GiB, maximum two processes.

Earlier 3 GiB guard breaches and commit-monitor race are preserved and

explained in phase2b/README.md. Candidate unchanged by QA guard revisions.






---

Prior phase report preserved below:



RESULT done



CHANGED: Phase-1 S167 release built from exact S156 + S163 + S164; 97 members.

Reproducible multi-arm builder, QA launcher, source/member manifests and release receipt saved.

ZIP: $DATA_DIR/releases/S167_stk.zip

SHA256: 057ac49a7e0f1aeb571b2f6973e290813c77172a80f7cc89fae254f8ff9481cb



VERIFIED: Real offline WSL CUDA parity: Stage1=S163 (13 clips); Stage2=S156

(12 clips); Stage3=S156 (5 clips, 5,992 rows). Full CRC/member hashes and

tools/validate_submission.py pass. Matched timing 924.77s ->

568.48s (38.5% reduction); ratio-scaled server estimate

30m32s from inherited 49m40s. Peak aggregate RAM

2.746 GiB; maximum 2 processes;

CUDA reserved 1.432 GiB.



RISKS: Timing uses historical base runs under variable shared load; no official

score or server measurement. S163 retains its known exploratory false-positive




NEXT: Stopped after phase 1. Await explicit phase-2 prompt for S160/S162 arms.

