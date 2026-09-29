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




