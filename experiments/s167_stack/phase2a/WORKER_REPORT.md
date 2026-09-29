RESULT done



CHANGED: S169_stk.zip = exact S167 + S162, only model/stage3/s141/predict.py differs;

96 other members unchanged. Reused multi-arm builder. Receipt and README saved.

ZIP: $DATA_DIR/releases/S169_stk.zip

SHA256: 94b963cd8949102ee1a0e53da9b9c57c74252219b863a5e15fbe3f6f937ba3ef



VERIFIED: Real offline CUDA Stage1=S163 (13 clips), Stage2=S156 (12 clips),

Stage3=S162 (2,998 rows; exactly 68 accel changes; steering/STOPPED=S156).

Two-clip cached-path fault injection restores 58 OPEN_004 rows to S156 after

partial mutation, preserves the other clip, and recovers S162 on the next call.

Hooks restore; no legacy fallback. CRC/member hashes and submission validation pass.

Peak aggregate RAM 2.766 GiB;

max 2 processes; CUDA reserved 1.432 GiB.



RISKS: Historical shared-load runtime 843.06s -> 447.54s;

conditional server estimate 26m22s, not official.




NEXT: Stopped after phase 2a; await S160 phase-2b prompt.

