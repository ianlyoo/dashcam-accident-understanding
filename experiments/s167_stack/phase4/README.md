# S175 S172 + S174 stack



Release: `$DATA_DIR/releases/S175_stk.zip`. SHA256 `3827273f061a7222e22d693207d1f5897bcefe114b20802adf48c971ff1e98be`.

Bytes 1788778606; 118 members. S172 carrier official

score 0.66245 and 34m18s; S175 has no official result or upload.



`build_stack.py --action overlay` uses exact SHA-pinned S171 as the common base,

S172 as carrier, and S174 as the entry-only arm. S172's member delta is

`inference.py`, `model/stage2/s144/predict.py`, and twelve additions; S174

changes only `model/stage2/s161/predict.py` and adds its notice. These deltas

do not overlap. S175 adds a tiny dispatcher and postprocessor; all other S172

members are byte-identical. The pinned S174 code is byte-identical to its ZIP.



Final order: complete exact S172 Stage2 prediction, including visual DINOv2-L

collision; run S161/S174 entry on each final S172 collision anchor; select an

accepted crossing or retain S172 entry, then clamp to S172 collision. A tracker

load, clip, or close error retains the exact S172 entry. Collision, side and

evasion remain exact S172. S172's original S161 pass is retained to provide

the exact fallback; the final entry pass adds work and is included in timing.



QA: S172 and S175 exported public, long, cascade and entry panels; protected

Stage2 columns identical; independent pinned-S174 replay on S172 collisions

matched every entry. Cascade-panel injected tracker errors returned exact S172

rows, with two observable changes after recovery, and

same-process recovery returned S175. Stage1 13 clips and Stage3 2998 rows

match S172 by unchanged members and fresh outputs. Network audit, at most two

processes, <=2 GiB CUDA lease, <=4 GiB aggregate RSS, >=12 GiB host free





Matched long panel S172 363.74s -> S175 266.73s; the negative wall

difference -97.01s reflects changing shared load and is not

credited as a speedup. The isolated S175 final entry pass took

7.98s for three clips. Conditional runtime: 40m23s

from official S172 34m18s plus that pass scaled to 137 clips and a

0.97s prior observed tracker load. S174 standalone local

projection was 74.63s/137 clips; S175 also repeats S161 on final collisions.

Shared RTX load, hidden stage mix and server hardware make this an estimate,

not an official runtime. Peak RSS 3.111 GiB, CUDA reserved

1.432 GiB, minimum free commit 50.37 GiB.


