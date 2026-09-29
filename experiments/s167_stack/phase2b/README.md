S170_all.zip: $DATA_DIR\releases\S170_all.zip

SHA256: 66d0be5c0efbe5256055a1eb6662b3106aa0b4b4813a61d29b35314fe02a94f4

Bytes: 1222034227; members: 101

Exact S169 plus S160 collision localizer. One changed S144 Python file and four additions;

96 S169 members unchanged. CRLF normalization resolves the S164/S160 textual overlap;

merged source equals S164 batch_patch applied to pinned S160 source. Default detector batch=1.



Real offline CUDA Stage1 equals S163; Stage2 public/long equals retained S160,

cascade equals a fresh pinned S160 reference; other Stage2 fields equal S156.

Stage3 equals S162 (2998 rows, 68 acceleration changes, steering/STOPPED unchanged).

Localizer exceptions on 00019 and worker-prefetched 00060 return complete S156 rows;

00019 collision restores 609->612. Unaffected 00076 remains 597. Next call recovers S160.

S164 lazy path and prefetch remain active, hooks restore. CPU fixtures additionally

exercise batch=2 detector retry and invalid localizer output. Actual CUDA uses batch=1.



Historical matched total 843.06s -> 540.92s;

conditional server estimate 31m52s from 49m40s.

Not an official runtime prediction: shared load and unseen stage mix remain uncertain.

Long/cascade timings and all resource measurements are in release.json.

Two fault recovery attempts hit the original 3 GiB aggregate RAM guard at 3.037

and 3.001 GiB; both were stopped and their logs/injected CSVs are preserved.


was a local machine-sharing guard, not a competition/server limit. Free host

commit was checked before each remaining run and every 5 seconds thereafter;

the QA guard required >=12 GiB. Allocator trimming was retained; the candidate

was unchanged. This is a documented criterion revision, not an unchanged pass.

The first revised-cap attempt stopped on a transient Windows/WSL commit-file

read race, while RSS and free commit both met the approved limits. That attempt

is retained; the reader now tolerates sharing races for at most 30 seconds

since the last valid sample, while preserving the low/stale-commit guard.

Final peak aggregate RSS: 3.068 GiB;

CUDA reserved: 1.465 GiB;

minimum monitored free host commit: 52.61 GiB.


S162/S163 accuracy risks remain; no hidden-score improvement is claimed.

