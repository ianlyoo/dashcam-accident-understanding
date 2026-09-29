# S176 bucket arm — delivery



RESULT done. Release `$DATA_DIR\releases\S176_bucket.zip`; SHA256

`fc5c3da13c1c0f00b61a73377d55c4d548d12c26031d0837aa042c3f61abda68`; 1,222,047,529 bytes; validator and member SHA/CRC

passed. Exact base S174 SHA256 `d2af1b1c63625035aa99199b0cebbb4d342eba7425190f75b9f2a6c49e182bdc`. Patch for integrator:


commit, or push.



CHANGED: only Stage2 entry_frame can change, then the inherited collision

clamp. Selected rules are first clip frame for S161 `inside_from_start` and

first observed inside minus 0.2s for `inside_when_first_seen`, only after

S174 abstains. Other buckets and accepted S161/S174 results remain unchanged.



VERIFIED: 243 public paired labels; `inside_from_start` eligible primary hits

6→28/44 (26 repairs, 4 breaks); `inside_when_first_seen` 0→1/12;

zero-gain buckets excluded. Full CCD consensus 20→

40/79, CCD A3 low

16→17/45,

confident 13 2→4. Exported offline CUDA QA passed

(9 clips, 5 natural changes), dependency install

2.99s, peak RSS

3.197 GiB and CUDA reserved




RISKS: paired 243-clip result is a cached-detector trace replay, not a full

exported run. The 0.2s rule gains only one low-confidence label, and

first-frame entry has prior hidden-set failure when used globally. No S176

official score is claimed. Full per-bucket counts and limitations are in

ANALYSIS.md.



NEXT: integrator can apply the clean S174 patch onto the S172-based stack;


a better official whole-artifact result exists.



---



# Retained progress history



# S176 bucket arm — work in progress



## 10:15 KST setup



The assigned base is exact S174, SHA256

`d2af1b1c63625035aa99199b0cebbb4d342eba7425190f75b9f2a6c49e182bdc`.

The paired study uses S174's public 243-clip audit and exact exported S171

entry/collision rows, with S174's cached-detector replay as the current

candidate proxy. S174 already changes 11 clips in the four target S161

abstention reasons; new rules will apply only where S174 itself abstains.

The study will report one primary label per clip by fixed source precedence

and the overlapping source views separately. No private evaluation data.



## 10:30 KST measured rule selection and staging



On the 44 eligible `inside_from_start` clips, the first clip frame scores

28/44 primary-label hits versus 6/44 S174 (26 repairs, 4 breaks). CCD

consensus in that bucket is 2->22/24 (20 repairs, no breaks); 2 confident

Nexar cases are 0->2. On 12 eligible `inside_when_first_seen` clips,

first observation minus 0.2s scores 1/12 versus 0/12. First observed

inside itself ties 0/12. `far_inside_when_first_seen` ties 1/10 for

first-inside or extrapolation; `not_in_lane_at_collision` drops 6->5/19

with first-inside/extrapolation. Those two buckets stay S174. The universal

first-clip alternative for latter buckets was measured for context only;

it is not definition-faithful there and was excluded from selection.



The exact S174 ZIP was verified and staged with an append-only S176 extension.

Only predict.py changes among S174 members; S176 adds a notice and exports

`S176_on_S174.patch`. Eight focused scope/failure tests pass. Inferred paired

public replay changes 50/243 entries; CCD consensus 20->40/79, CCD A3 low

16->17/45, confident Nexar 1->3/10 plus S117 unchanged 1/3. This replay

uses cached S161/S174 traces; exported CUDA QA remains pending.



## 11:00 KST exported paired phase



The actual offline WSL CUDA baseline and candidate exports both completed on

9 public clips. Five entry frames changed; protected output columns match

exactly. The three targeted CCD clips have the intended exact exported S174

S161 reasons. Baseline took 511.9s and S176 521.9s under shared-machine

contention. Candidate peak RSS was 3.197 GiB, CUDA reserved 1.465 GiB,

minimum free commit 52.57 GiB. The QA supervisor released its lease and





An integrator patch read-only check exposed Windows CRLF line endings in the

patch file, even though the staged candidate source is correct. The LF form

passed `git apply --reverse --check`. After QA exits, a metadata-only patch

rebind will regenerate the LF patch and prove every staged ZIP member hash is

unchanged. This does not affect the already measured model output.

