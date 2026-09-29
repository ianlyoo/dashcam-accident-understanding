# S182 entry-chain isolation diagnosis

RESULT: harness defect fixed; no S174/S176/S178/S180/S181 candidate-state
leakage was found. The corrected exact integrator public CUDA QA passes.

## Reproduced cause

The original `phase6/qa.py` bound `frame_paths()` over the local name
`old_paths`. Its fault block later rebound `old_paths` to the wrapper
itself. Python closures capture the variable binding, so every unaffected
clip called `frame_paths()` recursively. The instrumented original harness
reproduced the failure: the injected first clip raised the intended
`RuntimeError`, while all four unaffected clips recorded
`RecursionError: maximum recursion depth exceeded` and retained S177
entries. DataFrame dtypes were identical; the values differed because
the unaffected clips had hit the harness error path.

The saved phase5 S177 baseline and the S177 rows captured from the same
normal exported S182 run match exactly on all five public sample clips.
Thus baseline drift is excluded for the reproduced public failure.

I renamed only the fault block alias to `fault_source_paths`, leaving the
outer `old_paths` binding intact. No candidate model source, patch, ZIP,
or staged S182 runtime changed. The corrected `phase6/qa.py` has a new
source hash; the staged build hash is unchanged.

## Inference-state audit

S174 constructs its candidate tracks, lane variants, and detection cache
inside each `Tracker.analyze` call. S176 resets its two instance fields
`_s176_original_result` and `_s176_fallback_completed` at the start of
each clip. S178 reads that clip result to choose side; S180 reads the
current result to apply a fixed 0.2 s shift. S181 uses a literal 0.85 s
crossing-to-collision gap fitted offline on public S161 crossings; the
packaged `predict.py` computes no median or statistic from test clips.
The detector is in eval mode. No module-level clip accumulator, running
median, or previous-clip feature is used by these arms. The S182 wrapper
creates a new tracker and per-call diagnostics for each Stage2 invocation.

## Exported invariance evidence

With the same captured S177 rows, the five public sample clips have
identical values in original order, reversed order, a same-base fault run
for all unaffected clips, and same-process recovery. Each of the five
also matches when the S182 chain is run alone in a fresh WSL process.
Whole S182 Stage2 fresh-process singles match for 5/5 public
clips as of this report. The original sample panel exercises one S180
crossing shift and two S181 gap decisions.

A second three-clip panel on frozen S177 rows exercises S178 side:
`ccd_000080` changes LEFT to RIGHT. Original order, reversed order,
faulted unaffected outputs, recovery, and all three fresh-process
single-clip exports match exactly.

The corrected exact integrator public QA receipt reports five rows,
three changed entries, zero changed sides, one intentional fault call,
exact S177 fallback for the affected clip, and exact S182 recovery.
Peak aggregate RSS 2.35 GiB,
CUDA reservation 1.21 GiB, and minimum
host free commit 66.55 GiB;
no network attempts. The QA source and build hashes in the receipt match
the current files.

Evidence: `$DATA_DIR/s167_stack/phase6/qa_diagnose/`
(original reproduced fault, live base and row comparison);
`$DATA_DIR/s181_fallback/diagnosis/` (public order,
fault, recovery, singles); `.../diagnosis_side/` (active S178 side);
and `$DATA_DIR/s167_stack/phase6/qa/public.json`
(corrected integrator acceptance receipt).

NEXT: the integrator can reuse the corrected public QA receipt, run its
remaining panels and whole-artifact checks, then package. No entry-arm
patch revision is needed for this issue.
