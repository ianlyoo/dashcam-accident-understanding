"""Write final S174 analysis and delivery notes from bound receipts."""
import collections,json,pathlib
H=pathlib.Path(__file__).resolve().parent;D=pathlib.Path('$DATA_DIR')
release=json.loads((H/'release.json').read_text());metrics=json.loads((H/'metrics.json').read_text())
qa=json.loads((H/'qa.json').read_text());labeled=json.loads((H/'audit_labeled_summary.json').read_text())
nexar=json.loads((H/'audit_750_summary.json').read_text());anchors=json.loads((H/'anchors_receipt.json').read_text())
labrows=[json.loads(x) for x in (H/'audit_labeled.jsonl').read_text().splitlines() if x.strip()]
allrows=[json.loads(x) for x in (D/'s174_cov/audit_750.jsonl').read_text().splitlines() if x.strip()]
evaluated=[json.loads(x) for x in (D/'s174_cov/evaluation_gpu.jsonl').read_text().splitlines() if x.strip()]
assert len(labrows)==len(evaluated)==243 and len(allrows)==750
fallback=collections.Counter(r['original_s161_reason'] for r in evaluated if r['extension_accepted'])
changed=collections.Counter(r['original_s161_reason'] for r in evaluated if r['s171_entry']!=r['s174_entry'])
conf=sum(metrics['sources'][k]['s174_hits'] for k in ('nexar_blind_confident','s117_interval'))
conf_base=sum(metrics['sources'][k]['s171_hits'] for k in ('nexar_blind_confident','s117_interval'))
poor_horizon=sum(r['result'].get('horizon_n',0)<5 for r in labrows if r['result'].get('horizon_n') is not None)
def table(reasons):
 return '\n'.join(f'| {k} | {v} |' for k,v in sorted(reasons.items(),key=lambda kv:(-kv[1],kv[0])))
analysis=f'''# S174 S161 activation and coverage analysis

S161 is the first entry arm with an official hidden-set gain: team reported S171
Stage2 score .490828857, with 18 more entry hits on 137 hidden clips than S170.
This is the official result for S171, not a measured S174 score.

The public labeled audit used all 243 distinct CCD/Nexar clips and each clip's
exact exported S171 final collision row from S173's paired evaluation. It ran
the exact S171 S161 source and parameters using public cached Faster R-CNN
detections, filling misses with the same detector. It accepted
**{labeled['accepted']}/243** ({labeled['by_kind']['ccd']['accepted']}/217 CCD,
{labeled['by_kind']['nexar']['accepted']}/26 labeled Nexar).
One cached replay (`ccd/001118`) accepted a crossing at a different index from
the exported S171 entry. Exact exported S171 rows remain the paired baseline;
cached-detector replay is a diagnostic proxy, and this discrepancy is retained.

The 750 public Nexar positives were audited separately with final-fit S160
collision anchors reconstructed from retained per-clip feature rows and the
exported localizer. The 659/750 in-sample collision hit count exactly matches
S160's package parity receipt. S161 accepted **{nexar['accepted']}/750**. The
26 labeled Nexar clips overlap this 750; counts must not be added together.
The 750 audit used retained FRCNN boxes where available and decoded the public
MP4 for other sampled frames. Final-fit S160 anchors are in-sample and this is
not a fully exported S171 replay or a new entry-accuracy estimate.

| S161 algorithmic reason, labeled 243 | Clips |
|---|---:|
{table(labeled['reasons'])}

| S161 algorithmic reason, public Nexar 750 | Clips |
|---|---:|
{table(nexar['reasons'])}

The `crossing` reason counts observed box/corridor crossings before the strict
`decide` gate; acceptance also requires a short bracket, at least two outside
and two inside observations, a four-frame track, and entry before collision.
S161 has no explicit `no_lane` reason: it already uses a camera-perspective
ego-corridor instead of painted-line detection. {poor_horizon} labeled clips
with a recorded horizon estimate had fewer than five supporting vehicle boxes
and used the default horizon. `no_anchor`, `short_track`, and
`not_in_lane_at_collision` can reflect a wrong actor, occlusion, or detector
failure; the public labels do not provide actor identity, so a wrong-actor
count cannot be established honestly. First-seen-inside categories cannot
prove a crossing and remain conservative abstentions in S161.

S174 runs only after S161's decision abstains. It considers up to three
distinct vehicle seeds close to the final S160 collision, uses the same
appearance/IoU association with a wider anchor search, tests fixed plausible
camera-corridor width/horizon alternatives, and inspects native frames around
a supported outside-to-inside transition. It never overrides a S161 accepted
result. The optional source append also leaves the original S161 module loaded
if extension initialization fails. It adds no models or dependencies.

S174 accepted an extra **{metrics['activation']}/243** fallback crossings and
changed **{metrics['changed']}/243** entry frames in the paired cached-detector
replay. Added activations by original fallback reason: {dict(fallback)}.
Actual entry changes by original fallback reason: {dict(changed)}.
The replay used exact exported S171 rows as baseline, but S174's candidate
geometry was run directly with cached detector inputs. Exported CUDA QA on
a smaller public panel independently checked protected fields, original frame
IDs, S161 preservation, failure recovery and the shipping archive.

| Label view | n | S171 hit@0.3 | S174 hit@0.3 | Repairs / breaks |
|---|---:|---:|---:|---:|
'''
for key,val in metrics['sources'].items():
 analysis+=f"| {key} | {val['n']} | {val['s171_hits']} | {val['s174_hits']} | {len(val['repairs'])} / {len(val['breaks'])} |\n"
analysis+=f'''
CCD consensus is {metrics['sources']['ccd_consensus']['s171_hits']} ->
{metrics['sources']['ccd_consensus']['s174_hits']}/79; low-confidence CCD A3 is
{metrics['sources']['ccd_labels_A3_low']['s171_hits']} ->
{metrics['sources']['ccd_labels_A3_low']['s174_hits']}/45; the 13 distinct
confident Nexar/S117 clips are {conf_base} -> {conf}/13. Point-label hit@0.3
uses absolute time error <=0.3s; S117 intervals use their endpoints expanded
by 0.3s. Source views overlap and are not pooled as extra clips. CCD and Nexar
entry labels are reused proxies, not official hidden truth.

Added fallback time on 26 native-rate labeled Nexar clips averaged
{metrics['long_mean_added_seconds']:.3f}s/clip; the local 137-clip projection
is {metrics['long_137_seconds']:.1f}s ({metrics['long_137_seconds']/60:.2f}min),
excluding the original S171 workload. This is below the assigned 6-minute
allowance on the measured local mix, not an official server upper bound.
Exported Stage2 paired QA took {qa['seconds']['s171']:.1f}s S171 versus
{qa['seconds']['s174']:.1f}s S174 on its public panel. Offline dependency
installation took {qa['dependency_install_seconds']:.2f}s. The final ZIP is
bound by SHA256 to the staged tree, paired metrics, activation audits and QA.

Limits: fixed corridor geometry can misplace unseen lane markings, a larger
actor search can select a different nearby vehicle, detector boxes approximate
vehicle boundaries rather than tire contact, and cached public detections may
differ from an exported JPEG decode. No hidden S174 gain is claimed.
'''
(H/'ANALYSIS.md').write_text(analysis)
report=H/'WORKER_REPORT.md';history=report.read_text()
marker='\n---\n\n# Retained progress history\n\n'
if marker in history:history=history.split(marker,1)[1]
summary=f'''# S174 coverage arm - delivery

RESULT done - ready for manual selection before 17:00 KST; no upload,
commit or push. Release: `{release['path']}`. SHA256:
`{release['sha256']}`. Bytes: {release['bytes']:,}; {release['members']} members;
official ZIP/unpacked size limits and static validator passed. Exact base
S171 SHA256: `{release['base_sha256']}`. Only S161 predict.py changes among
the original 104 members; one S174 notice is added.

VERIFIED: all 243 labeled and all 750 public Nexar activation audits;
paired hit@0.3 against exact exported S171 rows; actual offline WSL CUDA
Stage2 QA; preserved accepted S161 decisions, protected columns, source frame
IDs, per-clip failure/recovery, native crossing mapping, ZIP member SHA/CRC.
Dependency install {qa['dependency_install_seconds']:.2f}s. Passing QA RSS
{qa['resources']['peak_rss']/2**30:.3f} GiB, CUDA reserved
{qa['cuda_peak_reserved']/2**30:.3f} GiB, free commit minimum
{qa['resources']['min_free_commit_gib']:.2f} GiB. S174 released its lease and
qa.lock on QA exit. The 750 audit yielded at a real gpu.request per clip and
resumed after the request cleared.

Measured S161 activation: {labeled['accepted']}/243 labeled and
{nexar['accepted']}/750 public Nexar. S174 added {metrics['activation']} accepted
fallback crossings and changed {metrics['changed']} entries on labeled clips.
CCD consensus {metrics['sources']['ccd_consensus']['s171_hits']} ->
{metrics['sources']['ccd_consensus']['s174_hits']}/79; CCD A3 low
{metrics['sources']['ccd_labels_A3_low']['s171_hits']} ->
{metrics['sources']['ccd_labels_A3_low']['s174_hits']}/45; confident
{conf_base} -> {conf}/13. Detailed reasons and limitations are in ANALYSIS.md.

RISKS: cached-detector candidate replay is not a full exported 243-clip S174
run; one cached S161 accepted output disagrees with exported S171. The local
137-long-clip added-time projection is {metrics['long_137_seconds']:.1f}s,
not an official runtime guarantee. Actor identity and absent road markings
remain unresolved; no S174 hidden-score gain is claimed.

NEXT: team/manual selection and upload. Keep the officially scored champion
until another official whole-artifact result warrants promotion.
'''
report.write_text(summary+marker+history)
print(release['path'],release['sha256'],'ccd',metrics['sources']['ccd_consensus']['s171_hits'],metrics['sources']['ccd_consensus']['s174_hits'])
