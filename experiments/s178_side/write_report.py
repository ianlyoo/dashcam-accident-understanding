"""Write the final S178 worker report from measured receipts."""
import json,pathlib

H=pathlib.Path(__file__).resolve().parent
release=json.loads((H/'release.json').read_text());qa=json.loads((H/'qa.json').read_text())
study=json.loads((H/'study.json').read_text());human=json.loads((H/'human.json').read_text())
def fmt(v):return f'{v:.3f}'
def pair(a,b):return fmt(a)+'->'+fmt(b)
panels=[
 ('CCD consensus','ccd_consensus_side'),
 ('CCD A1 confident','ccd_A1_confident_side'),
 ('CCD A1 low','ccd_A1_low_side'),
 ('CCD A2 confident','ccd_A2_confident_side'),
 ('CCD A2 low','ccd_A2_low_side'),
 ('CCD A3 confident','ccd_A3_confident_side'),
 ('CCD A3 low','ccd_A3_low_side'),
 ('Nexar confident','nexar_confident_side'),
 ('Nexar low','nexar_low_side'),
]
table=['| Labeled panel | n | LEFT F1 | RIGHT F1 | Macro-F1 | Repairs / breaks |',
       '|---|---:|---:|---:|---:|---:|']
for label,name in panels:
    result=study['side'][name]['s161_center_0.15']
    if result['measured']==0:
        table.append(f'| {label} | 0 | - | - | - | - |');continue
    left=result['per_class']['LEFT'];right=result['per_class']['RIGHT']
    table.append(f"| {label} | {result['measured']} | {pair(left['incumbent_f1'],left['candidate_f1'])} | "
                 f"{pair(right['incumbent_f1'],right['candidate_f1'])} | "
                 f"{pair(result['incumbent_f1'],result['candidate_f1'])} | "
                 f"{len(result['repairs'])} / {len(result['breaks'])} |")
for label,name in (('DKB human','dkb'),('Jungmin human','jungmin')):
    result=human[name];left=result['per_class']['LEFT'];right=result['per_class']['RIGHT']
    table.append(f"| {label} | {result['n']} | {pair(left['incumbent_f1'],left['candidate_f1'])} | "
                 f"{pair(right['incumbent_f1'],right['candidate_f1'])} | "
                 f"{pair(result['incumbent_f1'],result['candidate_f1'])} | "
                 f"{len(result['repairs'])} / {len(result['breaks'])} |")
evasion=['| Evasion panel | n | Incumbent -> clearance proxy macro-F1 | Repairs / breaks |',
         '|---|---:|---:|---:|']
for label,name in (('CCD consensus','ccd_consensus_evasion'),
                   ('CCD A1 confident','ccd_A1_confident_evasion'),
                   ('CCD A2 confident','ccd_A2_confident_evasion'),
                   ('CCD A3 confident','ccd_A3_confident_evasion'),
                   ('Nexar confident','nexar_confident_evasion'),
                   ('DKB human overlap','dkb_human_evasion'),
                   ('Jungmin human overlap','jungmin_human_evasion')):
    r=study['evasion'][name]['evasion']
    evasion.append(f"| {label} | {r['measured']} | {pair(r['incumbent_f1'],r['candidate_f1'])} | "
                   f"{len(r['repairs'])} / {len(r['breaks'])} |")
previous=(H/'WORKER_REPORT.md').read_text(encoding='utf-8')
history=previous[previous.index('## 11:25 KST setup'):]
report=f'''# S178 geometric side arm — delivery

RESULT done. Release `$DATA_DIR/releases/S178_side.zip`, SHA256
`{release['sha256']}`, {release['bytes']:,} bytes. Exact S176 base SHA256
`{release['base_sha256']}`. Clean integrator patch
`candidates/s178_side/S178_on_S176.patch`, SHA256 `{release['patch_sha256']}`.
No upload, commit or push.

CHANGED: only `entry_side` may change. A supported original S161 crossing uses
the last observed exterior ego-corridor boundary side. The two accepted S176
already-inside buckets use the collision actor's horizontal center only if it
is <=0.35 or >=0.65 of image width. S174-only crossings and every ambiguous
case retain S176 side. `collision_frame`, `entry_frame`, `evasion_space`, and
all other output fields remain S176; no new model weights or dependencies.

VERIFIED: actual offline exported WSL CUDA QA on {qa['rows']} public clips,
including all 76 previously uncovered human-labeled CCD clips and a paired
S176/S178 natural-change panel. {qa['changed']} side outputs changed. The
paired export preserved every other output field; original source-frame IDs,
entry/collision clamp, a per-clip fault and same-process recovery passed.
Dependency install {qa['dependency_install_seconds']:.2f}s, peak RSS
{qa['resources']['peak_rss']/2**30:.3f} GiB, CUDA reserved
{qa['cuda_peak_reserved']/2**30:.3f} GiB, minimum free commit
{qa['resources']['min_free_commit_gib']:.2f} GiB. No network inference attempt.
The 107-member ZIP passed static validator, member SHA256/CRC, size limits and
forward/reverse patch checks. S178's GPU lease and qa.lock were released.

Official side means screen-relative LEFT/RIGHT approach of the other vehicle,
not road bearing; `src/videohackathon/evaluation.py` scores fixed-class
LEFT/RIGHT macro-F1. Official clarification:
https://dacon.io/competitions/official/236753/talkboard/417186 and
https://dacon.io/competitions/official/236753/talkboard/417202.

## Paired side F1 by labeled source and class

Values are incumbent S172 side -> S178 side. The CCD and Nexar primary panels
are AI-made, not ground truth. Individual CCD annotator views overlap the CCD
consensus and each other. DKB and Jungmin are independent human annotations
that also overlap each other. DKB {human['dkb']['exported']}/74 and Jungmin
{human['jungmin']['exported']}/36 use actual exported S178 outputs; their
remaining cases use the cached S161/S174 trace replay against exact stored
incumbent sides. Selected exported/cached parity differences: 
{len(human['selected_export_parity_mismatches'])}; details in `human.json`.

{chr(10).join(table)}

The five official sample clips have no usable side truth (`-1`). SAMA provides
lane geometry rather than a crash entrant's screen-relative side. Hangi's
public Nexar review drafts mark themselves evaluation-ineligible and do not
promote their provisional side observations to ground truth; none is silently
counted as a labeled side panel.

The selected S161+bucket policy has {study['coverage']['s161_crossing']}
original S161 accepted crossings among 243 public labeled clips. S174 adds
{study['coverage']['s174_crossing']} crossings, but using their side lowered CCD
consensus F1 from {fmt(study['side']['ccd_consensus_side']['s161_only']['candidate_f1'])}
to {fmt(study['side']['ccd_consensus_side']['cross_only']['candidate_f1'])} and
CCD A2-confident from {fmt(study['side']['ccd_A2_confident_side']['s161_only']['candidate_f1'])}
to {fmt(study['side']['ccd_A2_confident_side']['cross_only']['candidate_f1'])};
S174 side stays incumbent. The bucket anchor-center rule adds only when
unambiguous: CCD consensus improves from
{fmt(study['side']['ccd_consensus_side']['s161_only']['candidate_f1'])} to
{fmt(study['side']['ccd_consensus_side']['s161_center_0.15']['candidate_f1'])};
Nexar confident is unchanged by that addition.

## Evasion signal checked, not shipped

The older tracker clearance prediction was tested on accepted S161 crossings
and aligned S176 buckets. Macro-F1 results (incumbent -> proxy) are:

{chr(10).join(evasion)}

CCD consensus has six repairs and seven breaks despite
a small class-balance F1 gain; confident Nexar has no changed outputs.
Evasion is a physical usable-passage judgment at collision, which the single
actor trace does not settle. S178 leaves `evasion_space` untouched.

RISKS: no S178 hidden score is claimed. The main 243-clip comparison is a
cached-detector replay; human extra clips were actually exported, while other
human rows retain that replay limitation. S172's visual collision changes the
anchor on long clips when the integrator stacks this patch, so the stacked
artifact needs its own output-isolation QA. A prior broad tracker side arm
lost hidden score even after local AI-label gains; the new rule is narrower
and grounded in S161's crossing, but hidden transfer remains unproven.

NEXT: integrator can apply the clean S176-based patch to its S172/S175 stack,
run stacked QA, and decide the manual upload. Keep the officially scored
champion until a better official whole-artifact result is obtained.

---

# Retained progress history

{history}'''
(H/'WORKER_REPORT.md').write_text(report,encoding='utf-8')
print('wrote',H/'WORKER_REPORT.md')
