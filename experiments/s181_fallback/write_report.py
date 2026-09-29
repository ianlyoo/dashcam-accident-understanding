"""Rewrite the complete S181 report from analysis, QA and release receipts."""
import json,pathlib

H=pathlib.Path(__file__).resolve().parent
d=json.loads((H/'analysis.json').read_text())
sensitivity=json.loads((H/'sensitivity.json').read_text()) if (H/'sensitivity.json').exists() else None
qa=json.loads((H/'qa.json').read_text()) if (H/'qa.json').exists() else None
release=json.loads((H/'release.json').read_text()) if (H/'release.json').exists() else None
lines=['# S181 S109 fallback entry audit and candidate','',
       'RESULT '+('done' if release else 'partial')+'. '+
       (f"Release `{release['path']}`, SHA256 `{release['sha256']}`." if release else
        'Exact S180 candidate staged; exported CUDA QA/packaging pending.'),
       '',
       '## Abstention and timing audit','',
       'The base is exact S180 (`S180_resid.zip`, SHA256',
       '`d5033adfce1d40fb5fc98289196fb791b274ad4529d111c3d2fbacaa9623a84b`).',
       'On the four lawful entry panels, 91 annotations (89 unique clips) use',
       'S109 after S161/S174/S176 abstain; 68 annotations miss hit@0.3.',
       'CCD consensus has 31 such rows, CCD A3-low 19, confident Nexar 5,',
       'and DKB human 36. The panels overlap by clip ID.',
       '',
       '| Final S161 reason | Rows | S109 hits | Misses | Why no geometric entry |',
       '|---|---:|---:|---:|---|']
notes={'short_track':'Track lacks enough actor observations.',
       'no_anchor':'No usable collision actor was selected.',
       'not_in_lane_at_collision':'Tracked actor is not inside the modeled ego corridor at its collision-end observation.',
       'far_inside_when_first_seen':'Actor was already well inside when first seen; crossing unobserved.',
       'crossing':'Observed crossing failed the strict acceptance gate.'}
for reason,data in d['s161_reasons'].items():
    lines.append(f"| `{reason}` | {data['n']} | {data['hits']} | {data['misses']} | {notes[reason]} |")
lines.extend(['',
              'Among rows with detailed S174 replay, 48 end with',
              '`s174_no_supported_crossing` (34 misses) and 16 with',
              '`s174_no_anchor` (11 misses). The other 27 have exported final',
              'reasons but no retained S174 candidate-reason replay, so their',
              'specific S174 failure cannot be assigned. S176 handles only',
              '`inside_from_start` and `inside_when_first_seen` after S174',
              'abstains; none of these 91 final fallback rows met that rule.',
              'Of the two S161 `crossing` rows, one has only one outside',
              'observation; the other has an overly wide bracket and one',
              'outside observation. Their strict S161 gate correctly abstains.',
              '',
              'The original S161 actor trace is missing in 78/91 fallback',
              'annotations, all outside in six, mixed in five, and all inside',
              'in two. Missing traces usually follow no anchor or short track;',
              'they do not prove that the real entrant never touched the lane.',
              'Public labels do not identify the collision actor, so an actor',
              'association error cannot be counted reliably. In five rows the',
              'collision prediction precedes even the allowed labeled entry',
              'interval: these are collision/entry inconsistencies, potentially',
              'an anchor error or annotation mismatch rather than a timing shift.',
              '',
              'The labeled entry-to-collision gap on these fallback rows has',
              f"median {d['label_to_collision_gap_s']['median']:.2f} s,",
              f"IQR {d['label_to_collision_gap_s']['q25']:.2f}–{d['label_to_collision_gap_s']['q75']:.2f} s;",
              f"among misses it is {d['missed_label_to_collision_gap_s']['median']:.2f} s",
              f"(IQR {d['missed_label_to_collision_gap_s']['q25']:.2f}–{d['missed_label_to_collision_gap_s']['q75']:.2f} s).",
              'Signed here means collision time minus labeled entry midpoint;',
              'negative values place the labeled entry after predicted collision.',
              '',
              '## Paired alternatives and selection','',
              'The median crossing-to-collision gap of 52 *accepted* S161',
              'public clips is 0.85 s (IQR 0.50–1.70 s). This geometric trace',
              'quantity does not use fallback entry labels. Applying collision',
              'minus that gap to every S109 fallback row gives 23->37/91 hits,',
              'with 22 repairs and eight breaks. Restricting it to final',
              '`short_track`, `not_in_lane_at_collision`, and',
              '`far_inside_when_first_seen` retains the same 23->37/91 score,',
              'with 17 repairs, three breaks and 53/91 rows changed. `no_anchor`',
              'has five repairs and five breaks; rejected `crossing` has no net',
              'gain. Both keep S180. This is a collision-relative proxy for the',
              'unseen wheel crossing, not a measured tire contact in each clip.',
              '',
              'The cheaper original-trace first-touch rule changes only four',
              'fallback rows: 23->22/91, zero repairs and one break. It is',
              'rejected. Most fallback rows lack a usable traced boundary',
              'transition, which limits this alternative.',
              '',
              '| Panel | S109 fallback S180 -> S181 hits | Fixed gated gain |',
              '|---|---:|---:|'])
panel_labels={'ccd_consensus_79':'CCD consensus 79',
              'ccd_a3_low_45':'CCD A3-low 45',
              'nexar_confident_13':'Confident Nexar 13',
              'dkb_human_74':'DKB human 74'}
for panel,label in panel_labels.items():
    item=d['gated_prior_panels'][panel]
    lines.append(f"| {label} | {item['before']} -> {item['after']} / {item['n']} | +{item['gain']} |")
lines.extend(['',
              'Leave-one-panel-out removes **all** training labels for a',
              'held-out clip ID. The 0.85 s S161 gap remains the median in all',
              'four folds after excluding held-out IDs. Selecting favorable',
              'S161 reason buckets on the other three panels independently',
              'yields held-out gains +3 CCD consensus, +3 CCD A3-low, +1',
              'Nexar, and +7 DKB. The fixed three-reason policy has identical',
              'held-out gains. S161 gap fitting is label-independent, but',
              'bucket selection uses these same public panels; the small Nexar',
              'sample and mixed AI-made labels limit transfer confidence.',
              '',
              '## Scope, verification, and limits','',
              'The patch changes only Stage2 `entry_frame`, then inherited',
              'collision clamp. It runs after S174/S176 complete and abstain;',
              'S161/S174/S176 accepted decisions and S180 timing remain intact.',
              'No weights, dependencies, or detector passes are added. The',
              'staged tree differs from S180 only in `model/stage2/s161/predict.py`',
              'plus `model/licenses/S181-NOTICE.txt`; `S181_on_S180.patch`',
              'reverse-applies cleanly.'])
if sensitivity:
    positive=sum(v['after']>v['before'] for v in sensitivity['views'].values())
    tied=sum(v['after']==v['before'] for v in sensitivity['views'].values())
    negative=sum(v['after']<v['before'] for v in sensitivity['views'].values())
    lines.extend(['',
                  f"Auxiliary retained AI-label sensitivity spans {sensitivity['selected_clips']}",
                  f"cached fallback clips: {positive} label views improve, {tied} tie,",
                  f"and {negative} decline (`sensitivity.json`). These views overlap",
                  'the primary panels and each other; they are not extra',
                  'independent clips or an additional pooled score.'])
if qa:
    lines.extend(['',f"Offline exported CUDA paired QA passed on {qa['clips']} public clips:",
                  f"{qa['changed']} entry changes and no protected-field changes.",
                  'Source frame IDs, entry<=collision, per-clip failure fallback,',
                  'and same-process recovery passed. Dependency install took',
                  f"{qa['dependency_install_seconds']:.2f} s; peak RSS",
                  f"{qa['resources']['peak_rss']/2**30:.2f} GiB; peak CUDA reservation",
                  f"{qa['cuda_peak_reserved']/2**30:.2f} GiB; minimum free commit",
                  f"{qa['resources']['min_free_commit_gib']:.2f} GiB. No network attempts."])
else:lines.extend(['','Exported CUDA QA pending.'])
if release:
    lines.extend(['',f"ZIP validator passed on `{release['path']}`. All",
                  f"{release['members']} member SHA256s/CRC were checked; ZIP SHA256",
                  f"`{release['sha256']}`."])
else:lines.extend(['','ZIP validator and release SHA256 pending.'])
lines.extend(['',
              'RISKS: The collision-gap rule is an exploratory proxy when the',
              'actor crossing is unobserved. `not_in_lane_at_collision` can',
              'reflect a wrong actor or geometry; public labels cannot separate',
              'them. No hidden or official gain is claimed. CCD and Nexar',
              'entry labels are AI-made, DKB labels human. Most 243-clip',
              'predictions are cached detector replay; exported CUDA QA checks',
              'a smaller selected public set.',
              '',
              'NEXT: Apply the clean S180-based patch to the stack and run',
              'whole-artifact QA before any manual upload decision.'])
(H/'WORKER_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
print('WROTE',H/'WORKER_REPORT.md',len(lines),'lines')
