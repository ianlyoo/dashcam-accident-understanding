"""Write the complete S180 worker report from frozen study and QA receipts."""
import json,pathlib

H=pathlib.Path(__file__).resolve().parent
analysis=json.loads((H/'analysis.json').read_text())
qa=json.loads((H/'qa.json').read_text())
release_file=H/'release.json'
release=json.loads(release_file.read_text()) if release_file.exists() else None
panels=(('ccd_consensus_79','CCD consensus 79'),('ccd_a3_low_45','CCD A3-low 45'),
        ('nexar_confident_13','Nexar confident 13'),('dkb_human_74','DKB human 74'))
sources=(('s161_crossing','S161 crossing'),('s174_crossing','S174 crossing'),
         ('s176_first_frame','S176 first frame'),
         ('s176_first_obs_minus_0p2','S176 first observation minus 0.2 s'),
         ('s109_fallback','S109 fallback'))
lines=['# S180 residual entry timing','',
       'RESULT '+('done' if release else 'partial')+'. '+
       ('Release `'+release['path']+'`, SHA256 `'+release['sha256']+'`.' if release else
        'Exact S178 candidate staged; packaging pending.'),
       '',
       '## Evidence and decision','',
       'The base is exact S178 (`S178_side.zip`, SHA256',
       '`bab785c758b82f7524bdaa1153220c3aa61a7d8b120ae5cac14c049ec83ec047`).',
       'The only inference change is +0.2 s for accepted S174-only geometric',
       'crossings, clamped to collision. All other entry sources and every',
       'collision, side, evasion, and Stage3 output remain S178.',
       '',
       'The study covers 211 lawful annotations: CCD consensus 79, CCD A3-low 45,',
       'confident Nexar 13, and DKB human 74. They are annotation counts rather',
       'than disjoint clips; six IDs overlap DKB and CCD. Signed error is predicted',
       'time minus the annotated interval midpoint in seconds. Hit@0.3 means',
       'the prediction lies within the full interval extended by 0.3 s.',
       '',
       '| Decision source | n | S178 hits | Mean signed s | Median signed s | Early / late misses |',
       '|---|---:|---:|---:|---:|---:|']
for source,label in sources:
    data=analysis['pooled'][source]
    lines.append(f"| {label} | {data['n']} | {data['hits']} | {data['mean_s']:+.3f} | {data['median_s']:+.3f} | {data['early_misses']} / {data['late_misses']} |")
lines.extend(['',
              'Means are sensitive to long Nexar outliers. The S174 error is',
              'heterogeneous rather than a global offset: its near-boundary early',
              'cases benefit from a small delay, while distant late misses remain.',
              '',
              '| Panel | S161 | S174 | S176 first | S176 obs-0.2 | S109 | Total S178 -> S180 |',
              '|---|---:|---:|---:|---:|---:|---:|'])
for panel,label in panels:
    values=[analysis['baseline'][panel][source] for source,_ in sources]
    before=sum(v['hits'] for v in values)
    gain=analysis['scan']['s174_crossing']['0.2']['panels'][panel]['gain']
    cells=[f"{v['hits']}/{v['n']}" for v in values]
    lines.append('| '+label+' | '+' | '.join(cells)+f' | {before} -> {before+gain} |')
lines.extend(['',
              'S174 +0.2 s repairs five labels (CCD `001115`, `001273`,',
              '`001427`; DKB `000021`, `000091`) and breaks none. +0.1 s repairs',
              'three; +0.3 s repairs six, both without paired losses. +0.2 s is',
              'the smaller gain on two panels and the shift selected when DKB',
              'is held out. A tracked vehicle rectangle can touch the corridor',
              'edge a few frames before its wheel/tire reaches it, so a two-frame',
              'delay at 10 fps is physically plausible. This remains a proxy.',
              '',
              '## Leave-one-panel-out and rejected shifts','',
              'Each fit excludes every training label whose clip ID occurs in',
              'the held-out panel. S174 selects +0.3 s holding out CCD consensus,',
              'CCD A3-low, or Nexar, and +0.2 s holding out DKB. Held-out S174',
              'hits are 1->1, 2->5, 0->0, 5->7 respectively. The shipped fixed',
              '+0.2 s has the same held-out hits on all four panels; no fold loses.',
              'This is limited evidence: only one confident Nexar annotation is',
              'an S174 crossing, and the CCD and Nexar labels are AI-made.',
              '',
              'S161 has median error zero; nonzero shifts tie with both repairs',
              'and breaks or lose pooled hits. S176 first-frame +0.1 s gains one',
              'pooled label but no held-out hit and delays a known already-inside',
              'frame-zero case. S176 first-observation-minus-0.2 s gains nothing',
              'within +/-0.3 s; eight misses are late, often by seconds. S109',
              'fallback has 68/91 misses with balanced early/late errors; small',
              'shifts have zero or negative pooled gain. These remain S178.',
              '',
              'The largest remaining source is S109 fallback (68 misses). Its',
              'largest identified reason is `s174_no_supported_crossing` (34',
              'misses / 48 rows); 15 of those misses have original S161 reason',
              '`short_track`. A concrete next fix is collision-anchored wheel',
              'tracking through dense nearby frames to recover a measured',
              'outside-to-inside transition when coarse tracking lacks two',
              'observations on each side; abstain without a measured crossing.',
              '',
              '## Verification and limits','',
              f"Exported offline CUDA QA passed on {qa['clips']} public CCD clips:",
              f"{qa['changed']} entry changes, no protected-field changes,",
              'correct source-frame IDs and collision clamp, per-clip failure',
              'fallback and same-process recovery. Dependency install took',
              f"{qa['dependency_install_seconds']:.2f} s; peak RSS was",
              f"{qa['resources']['peak_rss']/2**30:.2f} GiB, CUDA reserved",
              f"{qa['cuda_peak_reserved']/2**30:.2f} GiB, and minimum free commit",
              f"{qa['resources']['min_free_commit_gib']:.2f} GiB. No network attempt.",
              'The nine QA clips include five predicted S174 repairs and controls',
              'for the unchanged S161, S176, and S109 paths. S178 exported QA',
              'provided 84 additional clip outputs for analysis; three missing',
              'DKB clips were exported on exact S178. Eight overlaps with cached',
              'tracker replay agree in entry and source. Most of the 243-clip',
              'labeled replay is cached rather than newly exported. No private',
              'evaluation data and no official/hidden gain are claimed.',
              '',
              'Full rows, per-panel scans and overlap exclusions are in',
              '`rows.jsonl` and `analysis.json`; fallback reasons are in',
              '`fallback_buckets.json`. The clean patch is `S180_on_S178.patch`.'])
if release:
    lines.extend(['',
                  f"ZIP validator passed ({release['validator_exit']}); member CRC/SHA256",
                  f"verification passed for {release['members']} members. Release SHA256:",
                  f"`{release['sha256']}`. The ZIP is a fallback candidate; the",
                  'integrator must validate its own stacked whole artifact.'])
else:
    lines.extend(['','ZIP validator, release SHA256, and packaging pending.'])
lines.extend(['','RISKS: Local labels are partly AI-made and the S174 sample is',
              'small. A +0.2 s shift has no demonstrated hidden-set improvement.',
              '',
              'NEXT: Integrator may apply the clean S178-based patch to the stacked',
              'artifact and run whole-artifact QA before considering manual upload.'])
(H/'WORKER_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
print('WROTE',H/'WORKER_REPORT.md',len(lines),'lines')
