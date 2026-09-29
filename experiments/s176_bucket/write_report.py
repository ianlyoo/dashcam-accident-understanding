"""Write final human-readable S176 evidence from bound receipts."""
import json
import pathlib

H=pathlib.Path(__file__).resolve().parent
measure=json.loads((H/'measure.json').read_text())
paired=json.loads((H/'paired.json').read_text())
qa=json.loads((H/'qa.json').read_text())
release=json.loads((H/'release.json').read_text())

labels={'inside_from_start':'inside_from_start',
        'inside_when_first_seen':'inside_when_first_seen',
        'far_inside_when_first_seen':'far_inside_when_first_seen',
        'not_in_lane_at_collision':'not_in_lane_at_collision'}
rules={'first_clip':'first clip frame',
       'first_inside':'first observed inside',
       'minus_0p2s':'first inside minus 0.2s',
       'minus_0p5s':'first inside minus 0.5s',
       'extrapolate':'backward lateral extrapolation'}
lines=['# S176 bucket analysis','',
       'All measurements use public labels and exact exported S171 entry/collision rows.',
       'S174 candidate entries come from its cached-detector 243-clip replay.',
       'The S161 trace determines the abstention bucket and candidate first-inside',
       'observation. The primary label is selected once per clip using the fixed',
       'source precedence recorded in `measure.json`; source views overlap and',
       'are reported separately below. These reused labels are proxies, not',
       'official hidden-set results.','',
       '| S161 reason | Total | S174 abstentions eligible |',
       '|---|---:|---:|']
for bucket in labels:
    b=measure['buckets'][bucket]
    lines.append(f"| `{bucket}` | {b['n']} | {b['eligible']} |")
lines += ['',
          '## Eligible abstentions: one primary label per clip','',
          'The table scores the clips S176 can actually change. Repairs and breaks',
          'are relative to S174. The `first clip frame` experiment outside',
          '`inside_from_start` is shown for completeness but is not considered',
          'definition-faithful and was not eligible for selection. A missing',
          'first-inside trace keeps S174 unchanged.','',
          '| Reason | Rule | n | S171 | S174 | Alternative | Repairs | Breaks |',
          '|---|---|---:|---:|---:|---:|---:|---:|']
for bucket in labels:
    b=measure['buckets'][bucket]
    for rule,name in rules.items():
        x=b['eligible_primary'][rule]
        lines.append(f"| `{bucket}` | {name} | {x['n']} | {x['s171']} | {x['s174']} | {x['candidate']} | {x['repairs']} | {x['breaks']} |")
lines += ['',
          '## All clips in each S161 bucket','',
          'This second view includes S174-accepted clips too. Applying an',
          'alternative to those clips is hypothetical; the shipping S176',
          'implementation preserves them. The eligible table above is the',
          'selection evidence.','',
          '| Reason | Rule | n | S171 | S174 | Hypothetical alternative |',
          '|---|---|---:|---:|---:|---:|']
for bucket in labels:
    for rule,name in rules.items():
        x=measure['buckets'][bucket]['all'][rule]
        lines.append(f"| `{bucket}` | {name} | {x['n']} | {x['s171']} | {x['s174']} | {x['candidate']} |")
lines += ['',
          'S176 selects first clip frame only for `inside_from_start` and',
          'first observed inside minus 0.2s only for `inside_when_first_seen`.',
          'On the latter bucket the sole repair is a low-confidence CCD A3',
          'label; no confident-label gain was measured. First-observed-inside',
          'and extrapolation give no net gain for `far_inside_when_first_seen`;',
          'the same alternatives lose one primary hit for',
          '`not_in_lane_at_collision`. Those buckets retain S174.','',
          'The extrapolator fits the first three inside penetration samples',
          'against sampled frame position and projects a positive inward slope',
          'back to zero penetration. If there is no positive trend, it returns',
          'the first observed inside frame. Thus it does not fabricate an',
          'outside observation. Fixed offsets tested were 0.2s and 0.5s.','',
          '## Paired labels on all 243 clips','',
          'The selected S176 rules change 50/243 entries (39 from',
          '`inside_from_start`, 11 from `inside_when_first_seen`) in the trace',
          'replay. Accepted S161/S174 crossings are preserved.','',
          '| Label view | n | S171 | S174 | S176 | Repairs / breaks vs S174 |',
          '|---|---:|---:|---:|---:|---:|']
for source,x in paired['sources'].items():
    lines.append(f"| {source} | {x['n']} | {x['s171']} | {x['s174']} | {x['s176']} | {len(x['repairs'])} / {len(x['breaks'])} |")
conf_base=sum(paired['sources'][x]['s174'] for x in ('nexar_blind_confident','s117_interval'))
conf_new=sum(paired['sources'][x]['s176'] for x in ('nexar_blind_confident','s117_interval'))
lines += ['',
          f"CCD consensus: {paired['sources']['ccd_consensus']['s174']}→{paired['sources']['ccd_consensus']['s176']}/79;",
          f"CCD A3 low: {paired['sources']['ccd_labels_A3_low']['s174']}→{paired['sources']['ccd_labels_A3_low']['s176']}/45;",
          f'13 distinct confident Nexar/S117 clips: {conf_base}→{conf_new}/13.',
          'Point labels use absolute error ≤0.3s; S117 interval endpoints',
          'are expanded by 0.3s. Source views overlap and cannot be summed',
          'as independent clips.','',
          '## Exported package verification','',
          f"S174 ZIP base SHA256: `{release['base_sha256']}`. The S176 patch",
          f"[`S176_on_S174.patch`](S176_on_S174.patch) has SHA256 `{release['patch_sha256']}`.",
          'The patch line endings were normalized to LF and the S176 notice',
          'was included in the integrator patch after exported QA;',
          'the QA and final build receipts verify that every staged package',
          'member is byte-identical before and after that metadata-only fix.',
          'Only `model/stage2/s161/predict.py` changes among S174 members;',
          'S176 adds one notice and no dependency or weight. On S174 accepted',
          'results it returns the exact previous result; on an S174 fault or',
          'S176 rule fault it returns the S174 result for that clip.','',
          f"Offline exported WSL CUDA Stage2 QA used {qa['rows']} public clips:",
          f"{qa['changed']} natural entry changes, identical protected fields,",
          'source frame IDs, collision clamp, single/batch parity, injected',
          'fallback fault, same-process recovery, and forced crossing/bucket',
          'frame mapping. No network attempts.',
          f"S174 {qa['seconds']['s174']:.1f}s; S176 {qa['seconds']['s176']:.1f}s;",
          f"dependency installation {qa['dependency_install_seconds']:.2f}s.",
          f"Peak RSS {qa['resources']['peak_rss']/2**30:.3f} GiB; peak CUDA",
          f"reserved {qa['cuda_peak_reserved']/2**30:.3f} GiB; minimum free",
          f"commit {qa['resources']['min_free_commit_gib']:.2f} GiB.",
          f"The validated ZIP is `{release['path']}` with SHA256",
          f"`{release['sha256']}`. Member SHA/CRC and the submission",
          'validator passed. No S176 hidden score or official server runtime',
          'has been measured. First-frame transfer remains the main risk:',
          'S145 applied it indiscriminately and lost 24 hidden entry hits;',
          'S176 confines it to S161 `inside_from_start` after S174 abstains.',
          'The one known S161 cached-replay/export disagreement in S174',
          'remains a limitation of the paired proxy.']
(H/'ANALYSIS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

report=H/'WORKER_REPORT.md';history=report.read_text(encoding='utf-8')
marker='\n---\n\n# Retained progress history\n\n'
if marker in history:history=history.split(marker,1)[1]
summary=f'''# S176 bucket arm — delivery

RESULT done. Release `{release['path']}`; SHA256
`{release['sha256']}`; {release['bytes']:,} bytes; validator and member SHA/CRC
passed. Exact base S174 SHA256 `{release['base_sha256']}`. Patch for integrator:
`{release['patch_path']}` (SHA256 `{release['patch_sha256']}`). No upload,
commit, or push.

CHANGED: only Stage2 entry_frame can change, then the inherited collision
clamp. Selected rules are first clip frame for S161 `inside_from_start` and
first observed inside minus 0.2s for `inside_when_first_seen`, only after
S174 abstains. Other buckets and accepted S161/S174 results remain unchanged.

VERIFIED: 243 public paired labels; `inside_from_start` eligible primary hits
6→28/44 (26 repairs, 4 breaks); `inside_when_first_seen` 0→1/12;
zero-gain buckets excluded. Full CCD consensus {paired['sources']['ccd_consensus']['s174']}→
{paired['sources']['ccd_consensus']['s176']}/79, CCD A3 low
{paired['sources']['ccd_labels_A3_low']['s174']}→{paired['sources']['ccd_labels_A3_low']['s176']}/45,
confident 13 {conf_base}→{conf_new}. Exported offline CUDA QA passed
({qa['rows']} clips, {qa['changed']} natural changes), dependency install
{qa['dependency_install_seconds']:.2f}s, peak RSS
{qa['resources']['peak_rss']/2**30:.3f} GiB and CUDA reserved
{qa['cuda_peak_reserved']/2**30:.3f} GiB. S176's GPU lease and qa.lock released.

RISKS: paired 243-clip result is a cached-detector trace replay, not a full
exported run. The 0.2s rule gains only one low-confidence label, and
first-frame entry has prior hidden-set failure when used globally. No S176
official score is claimed. Full per-bucket counts and limitations are in
ANALYSIS.md.

NEXT: integrator can apply the clean S174 patch onto the S172-based stack;
team decides manual upload. Retain the officially scored S172 champion until
a better official whole-artifact result exists.
'''
report.write_text(summary+marker+history,encoding='utf-8')
print(release['path'],release['sha256'],paired['changed'],qa['changed'])
