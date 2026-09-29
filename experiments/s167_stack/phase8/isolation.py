"""Assert S185 versus S182 differs only where S176 buckets fired."""
import json
import argparse
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
D = Path('$DATA_DIR')
P6 = D/'s167_stack/phase6/qa'
P8 = D/'s167_stack/phase8/qa'
ALL_PANELS = ('public', 'bucket', 'long', 'cascade', 'entry')
BUCKETS = {'s176_inside_from_start', 's176_inside_first_minus_0p2s'}
parser = argparse.ArgumentParser()
parser.add_argument('--panel', choices=ALL_PANELS)
args = parser.parse_args()
panels = (args.panel,) if args.panel else ALL_PANELS
summary = {}

for panel in panels:
    reference = (D/'s167_stack/phase5/qa/stage2_bucket_baseline.csv'
                 if panel == 'bucket' else
                 D/'s167_stack/phase4/qa'/f'stage2_s175_{panel}.csv')
    base = pd.read_csv(reference, dtype={'ID': str}).to_dict('records')
    s182 = pd.read_csv(P6/f'stage2_{panel}.csv', dtype={'ID': str}).to_dict('records')
    s185 = pd.read_csv(P8/f'stage2_{panel}.csv', dtype={'ID': str}).to_dict('records')
    reasons = [row['reason'] for row in json.loads((P6/f'{panel}.json').read_text())['replay']]
    assert len(base) == len(s182) == len(s185) == len(reasons), panel
    bucket_rows = []
    for old, before, after, reason in zip(base, s182, s185, reasons):
        assert old['ID'] == before['ID'] == after['ID'], panel
        expected = old if reason in BUCKETS else before
        assert after == expected, (panel, old['ID'], reason, after, expected)
        if reason in BUCKETS:
            bucket_rows.append(dict(ID=old['ID'], reason=reason,
                                    s175_entry=old['entry_frame'],
                                    s182_entry=before['entry_frame'],
                                    s175_side=old['entry_side'],
                                    s182_side=before['entry_side']))
    summary[panel] = dict(rows=len(base), bucket_rows=bucket_rows,
                          nonbucket_exact_s182=len(base)-len(bucket_rows))

if not args.panel:
    assert any(x['bucket_rows'] for x in summary.values())
receipt = dict(passed=True, comparison='S185=S175 on S176 bucket rows; S185=S182 elsewhere',
               panels=summary)
output = HERE/('isolation_'+args.panel+'.json' if args.panel else 'isolation.json')
output.write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps(dict(passed=True, bucket_rows=sum(len(x['bucket_rows'])
                                                    for x in summary.values()))))
