"""Make a read-only-logic probe of the phase6 QA with persisted mismatch rows."""
from pathlib import Path

here=Path(__file__).resolve().parent
source=(here.parents[1]/'candidates/s167_stack/phase6/qa.py').read_text()
original="H = Path(__file__).resolve().parent\nREPO = H.parents[2]"
replacement="H = Path(__file__).resolve().parents[2]/'candidates/s167_stack/phase6'\nREPO = H.parents[2]"
assert source.count(original)==1
source=source.replace(original,replacement)
marker="REQUEST = Path('$GPU_REQUEST_PATH')"
assert source.count(marker)==1
source=source.replace(marker,marker+"\nos.environ['S171_ADMISSION_FREE_COMMIT_GIB'] = str(json.loads((D/'s181_fallback/host_commit.json').read_text())['free_commit_gib'])")
original="output = W/'qa'"
replacement="output = W/'qa_diagnose'"
assert source.count(original)==1
source=source.replace(original,replacement)
original="    # Preserve the working S164/S172 clipping and resource checkpoints."
replacement="""    # Persist the exact S177 base consumed by the normal S182 pass.
    original_s177_base = ns['_S182_BASE_STAGE2']
    def capture_s177(*a, **kw):
        live = original_s177_base(*a, **kw)
        live.to_csv(output/'live_s177.csv', index=False, lineterminator='\\n')
        return live
    ns['_S182_BASE_STAGE2'] = capture_s177

    # Preserve the working S164/S172 clipping and resource checkpoints."""
assert source.count(original)==1
source=source.replace(original,replacement)
original="        failed = artifact.predict_stage2(inputs, ROOT/'model/stage2')\n"
replacement="""        failed = artifact.predict_stage2(inputs, ROOT/'model/stage2')
        failed.to_csv(output/'faulted.csv', index=False, lineterminator='\\n')
        frame.to_csv(output/'normal_again.csv', index=False, lineterminator='\\n')
        comparison = {
            'same_values_numpy': failed.iloc[1:].to_numpy().tolist() == frame.iloc[1:].to_numpy().tolist(),
            'same_values_json': json.loads(failed.iloc[1:].to_json(orient='records')) == json.loads(frame.iloc[1:].to_json(orient='records')),
            'normal_dtypes': {k: str(v) for k, v in frame.dtypes.items()},
            'fault_dtypes': {k: str(v) for k, v in failed.dtypes.items()},
            'fault_diagnostics': ns['_S182_DIAGNOSTICS'],
        }
        (output/'comparison.json').write_text(json.dumps(comparison, indent=2)+'\\n')
"""
assert source.count(original)==1
source=source.replace(original,replacement)
dest=here/'diagnose_integrator.py'
dest.write_text(source)
compile(source,str(dest),'exec')
print('WROTE',dest)
