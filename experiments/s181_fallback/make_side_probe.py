"""Derive a focused S178-side order probe from the phase6 diagnostic harness."""
from pathlib import Path

here=Path(__file__).resolve().parent
s=(here/'diagnose_phase6.py').read_text()
replacements={
    "W=D/'s181_fallback/diagnosis'":"W=D/'s181_fallback/diagnosis_side'",
    "INPUT=REPO/'Baseline/sample_evaluation_data/stage2'":"INPUT=D/'s176_bucket/qa_inputs'",
    "saved=pd.read_csv(D/'s167_stack/phase5/qa/stage2_public.csv',dtype={'ID':str})":
        "saved=pd.read_csv(D/'s167_stack/phase5/qa/stage2_bucket.csv',dtype={'ID':str})\n"
        "saved=saved[saved.ID.isin(('SAMPLE_S2_003','ccd_000080','ccd_000224'))].reset_index(drop=True)\n"
        "assert len(saved)==3",
    "    source=ns['_S182_BASE_STAGE2'];capture=[]\n"
    "    def capture_base(*a,**kw):\n"
    "        out=source(*a,**kw);capture.append(out.copy());save('captured_base',out);return out\n"
    "    ns['_S182_BASE_STAGE2']=capture_base\n"
    "    normal=artifact.predict_stage2(INPUT,ROOT/'model/stage2')\n"
    "    assert len(capture)==1\n"
    "    live=capture[0];save('normal',normal)":
        "    live=saved.copy();save('captured_base',live)\n"
        "    ns['_S182_BASE_STAGE2']=lambda *a,**kw:live.copy()\n"
        "    normal=artifact.predict_stage2(INPUT,ROOT/'model/stage2');save('normal',normal)",
}
for old,new in replacements.items():
    assert s.count(old)==1,old
    s=s.replace(old,new)
dest=here/'diagnose_side.py'
dest.write_text(s)
compile(s,str(dest),'exec')
print('WROTE',dest)
