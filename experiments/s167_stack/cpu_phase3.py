"""Actual S161 adapter CPU wiring tests using S170-shaped synthetic base rows."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as N
from unittest.mock import patch
import sys
import pandas as pd
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from candidates.s167_stack.build_stack import sha
WORK=Path('$DATA_DIR/s167_stack/phase3')

def main():
    root=WORK/'candidate'
    spec=importlib.util.spec_from_file_location('_s171_cpu_adapter',root/'model/stage2/s118/adapter.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    rule=json.loads((root/'model/stage2/s118/rule.json').read_text());mod.validate_rule(rule)
    bad=copy.deepcopy(rule);bad['entry_frame']['side']=True
    try: mod.validate_rule(bad)
    except ValueError: pass
    else: raise AssertionError('Mode protection lost')
    base=pd.DataFrame([dict(ID='A',collision_frame=60,entry_frame=50,entry_side='LEFT',evasion_space=0),dict(ID='B',collision_frame=80,entry_frame=70,entry_side='RIGHT',evasion_space=1)])
    ns={'_s008_frame_paths':lambda p:[p/f'{i}.jpg' for i in range(100)],'_s008_frame_number':lambda p:int(p.stem)}
    state={'mode':'healthy'};anchors=[]
    class Tracker:
        device='cpu'
        def __init__(self,*args):
            if state['mode']=='load': raise OSError('CPU injected tracker load failure')
        def analyze(self,paths,numbers,anchor):
            ident=paths[0].parent.name;anchors.append((ident,anchor))
            if state['mode']=='clip' and ident=='B':raise RuntimeError('CPU injected clip failure')
            return {'entry_index':90 if ident=='A' else 65,'reason':'synthetic_crossing'}
        def close(self):pass
    package=N(Tracker=Tracker,decide=lambda result,fallback:(result['entry_index'],None))
    with patch.object(mod,'load_package',return_value=package):
        for mode in ('healthy','clip','load','healthy'):
            state['mode']=mode;diag={'rule_error':None,'clips':{}}
            frame=mod._predict_track(ns,WORK/'cpu_inputs',root/'model/stage2',rule,base_call=lambda:base.copy(),diagnostics=diag)
            expected={'healthy':[60,65],'clip':[60,70],'load':[50,70]}[mode]
            assert frame.entry_frame.tolist()==expected,(mode,frame.to_dict('records'),diag)
            assert frame.drop(columns='entry_frame').equals(base.drop(columns='entry_frame'))
        assert anchors and all(a in (60,80) for _,a in anchors)
    proof=dict(passed=True,scope='CPU synthetic detector results; actual merged S161 adapter',mode_fix_passed=True,
        clamp_after_entry_passed=True,entry_only_passed=True,load_fallback_s170=True,clip_fallback_s170=True,recovery_passed=True,
        build_sha256=sha(WORK/'build.json'),adapter_sha256=sha(root/'model/stage2/s118/adapter.py'),source_sha256=sha(__file__))
    (WORK/'cpu.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof),flush=True)

if __name__=='__main__':main()
