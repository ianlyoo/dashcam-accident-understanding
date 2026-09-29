"""Read-only inventory of public detector and collision artifacts for S174."""
import json,pathlib,pickle,zipfile,collections
import numpy as np
D=pathlib.Path('$DATA_DIR')
def shape(name,obj):
    if isinstance(obj,dict):
        print(name,'dict',len(obj),'keys',list(obj)[:12])
        if obj:
            key=next(iter(obj));value=obj[key]
            print(name,'first',repr(key),'value_type',type(value).__name__,
                  'value_keys',list(value)[:12] if isinstance(value,dict) else '',
                  'value_len',len(value) if hasattr(value,'__len__') else '')
    elif isinstance(obj,list):
        print(name,'list',len(obj),'first',str(obj[0])[:500] if obj else '')
    else:print(name,type(obj).__name__,str(obj)[:200])
for rel in ('s142_rr2/cache/nexar_00060.json','s160_coll_loc/final_parity.json','s160_coll_loc/names.json','s160_coll_loc/integration_E10_base.json'):
    p=D/rel
    if p.exists():shape(rel,json.loads(p.read_text()))
sample=json.loads((D/'s142_rr2/cache/nexar_00060.json').read_text())
print('S142 all keys',list(sample))
for k in ('detected_frames','candidates','baseline_index','box_names'):
    shape('S142 '+k,sample[k])
parity=json.loads((D/'s160_coll_loc/final_parity.json').read_text())
print('S160 parity scalar fields',{k:v for k,v in parity.items() if not isinstance(v,(list,dict))})
with np.load(D/'s160_coll_loc/rows/00000.npz') as rows:
    print('S160 row npz',[(k,rows[k].shape,str(rows[k].dtype)) for k in rows.files])
for rel in ('s132_s2track/cache_nexar_frcnn_640_0.3.pkl','s132_s2track/cache_ccd_frcnn_640_0.3.pkl'):
    p=D/rel
    if p.exists():
        with p.open('rb') as f: obj=pickle.load(f)
        shape(rel,obj)
with zipfile.ZipFile(D/'releases/S171_ent.zip') as z:
    print('S171 members',len(z.namelist()),'size',sum(i.file_size for i in z.infolist()))
    rule=json.loads(z.read('model/stage2/s118/rule.json'))
    shape('S171 rule',rule)
