"""S172 exported per-file visual collision update after the complete S171 row."""
import importlib.util
from pathlib import Path
import time
import numpy as np

OWN_LIMIT_SECONDS=600
PROCESS_LIMIT_SECONDS=45*60

class TimeBudgetExceeded(RuntimeError):
    pass

def load_source(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

class VisualCollision:
    def __init__(self,root):
        import json
        import torch
        from transformers import Dinov2Model
        self.root=Path(root)
        self.check_budget=lambda:None
        self.head_module=load_source('_s172_head',self.root/'head.py')
        # refinement.py imports head; bind only for the module load and restore.
        import sys
        previous=sys.modules.get('head')
        sys.modules['head']=self.head_module
        try: self.refine=load_source('_s172_refinement',self.root/'refinement.py')
        finally:
            if previous is None: sys.modules.pop('head',None)
            else: sys.modules['head']=previous
        self.cfg=json.loads((self.root/'config.json').read_text())
        if self.cfg['format']!='s172_visual_v1': raise ValueError('Unexpected S172 format')
        if not torch.cuda.is_available():
            raise RuntimeError('S172 visual arm requires CUDA; preserve S171')
        self.device=torch.device('cuda')
        self.models=[]
        for filename in self.cfg['heads']:
            record=torch.load(self.root/filename,map_location='cpu',weights_only=True)
            model=self.head_module.TemporalHead(record['hidden'],drop=0)
            model.load_state_dict(record['state'],strict=True)
            self.models.append(model.eval().requires_grad_(False).to(self.device))
        backbone,loading=Dinov2Model.from_pretrained(str(self.root/'backbone'),local_files_only=True,
            trust_remote_code=False,torch_dtype=torch.float16 if self.device.type=='cuda' else torch.float32,
            attn_implementation='sdpa',output_loading_info=True)
        if any(loading.get(key) for key in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
            raise ValueError('Incomplete DINOv2-L checkpoint')
        if backbone.config.hidden_size!=1024 or backbone.config.num_hidden_layers!=24:
            raise ValueError('Unexpected DINOv2-L architecture')
        self.backbone=backbone.eval().requires_grad_(False).to(self.device)

    def embed(self,paths,frames):
        import cv2
        import torch
        mean=np.array([.485,.456,.406],np.float32)
        std=np.array([.229,.224,.225],np.float32)
        out=[]
        batch=int(self.cfg['batch'])
        if batch not in (4,8,16): raise ValueError('Unexpected S172 batch size')
        with torch.inference_mode():
            for start in range(0,len(frames),batch):
                self.check_budget()
                images=[]
                for index in frames[start:start+batch]:
                    im=cv2.imread(str(paths[int(index)]),cv2.IMREAD_COLOR)
                    if im is None: raise ValueError(f'Unreadable frame {index}')
                    im=cv2.cvtColor(cv2.resize(im,(224,224),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB)
                    images.append(((im.astype(np.float32)/255-mean)/std).transpose(2,0,1))
                x=torch.from_numpy(np.stack(images)).to(self.device,dtype=self.backbone.dtype)
                with torch.autocast(device_type='cuda',dtype=torch.float16):
                    h=self.backbone(pixel_values=x).last_hidden_state
                out.append(torch.cat([h[:,0],h[:,1:].mean(1)],1).float().cpu().numpy())
                self.check_budget()
        result=np.concatenate(out)
        if result.shape!=(len(frames),2048) or not np.isfinite(result).all():
            raise ValueError('Invalid DINO features')
        return result

    def position(self,paths,prior):
        import torch
        n=len(paths); fps=30.
        bf=np.asarray(prior['frames'],np.int32); bp=np.asarray(prior['probs'],np.float64)
        if bf.ndim!=1 or bp.ndim!=1 or len(bf)!=len(bp) or not len(bf) or bf.min()<0 or bf.max()>=n or not np.isfinite(bp).all() or (bp<0).any() or bp.sum()<=0:
            raise ValueError('Invalid S160 prior')
        full=np.unique(np.minimum(np.rint(np.arange(0,n/fps,.25)*fps).astype(np.int32),n-1))
        rate=self.cfg['coarse_hz']
        if rate==2: frames=full[::2]
        elif rate==2.5: frames=full[[i%8 in (0,2,3,5,6) for i in range(len(full))]]
        elif rate==3: frames=full[[i%4 in (0,1,2) for i in range(len(full))]]
        elif rate==4: frames=full
        else: raise ValueError('Unverified coarse rate')
        features=self.embed(paths,frames)
        with torch.inference_mode():
            x=torch.from_numpy(features.astype(np.float32))[None].to(self.device)
            logits=torch.stack([model(x)[0] for model in self.models]).mean(0).cpu().numpy()
        kind,weight=self.cfg['kind'],float(self.cfg['weight'])
        probability=self.head_module.dense_fusion(logits,frames,bf,bp,n,kind,weight)
        fine_count=0
        if self.cfg['refine']:
            schedule=self.cfg.get('coarse_schedule',dict(kind=kind,weight=weight))
            schedule_probability=self.head_module.dense_fusion(logits,frames,bf,bp,n,schedule['kind'],float(schedule['weight']))
            fine=self.refine.leading_frames(schedule_probability,fps)
            extra=self.embed(paths,fine)
            fine_logits=self.refine.refine_logits(self.models,frames,features,fine,extra,fps,self.device)
            merged,values=self.refine.merge_logits(frames,logits,fine,fine_logits)
            probability=self.head_module.dense_fusion(values,merged,bf,bp,n,kind,weight)
            fine_count=len(fine)
        pick=self.head_module.decode(probability,int(self.cfg['half']))
        if not np.isfinite(probability).all() or not 0<=pick<n:
            raise ValueError('Invalid S172 prediction')
        self.check_budget()
        return pick,dict(coarse_frames=len(frames),fine_frames=fine_count)

    def close(self):
        import gc
        import torch
        self.models.clear(); self.backbone=None
        gc.collect()
        if self.device.type=='cuda': torch.cuda.empty_cache()

def apply(ns,base,data_dir,root,clock=None):
    clock=clock or time.perf_counter
    own_start=clock()
    process_start=float(ns.get('_S172_PROCESS_START',own_start))
    diagnostics={'load_error':None,'clips':{},'budget_triggered':None,
                 'covered_clips':0,'remaining_s171_clips':0,
                 'own_limit_seconds':OWN_LIMIT_SECONDS,'process_limit_seconds':PROCESS_LIMIT_SECONDS}
    ns['_S172_DIAGNOSTICS']=diagnostics
    priors=ns.get('_S118_LAST_DIAGNOSTICS',{}).get('s142_clips',{})
    if len(base)==0: return base
    if not any(priors.get(str(ident),{}).get('s172_prior') for ident in base['ID']):
        diagnostics['skipped_no_s160_prior']=True
        return base
    def check_budget():
        now=clock()
        if now-process_start>=PROCESS_LIMIT_SECONDS: raise TimeBudgetExceeded('process_wall_45m')
        if now-own_start>=OWN_LIMIT_SECONDS: raise TimeBudgetExceeded('s172_cumulative_600s')
    engine=None
    start=time.perf_counter()
    try:
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        # Module constructors initialize CPU parameters before loading weights.
        # Preserve the incumbent's RNG state for subsequent stages/calls.
        with torch.random.fork_rng(devices=[]):
            engine=VisualCollision(root)
    except Exception as exc:
        diagnostics['load_error']=type(exc).__name__+': '+str(exc)
        return base
    diagnostics['load_seconds']=time.perf_counter()-start
    engine.check_budget=check_budget
    out=base.copy()
    try:
        for position,(index,row) in enumerate(base.iterrows()):
            ident=str(row['ID']); record={'base_collision':int(row['collision_frame'])}
            start=time.perf_counter()
            try:
                check_budget()
                prior=priors.get(ident,{}).get('s172_prior')
                if prior is None:
                    record['reason']='no_s160_prior'
                    continue
                paths=list(ns['_s008_frame_paths'](Path(data_dir)/'images'/ident))
                numbers=[int(ns['_s008_frame_number'](p)) for p in paths]
                pick,extra=engine.position(paths,prior)
                check_budget()
                if isinstance(pick,bool) or not isinstance(pick,(int,np.integer)) or not 0<=pick<len(paths):
                    raise ValueError('Invalid S172 index')
                value=int(numbers[pick])
                out.at[index,'collision_frame']=value
                out.at[index,'entry_frame']=min(int(row['entry_frame']),value)
                record.update(extra,collision=value,entry=int(out.at[index,'entry_frame']))
                diagnostics['covered_clips']+=1
            except TimeBudgetExceeded as exc:
                out.at[index,'collision_frame']=row['collision_frame']
                out.at[index,'entry_frame']=row['entry_frame']
                record['reason']='budget_guard'
                diagnostics['budget_triggered']=str(exc)
                diagnostics['remaining_s171_clips']=len(base)-position
                break
            except Exception as exc:
                # Explicitly restore both fields even if a late validation fails.
                out.at[index,'collision_frame']=row['collision_frame']
                out.at[index,'entry_frame']=row['entry_frame']
                record['error']=type(exc).__name__+': '+str(exc)
            finally:
                record['seconds']=time.perf_counter()-start
                diagnostics['clips'][ident]=record
    finally:
        try: engine.close()
        except Exception as exc:
            diagnostics['close_error']=type(exc).__name__+': '+str(exc)
        diagnostics['s172_cumulative_seconds']=clock()-own_start
        diagnostics['process_wall_seconds']=clock()-process_start
    return out
