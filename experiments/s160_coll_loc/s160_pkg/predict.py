"""S144 + S147: one unchanged S109 pass, collision-only v3 reranking, then the S147 candidate-offset refiner.

S147: if refiner.json is present, the chosen onset is refined within +-45 frames (s147_refine.py); a refiner
error keeps the unrefined S144 onset for that clip.
S160: if localizer.json is present, the temporal localizer (s160_loc.py) scores every frame of the top-10
candidate windows jointly and replaces the S156 onset; a localizer error keeps the S156 (or unrefined) onset.

ranker.json members: {'name','variant','k','vote_k','model'}; variant tokens: 'contact', 'embed', 'ctx' (as in
candidates/s143_s144_overnight/s144_train.py). A member builds features/context over the first k candidates
(10 or 20) and votes on the first vote_k (default k).
One member: argmax of its GBDT margin. Several: argmax of the mean within-clip rank percentile over the
members that score a candidate. Embeddings: frozen DINOv2-small CLS already shipped in model/stage1_dino.
"""
import importlib.util
import json
import time
from pathlib import Path
import numpy as np

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
EMBED_NAMES = ['emb_step%d' % j for j in range(4)] + ['emb_pre_post', 'emb_max_step', 'emb_peak_excess']


def module(path, name):
    spec=importlib.util.spec_from_file_location(name,path)
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj)
    return obj


def margin(x, model):
    out=np.full(len(x),float(model['base_logit']))
    rate=float(model['learning_rate'])
    for tree in model['trees']:
        left,right,feature,threshold,value=tree['left'],tree['right'],tree['feature'],tree['threshold'],tree['value']
        for r in range(len(x)):
            node=0
            while left[node]>=0:
                node=left[node] if x[r,feature[node]]<=threshold[node] else right[node]
            out[r]+=rate*value[node]
    return out


def percentile(s):
    s=np.asarray(s,float)
    return np.argsort(np.argsort(s,kind='stable'),kind='stable')/max(len(s)-1,1)


def embed_features(cls, schedules):
    out=[]
    for sch in schedules:
        e=np.stack([cls[i] for i in sch])  # float16, exactly as s144_train.embed_matrix
        e=e/np.maximum(np.linalg.norm(e,axis=1,keepdims=True),1e-8)
        steps=[1.0-float(e[j]@e[j+1]) for j in range(4)]
        pre,post=e[:2].mean(0),e[3:].mean(0)
        out.append(steps+[1.0-float(pre@post/max(np.linalg.norm(pre)*np.linalg.norm(post),1e-8)),
                          max(steps),steps[1]+steps[2]-steps[0]-steps[3]])
    return np.asarray(out)


def member_matrix(variant, base, contact, embed, base_names, F):
    x=base;names=list(base_names)
    if 'contact' in variant:
        x=np.concatenate([x,contact],1);names+=['contact_'+k for k in F.CONTACT_NAMES]
    if 'embed' in variant:
        x=np.concatenate([x,embed],1);names+=EMBED_NAMES
    if 'ctx' in variant:
        x=np.concatenate([x,F.context_matrix(x,names)],1);names=names+F.context_names(names)
    return x,names


def choose(members, base, contact, embed, base_names, F, info=None):
    """Winner index from per-candidate matrices (shared by the package and the parity test).
    info (dict, optional) receives votes/count and each member's percentile array (S156 cascade)."""
    n=len(base);votes=np.zeros(n);count=np.zeros(n);single=None;pcts=[]
    for m in members:
        k=min(m['k'],n);vote=min(m.get('vote_k',m['k']),k)  # features/context over k, votes on the first vote_k
        x,names=member_matrix(m['variant'],base[:k],contact[:k],None if embed is None else embed[:k],base_names,F)
        if names!=m['model']['feature_names']:raise ValueError('S144 feature schema mismatch: '+m['name'])
        s=margin(x,m['model'])[:vote];single=s
        pc=percentile(s);pcts.append(pc)
        votes[:vote]+=pc;count[:vote]+=1
    if info is not None:info.update(votes=votes,count=count,pcts=pcts)
    if len(members)==1:
        return int(np.argmax(single))
    return int(np.argmax(np.where(count>0,votes/np.maximum(count,1),-1.)))


class Reranker:
    def __init__(self, emb, package_dir):
        self.path=Path(package_dir);self.scope=dict(emb)
        source=self.path/'rerank_dropin.py'
        exec(compile(source.read_text(encoding='utf-8'),str(source),'exec'),self.scope)
        self.config=json.loads((self.path/'ranker.json').read_text(encoding='utf-8'))
        if self.config.get('format')!='s144_members_v1':raise ValueError('S144 ranker format')
        self.box=module(self.path/'box_features.py','_s144_box_features')
        self.track=module(self.path/'tracker_source.py','_s144_tracker_source')
        self.F=module(self.path/'s144_features.py','_s144_features')
        self.base_names=list(self.scope['H1_FEATURE_NAMES'])+['box_'+s for s in self.box.BOX_NAMES]
        self.members=self.config['members']
        if not self.members:raise ValueError('S144 needs a member')
        for m in self.members:
            if m['k'] not in (10,20):raise ValueError('member k must be 10 or 20')
        self.k=max(m['k'] for m in self.members)
        self.embed_k=max([m['k'] for m in self.members if 'embed' in m['variant']],default=0)
        self.detector=None;self.dino=None;self.device=None
        self.refiner=None
        if (self.path/'refiner.json').is_file():
            self.refiner=json.loads((self.path/'refiner.json').read_text(encoding='utf-8'))
            if self.refiner.get('format') not in ('s147_refiner_v1','s147_refiner_v2','s156_cascade_v1'):raise ValueError('S147 refiner format')
            self.R=module(self.path/'s147_refine.py','_s147_refine')
        self.loc=None;self.loc_error=None
        if (self.path/'localizer.json').is_file():
            try:
                self.loc=json.loads((self.path/'localizer.json').read_text(encoding='utf-8'))
                if self.loc.get('format')!='s160_loc_v1':raise ValueError('S160 localizer format')
                self.L=module(self.path/'s160_loc.py','_s160_loc')
                if self.refiner is None:self.R=module(self.path/'s147_refine.py','_s147_refine')
                z=np.load(self.path/self.loc['nn_file'],allow_pickle=False);self.nets=[]
                for key in self.loc['nn_keys']:
                    pre=key+'/'
                    self.nets.append({'med':z[pre+'med'],'scale':z[pre+'scale'],
                                      'params':{k[len(pre):]:z[k] for k in z.files if k.startswith(pre) and k[len(pre):] not in ('med','scale')}})
            except Exception as exc:  # keep S156 for every clip
                self.loc=None;self.loc_error=type(exc).__name__+': '+str(exc)

    def close(self):
        self.detector=None;self.dino=None
        import torch
        if torch.cuda.is_available():torch.cuda.empty_cache()

    def _candidates(self, features, n, fps):
        """S142 _h1_candidates with k candidates (identical for k=10)."""
        scope=self.scope;decision=scope['DEFAULT_DECISION']
        score=scope['collision_saliency'](features)
        if n!=len(score):raise ValueError('feature length mismatch')
        if n==0:return [],score
        guard=int(decision.guard_frames);lo=min(guard,n-1);hi=max(lo+1,n-guard)
        peaks=[i for i in range(lo,hi) if score[i]>=score[i-1] and (i+1==n or score[i]>score[i+1])]
        top=scope['_guarded_argmax'](score,guard)
        if top not in peaks:peaks.append(top)
        peaks.sort(key=lambda i:(-float(score[i]),i))
        separation=max(3,int(round(0.25*fps)));selected=[]
        for i in peaks:
            if all(abs(i-j)>=separation for j in selected):
                selected.append(i)
                if len(selected)==self.k:break
        for i in peaks:
            if len(selected)==self.k:break
            if i not in selected:selected.append(i)
        return [(i,scope['_onset_index'](score,i,decision.onset_ratio,decision.max_backtrack)) for i in selected],score

    def _embed(self, images):
        import torch
        if self.dino is None:
            from transformers import Dinov2Model
            root=(self.path.parents[1]/'stage1_dino'/'backbone').resolve()
            for name in ('config.json','model.safetensors'):
                if not (root/name).is_file():raise FileNotFoundError(root/name)
            self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.dino=Dinov2Model.from_pretrained(str(root),local_files_only=True,trust_remote_code=False,
                                                  use_safetensors=True).requires_grad_(False).eval().to(self.device)
        keys=sorted(images);out=[]
        with torch.inference_mode(),torch.autocast(device_type=self.device.type,enabled=self.device.type=='cuda'):
            for s in range(0,len(keys),16):
                x=torch.from_numpy(np.stack([images[k] for k in keys[s:s+16]])).to(self.device)
                out.append(self.dino(pixel_values=x).last_hidden_state[:,0].float().cpu().numpy())
        cls=np.concatenate(out).astype(np.float16)  # the training cache stored float16 CLS
        return {k:cls[i] for i,k in enumerate(keys)}

    def locate(self, features, paths):
        import cv2
        n=len(paths);scope=self.scope
        candidates,score=self._candidates(features,n,30.)
        if not candidates:raise ValueError('no candidate peaks')
        schedules=[self.box.indices(peak,n,30.) for peak,onset in candidates]
        wanted=sorted({i for s in schedules for i in s})
        embed_frames={i for s in schedules[:self.embed_k] for i in s}
        if self.detector is None:
            self.detector=self.track.Detector('frcnn',self.path/'detector.pth',max_side=480,score=.3)
        detection={};pixels={}
        for i in wanted:
            im=cv2.imread(str(paths[i]))
            if im is None:raise ValueError('cannot decode frame '+str(paths[i]))
            if i in embed_frames:
                rgb=cv2.cvtColor(cv2.resize(im,(392,224),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB)
                pixels[i]=((rgb.astype(np.float32)/255.0-MEAN)/STD).transpose(2,0,1)
            h,w=im.shape[:2];scale=min(1.,480/max(h,w))
            if scale<1:im=cv2.resize(im,(round(w*scale),round(h*scale)),interpolation=cv2.INTER_AREA)
            rows=self.detector([im])[0]
            hist=np.stack([self.track._hist(im,r) for r in rows]) if len(rows) else np.zeros((0,64),np.float32)
            detection[i]=(rows,hist)
        window=scope['DEFAULT_DECISION'].baseline_window
        jolts=(scope['_normalized_jolt'](np.asarray(features['ego_speed']),window),
               scope['_normalized_jolt'](np.abs(np.asarray(features['ego_theta'])),window),
               scope['_normalized_jolt'](np.asarray(features['warp_diff']),window))
        base=np.asarray([scope['_h1_candidate_vector'](features,score,peak,onset,rank,jolts)+
                         self.box.vector([detection[i] for i in schedules[rank]],self.track)
                         for rank,(peak,onset) in enumerate(candidates)],dtype=np.float64)
        contact=np.asarray([self.F.contact_vector([[[float(v) for v in b] for b in detection[i][0]] for i in sch])
                            for sch in schedules],dtype=np.float64)
        embed=embed_features(self._embed(pixels),schedules[:self.embed_k]) if self.embed_k else None
        info={}
        winner=choose(self.members,base,contact,embed,self.base_names,self.F,info)
        onset=int(candidates[winner][1])
        detail={'candidates':len(candidates),'detected_frames':len(wanted),'embedded_frames':len(pixels),'winner':winner}
        if self.refiner is not None:
            try:
                if self.refiner['format']=='s156_cascade_v1':
                    m=int(self.refiner['n_ranked']);v=info['votes'][:m]
                    order=np.argsort(-v,kind='stable')
                    if int(order[0])!=winner:raise ValueError('cascade order disagrees with the winner')
                    pc=[info['pcts'][q][:m] for q in self.refiner['member_pct_order']]
                    top=[(int(candidates[c][1]),int(candidates[c][0]),[float(r),float(v[c]),float(v[order[0]]-v[c])]+[float(a[c]) for a in pc])
                         for r,c in enumerate(order[:int(self.refiner['k'])])]
                    onset,extra=self.R.cascade(features,n,top,self.refiner,scope['collision_saliency'])
                else:
                    onset,extra=self.R.refine(features,n,onset,int(candidates[winner][0]),self.refiner,scope['collision_saliency'])
                detail.update(extra)
            except Exception as exc:  # keep the unrefined S144 onset for this clip
                detail['refine_error']=type(exc).__name__+': '+str(exc)
        if self.loc_error is not None:detail['loc_load_error']=self.loc_error
        if self.loc is not None:
            try:
                m=int(self.loc['n_ranked']);v=info['votes'][:m]
                order=np.argsort(-v,kind='stable')
                if int(order[0])!=winner:raise ValueError('localizer order disagrees with the winner')
                k=min(int(self.loc['k']),len(order))
                pc=[info['pcts'][q][:m] for q in self.loc['member_pct_order']]
                top=[(int(candidates[c][1]),int(candidates[c][0]),[float(r),float(v[c]),float(v[order[0]]-v[c])]+[float(a[c]) for a in pc])
                     for r,c in enumerate(order[:k])]
                cm=np.concatenate([base[:m],contact[:m]],1)
                cnames=list(self.base_names)+['contact_'+q for q in self.F.CONTACT_NAMES]
                cv=np.concatenate([cm,self.F.context_matrix(cm,cnames)],1)
                if cv.shape[1]+106!=len(self.loc['feature_names']):raise ValueError('S160 feature schema mismatch')
                X,M,FR=self.L.rows(features,n,top,cv[order[:k]],self.loc['signals'],scope['collision_saliency'],self.R)
                new,_p=self.L.localize(X,M,FR,self.loc,self.nets)
                if not 0<=int(new)<n or not np.isfinite(_p).all():raise ValueError('S160 invalid localizer output')
                detail.update(loc_from=int(onset),loc_to=int(new));onset=int(new)
            except Exception as exc:  # keep the S156 onset for this clip
                detail['loc_error']=type(exc).__name__+': '+str(exc)
        return onset,detail


def predict_collision_only(ns,data_dir,model_dir,package_dir,min_n,diagnostics):
    """Capture public API motion arrays, return original indices to every S109 downstream field."""
    emb=ns['_S2_COLLISION_NAMESPACE']
    own_collision,own_folder=emb['locate_collision'],emb['predict_folder']
    active=[None];captures={}
    def locate(features,decision=None):
        result=own_collision(features,decision)
        try:
            if active[0] is not None and len(features['ego_speed'])>int(min_n):
                active[0]['features']={k:np.asarray(v).copy() for k,v in features.items()}
        except Exception as exc:
            diagnostics.setdefault('s142_capture_errors',[]).append(type(exc).__name__)
        return result
    def folder(path,*args,**kwargs):
        current={'path':Path(path)};active[0]=current
        try:
            row=own_folder(path,*args,**kwargs)
            if 'features' in current:captures[str(row['ID'])]=current
            return row
        finally:active[0]=None
    emb['locate_collision'],emb['predict_folder']=locate,folder
    try:baseline=ns['_S118_BASE_PREDICT_STAGE2'](data_dir,model_dir)
    finally:
        emb['locate_collision'],emb['predict_folder']=own_collision,own_folder
        active[0]=None
    diagnostics['single_base_call']=True
    diagnostics['s142_capture_count']=len(captures)
    diagnostics['captured_collision_frames']={}
    diagnostics['entry_validity_clamps']=[]
    out=baseline.copy();engine=None
    if not captures:return out
    started=time.perf_counter()
    try:
        engine=Reranker(emb,package_dir)
        for row_index,row in out.iterrows():
            ident=str(row['ID']);capture=captures.pop(ident,None)
            if capture is None:continue
            t=time.perf_counter()
            try:
                paths=emb['frame_paths'](capture['path'])
                numbers=emb['frame_numbers'](capture['path'])
                if len(paths)!=len(capture['features']['ego_speed']):raise ValueError('frame count mismatch')
                index,detail=engine.locate(capture['features'],paths)
                if not 0<=index<len(numbers):raise ValueError('collision index out of range')
                value=int(numbers[index]);out.at[row_index,'collision_frame']=value
                if int(row['entry_frame'])>value:
                    out.at[row_index,'entry_frame']=value
                    diagnostics['entry_validity_clamps'].append(ident)
                diagnostics['captured_collision_frames'][ident]=value
                diagnostics.setdefault('s142_clips',{})[ident]=dict(detail,seconds=time.perf_counter()-t)
            except Exception as exc:
                diagnostics.setdefault('s142_fallbacks',{})[ident]=type(exc).__name__+': '+str(exc)
    except Exception as exc:
        diagnostics['s142_load_error']=type(exc).__name__+': '+str(exc)
    finally:
        captures.clear()
        if engine is not None:engine.close()
    diagnostics['s142_extra_seconds']=time.perf_counter()-started
    return out
