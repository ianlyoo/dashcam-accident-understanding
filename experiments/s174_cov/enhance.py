# S174 append-only extension to the exact S171 S161 tracker.
# The original S161 result is returned byte-for-byte in decision terms whenever
# its conservative observed-crossing gate accepts. Only abstentions get a
# second collision-anchored, multi-actor geometric observation pass.

_s174_original_analyze = Tracker.analyze


def _s174_refine(clip, outside, inside, horizon, aspect, p, lane_k):
    """Check native frames around one observed coarse crossing."""
    lo, lr = outside
    hi, hr = inside
    if hi <= lo + 1:
        return hi, lo, hi
    geo = Params(**vars(p));geo.lane_k = lane_k
    frames = list(range(lo + 1, hi))
    if len(frames) > 12:
        frames = sorted(set(int(round(v)) for v in np.linspace(lo + 1, hi - 1, 12)))
    last_out, first_in = lo, hi
    def inspect(i,last_out,first_in):
        if i not in clip.dets and len(clip.dets) >= p.max_detect_frames:
            return last_out,first_in
        try:clip.ensure([i])
        except FrameBudgetExceeded:return last_out,first_in
        t = (i - lo) / float(hi - lo)
        predicted = (1 - t) * lr[:4] + t * hr[:4]
        rows = clip.dets.get(i, [])
        if not len(rows):return last_out,first_in
        best = max(rows, key=lambda r: _iou(r, predicted))
        if _iou(best, predicted) < 0.25:return last_out,first_in
        state = _state(best, horizon, aspect, geo)
        if state is None:return last_out,first_in
        if state[0] and i < first_in:first_in = i
        elif not state[0] and i > last_out and i < first_in:last_out = i
        return last_out,first_in
    for i in frames:last_out,first_in=inspect(i,last_out,first_in)
    # After the coarse bracket narrows, scan its actual native frames.
    if 1 < first_in-last_out <= 14:
        for i in range(last_out+1,first_in):
            last_out,first_in=inspect(i,last_out,first_in)
    return first_in, last_out, first_in


def _s174_fallback(self, paths, numbers, collision_index):
    n=len(paths);c=max(0,min(n-1,int(collision_index)))
    if n<5:return Result(entry_index=None,reason='s174_short')
    fps=30.0 if n>self.params.long_n else 10.0
    p=Params(**vars(self.params));p.batch=2;p.max_detect_frames=48
    step=max(1,int(round(fps/2.5)))
    near=sorted(set(max(0,min(n-1,c+int(round(v*fps)))) for v in (-0.6,-0.3,0.0,0.2)),reverse=True)
    clip=Clip(paths,self.detector,p)
    clip.ensure(near)
    seeds=[]
    for i in near:
        for j,row in enumerate(clip.dets.get(i,[])):
            score=float(_impact(row,(i-c)/fps))
            seeds.append((score,i,row,clip.hists[i][j]))
    tracks=[]
    for score,i,row,hist in sorted(seeds,key=lambda x:-x[0]):
        if any(_iou(row,t['rows'][0][1])>=0.60 for t in tracks):continue
        tracks.append(dict(score=score,rows=[(i,row)],hist=hist,velocity=np.zeros(4,np.float32),misses=0))
        if len(tracks)==3:break
    if not tracks:return Result(entry_index=None,reason='s174_no_anchor',detected_frames=len(clip.dets))
    lo=max(0,c-int(round(6*fps)))
    grid=list(range(min(n-1,c+step),lo-1,-step))
    if lo>0:grid += [int(v) for v in np.linspace(0,lo,7)]
    grid=sorted(set(grid+near),reverse=True)
    for i in grid:
        active=[t for t in tracks if i<t['rows'][-1][0] and t['misses']<=3]
        if not active:continue
        if i not in clip.dets:
            if len(clip.dets)>=p.max_detect_frames:break
            clip.ensure([i])
        used=set()
        for track in active:
            last_i,last=track['rows'][-1];gap=last_i-i
            predicted=last[:4]-track['velocity']*min(gap,2*step)
            scored=[(_assoc(row,predicted,last,clip.hists[i][j],track['hist']),j,row)
                    for j,row in enumerate(clip.dets.get(i,[])) if j not in used]
            scored=[x for x in scored if x[0]>1.0]
            if not scored:
                track['misses']+=1;continue
            _,j,row=max(scored,key=lambda x:x[0]);used.add(j)
            track['velocity']=0.5*track['velocity']+0.5*(last[:4]-row[:4])/max(gap,1)
            track['rows'].append((i,row));track['misses']=0
    horizon,hn=estimate_horizon(clip,p);aspect=clip.aspect or 0.5625
    geometries=[(horizon,p.lane_k)]
    geometries += [(horizon,1.20),(horizon,1.50)]
    if hn<5:geometries += [(0.47,p.lane_k),(0.53,p.lane_k)]
    for track in sorted(tracks,key=lambda t:-t['score']):
        rows=sorted(track['rows'],key=lambda x:x[0])
        if len(rows)<4:continue
        for h,lane_k in geometries:
            geo=Params(**vars(p));geo.lane_k=lane_k
            states=[(i,row,_state(row,h,aspect,geo)) for i,row in rows]
            states=[x for x in states if x[2] is not None]
            if len(states)<4 or not states[-1][2][0]:continue
            j=len(states)-1
            while j>0 and states[j-1][2][0]:j-=1
            if j==0:continue
            outside=states[j-1];inside=states[j]
            if outside[2][1] not in ('LEFT','RIGHT'):continue
            outside_count=sum(not x[2][0] for x in states[:j]);inside_count=len(states)-j
            if outside_count<2 or inside_count<2:continue
            entry,low,high=_s174_refine(clip,(outside[0],outside[1]),(inside[0],inside[1]),h,aspect,p,lane_k)
            result=Result(entry_index=int(entry),side=outside[2][1],reason='crossing',
                          bracket=[int(low),int(high)],collision_index=c,fps=fps,
                          inside_samples=inside_count,outside_samples=outside_count,
                          track_len=len(rows),detected_frames=len(clip.dets),
                          horizon=float(h),horizon_n=int(hn),lane_k=float(lane_k),
                          anchor=int(track['rows'][0][0]),
                          anchor_box=[round(float(v),4) for v in track['rows'][0][1][:4]],
                          s174_actor_candidates=len(tracks),s174_seed_score=track['score'])
            if decide(result,'s109')[0] is not None:
                return result
    return Result(entry_index=None,reason='s174_no_supported_crossing',
                  detected_frames=len(clip.dets),s174_actor_candidates=len(tracks),horizon=float(horizon))


def _s174_analyze(self,paths,numbers,collision_index):
    original=_s174_original_analyze(self,paths,numbers,collision_index)
    if decide(original,'s109')[0] is not None:
        return original
    try:
        result=_s174_fallback(self,paths,numbers,collision_index)
        if decide(result,'s109')[0] is not None:
            return result
    except torch.cuda.OutOfMemoryError:
        raise
    except Exception:
        pass
    return original


Tracker.analyze=_s174_analyze
