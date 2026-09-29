"""Candidate-local S132 participant geometry; no clip-position fields."""
import numpy as np

OFFSETS = (-0.6, -0.3, 0.0, 0.3, 0.6)
BOX_NAMES = ('found','matches','anchor_area','anchor_height','anchor_width','anchor_bottom',
 'anchor_center_x','anchor_score','anchor_lane_overlap','anchor_bottom_overlap',
 'anchor_edge_touch','pre_area','post_area','area_growth','area_log_ratio','scale_jump',
 'pre_bottom','post_bottom','bottom_growth','center_shift','pre_overlap','post_overlap',
 'overlap_growth','match_iou_min','match_iou_mean','max_area','max_impact',
 'mean_vehicle_count','max_pair_iou','disappeared_after')

def indices(peak,n,fps):
    return [max(0,min(n-1,int(peak)+int(round(t*fps)))) for t in OFFSETS]

def area(r):
    return max(0.,float((r[2]-r[0])*(r[3]-r[1])))

def overlap(r, region):
    x=max(0.,min(r[2],region[2])-max(r[0],region[0]))
    y=max(0.,min(r[3],region[3])-max(r[1],region[1]))
    return float(x*y/max(area(r),1e-8))

def vector(sample, track):
    # sample contains five (boxes, appearance histograms) pairs.
    counts=[len(s[0]) for s in sample]
    anchors=[(track._impact(r,OFFSETS[j]),j,k,r) for j in (1,2,3)
             for k,r in enumerate(sample[j][0])]
    vals=dict.fromkeys(BOX_NAMES,0.)
    vals['mean_vehicle_count']=float(np.mean(counts))
    if not anchors:
        return [vals[k] for k in BOX_NAMES]
    impact,aj,ak,anchor=max(anchors,key=lambda a:a[0])
    hist=sample[aj][1][ak]
    matched=[]
    for j,(rows,hists) in enumerate(sample):
        scores=[(track._assoc(r,anchor,anchor,h,hist),r) for r,h in zip(rows,hists)]
        best=max(scores,key=lambda a:a[0]) if scores else None
        matched.append(best[1] if best is not None and best[0]>1.2 else None)
    def mean_at(js,fn):
        arr=[fn(matched[j]) for j in js if matched[j] is not None]
        return float(np.mean(arr)) if arr else 0.
    pre=mean_at((0,1),area);post=mean_at((3,4),area)
    lane=lambda r:overlap(r,(.25,.5,.75,1.))
    bottom=lambda r:float(r[3])
    center=lambda r:float((r[0]+r[2])*.5)
    pre_bottom=mean_at((0,1),bottom);post_bottom=mean_at((3,4),bottom)
    pre_overlap=mean_at((0,1),lane);post_overlap=mean_at((3,4),lane)
    adjacent=[abs(np.log((area(b)+1e-5)/(area(a)+1e-5)))
              for a,b in zip(matched,matched[1:]) if a is not None and b is not None]
    ious=[track._iou(anchor,r) for r in matched if r is not None]
    allrows=[r for s in sample for r in s[0]]
    pairs=[track._iou(a,b) for rows,_ in sample for i,a in enumerate(rows) for b in rows[i+1:]]
    vals.update(found=1.,matches=sum(r is not None for r in matched),anchor_area=area(anchor),
      anchor_height=float(anchor[3]-anchor[1]),anchor_width=float(anchor[2]-anchor[0]),
      anchor_bottom=bottom(anchor),anchor_center_x=center(anchor),anchor_score=float(anchor[4]),
      anchor_lane_overlap=lane(anchor),anchor_bottom_overlap=overlap(anchor,(.25,.75,.75,1.)),
      anchor_edge_touch=float(anchor[0]<.02 or anchor[2]>.98 or anchor[3]>.98),
      pre_area=pre,post_area=post,area_growth=post-pre,
      area_log_ratio=float(np.clip(np.log((post+1e-5)/(pre+1e-5)),-8,8)),
      scale_jump=max(adjacent,default=0.),pre_bottom=pre_bottom,post_bottom=post_bottom,
      bottom_growth=post_bottom-pre_bottom,center_shift=mean_at((3,4),center)-mean_at((0,1),center),
      pre_overlap=pre_overlap,post_overlap=post_overlap,overlap_growth=post_overlap-pre_overlap,
      match_iou_min=min(ious,default=0.),match_iou_mean=float(np.mean(ious)) if ious else 0.,
      max_area=max(map(area,allrows),default=0.),max_impact=float(impact),
      max_pair_iou=max(pairs,default=0.),disappeared_after=float(matched[2] is not None and matched[4] is None))
    return [float(vals[k]) for k in BOX_NAMES]
