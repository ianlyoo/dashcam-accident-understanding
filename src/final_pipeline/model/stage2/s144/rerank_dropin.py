"""H1 S109 namespace drop-in: append after s109_embedded_s2.py definitions.

Call rerank_locate_collision(features, n, model_data, fps=30.0).  The model_data
mapping comes from h1_ranker_portable.json, loaded once with load_h1_ranker().
No clip-position feature or scikit-learn dependency is used at inference.
"""
import json as _h1_json
import numpy as _h1_np

H1_FEATURE_NAMES = (
    'saliency_peak','saliency_onset','saliency_rank','saliency_prominence',
    'saliency_left3','saliency_right3','saliency_left9','saliency_right9',
    'saliency_width_half','saliency_1step_drop','saliency_3step_drop',
    'jolt_speed','jolt_theta','jolt_warp',
    'jolt_speed_rank','jolt_theta_rank','jolt_warp_rank',
    'ego_speed_peak','ego_speed_pre','ego_speed_post','ego_speed_drop',
    'ego_theta_abs_peak','warp_diff_peak',
    'resid_energy_peak','resid_energy_pre','resid_energy_post',
    'resid_energy_rise','resid_outlier_peak','resid_outlier_pre',
    'resid_center_x','resid_center_abs','resid_center_shift',
    'resid_median_peak','tracked_peak',
)


def load_h1_ranker(path):
    with open(path,encoding='utf-8') as handle:
        model = _h1_json.load(handle)
    if tuple(model['feature_names']) != H1_FEATURE_NAMES:
        raise ValueError('H1 feature order mismatch')
    return model


def _h1_mean(values, lo, hi):
    lo = max(0,int(lo))
    hi = min(len(values),int(hi))
    return float(_h1_np.mean(values[lo:hi])) if hi > lo else 0.0


def _h1_at(values,index):
    return float(values[max(0,min(len(values)-1,int(index)))])


def _h1_candidates(features,n,fps):
    score = collision_saliency(features)
    if n != len(score):
        raise ValueError(f'feature length {len(score)} != n {n}')
    if n == 0:
        return [],score
    guard = int(DEFAULT_DECISION.guard_frames)
    lo = min(guard,n-1)
    hi = max(lo+1,n-guard)
    peaks = [i for i in range(lo,hi)
             if score[i] >= score[i-1] and (i+1 == n or score[i] > score[i+1])]
    global_peak = _guarded_argmax(score,guard)
    if global_peak not in peaks:
        peaks.append(global_peak)
    peaks.sort(key=lambda i:(-float(score[i]),i))
    separation = max(3,int(round(0.25*fps)))
    selected = []
    for i in peaks:
        if all(abs(i-j) >= separation for j in selected):
            selected.append(i)
            if len(selected) == 10:
                break
    for i in peaks:
        if len(selected) == 10:
            break
        if i not in selected:
            selected.append(i)
    candidates = [(i,_onset_index(score,i,DEFAULT_DECISION.onset_ratio,
                                  DEFAULT_DECISION.max_backtrack)) for i in selected]
    return candidates,score


def _h1_candidate_vector(features,score,peak,onset,rank,jolts):
    n = len(score)
    speed_jolt,theta_jolt,warp_jolt = jolts
    pre_speed = _h1_mean(features['ego_speed'],peak-9,peak)
    post_speed = _h1_mean(features['ego_speed'],peak+1,peak+10)
    pre_energy = _h1_mean(features['resid_energy'],peak-9,peak)
    post_energy = _h1_mean(features['resid_energy'],peak+1,peak+10)
    left3 = _h1_mean(score,peak-3,peak)
    right3 = _h1_mean(score,peak+1,peak+4)
    left9 = _h1_mean(score,peak-9,peak)
    right9 = _h1_mean(score,peak+1,peak+10)
    height = float(score[peak])
    threshold = 0.5*height
    left = peak
    while left > 0 and score[left-1] >= threshold and peak-left < 15:
        left -= 1
    right = peak
    while right+1 < n and score[right+1] >= threshold and right-peak < 15:
        right += 1
    center = _h1_at(features['resid_center_x'],peak)
    vector = {
        'saliency_peak':height,
        'saliency_onset':_h1_at(score,onset),
        'saliency_rank':float(rank),
        'saliency_prominence':height-max(left9,right9),
        'saliency_left3':left3,
        'saliency_right3':right3,
        'saliency_left9':left9,
        'saliency_right9':right9,
        'saliency_width_half':float(right-left+1),
        'saliency_1step_drop':height-_h1_at(score,peak+1),
        'saliency_3step_drop':height-_h1_at(score,peak+3),
        'jolt_speed':_h1_at(speed_jolt,peak),
        'jolt_theta':_h1_at(theta_jolt,peak),
        'jolt_warp':_h1_at(warp_jolt,peak),
        'jolt_speed_rank':float(_h1_np.mean(speed_jolt <= speed_jolt[peak])),
        'jolt_theta_rank':float(_h1_np.mean(theta_jolt <= theta_jolt[peak])),
        'jolt_warp_rank':float(_h1_np.mean(warp_jolt <= warp_jolt[peak])),
        'ego_speed_peak':_h1_at(features['ego_speed'],peak),
        'ego_speed_pre':pre_speed,
        'ego_speed_post':post_speed,
        'ego_speed_drop':pre_speed-post_speed,
        'ego_theta_abs_peak':abs(_h1_at(features['ego_theta'],peak)),
        'warp_diff_peak':_h1_at(features['warp_diff'],peak),
        'resid_energy_peak':_h1_at(features['resid_energy'],peak),
        'resid_energy_pre':pre_energy,
        'resid_energy_post':post_energy,
        'resid_energy_rise':post_energy-pre_energy,
        'resid_outlier_peak':_h1_at(features['resid_outlier'],peak),
        'resid_outlier_pre':_h1_mean(features['resid_outlier'],peak-9,peak),
        'resid_center_x':center,
        'resid_center_abs':abs(center),
        'resid_center_shift':center-_h1_mean(features['resid_center_x'],peak-9,peak),
        'resid_median_peak':_h1_at(features['resid_median'],peak),
        'tracked_peak':_h1_at(features['tracked'],peak),
    }
    return [float(vector[name]) for name in H1_FEATURE_NAMES]


def _h1_margin(vector,model_data):
    vector = _h1_np.asarray(vector,dtype=_h1_np.float32)
    margin = float(model_data['base_logit'])
    rate = float(model_data['learning_rate'])
    for tree in model_data['trees']:
        node = 0
        while tree['left'][node] >= 0:
            if vector[tree['feature'][node]] <= tree['threshold'][node]:
                node = tree['left'][node]
            else:
                node = tree['right'][node]
        margin += rate*tree['value'][node]
    return margin


def rerank_locate_collision(features,n,model_data=None,fps=30.0):
    """Return a frame index; preserve S109 on short clips or absent model."""
    if n <= 310 or model_data is None:
        score = collision_saliency(features)
        if not len(score):
            return 0
        peak = _guarded_argmax(score,DEFAULT_DECISION.guard_frames)
        return _onset_index(score,peak,DEFAULT_DECISION.onset_ratio,
                            DEFAULT_DECISION.max_backtrack)
    if tuple(model_data['feature_names']) != H1_FEATURE_NAMES:
        raise ValueError('H1 feature order mismatch')
    candidates,score = _h1_candidates(features,n,fps)
    if not candidates:
        return 0
    window = DEFAULT_DECISION.baseline_window
    jolts = (
        _normalized_jolt(_h1_np.asarray(features['ego_speed']),window),
        _normalized_jolt(_h1_np.abs(_h1_np.asarray(features['ego_theta'])),window),
        _normalized_jolt(_h1_np.asarray(features['warp_diff']),window),
    )
    vectors = [_h1_candidate_vector(features,score,peak,onset,rank,jolts)
               for rank,(peak,onset) in enumerate(candidates)]
    winner = max(range(len(candidates)),key=lambda i:_h1_margin(vectors[i],model_data))
    return int(candidates[winner][1])
