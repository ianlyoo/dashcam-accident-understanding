"""Bounded, pixel-only temporal cadence and spatial/rolling-band descriptors.

No container/codec/fps/size or file identity is a learned feature. One file is
decoded independently; extraction keeps 24 low-resolution consecutive frames.
"""
from pathlib import Path
import time

import cv2
import numpy as np

VERSION = 's163-cue-v1'
FRAMES = 24


def describe(path):
    start = time.perf_counter()
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError('Cannot open video')
    gray, rgb, spatial = [], [], []
    try:
        for i in range(FRAMES):
            ok, frame = cap.read()
            if not ok:
                break
            small = cv2.resize(frame, (192, 108), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.
            rgb.append(small.mean((0, 1)))
            gray.append(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
            if i in (0, 12, 23):
                h, w = frame.shape[:2]
                for cy, cx in ((.5, .5), (.25, .25), (.75, .75)):
                    y = max(0, min(h - 256, int(cy * h) - 128))
                    x = max(0, min(w - 256, int(cx * w) - 128))
                    patch = frame[y:y + min(h, 256), x:x + min(w, 256)]
                    patch = cv2.resize(patch, (256, 256), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.
                    spatial.append(spatial_features(patch))
    finally:
        cap.release()
    if len(gray) < 12:
        raise ValueError('Fewer than 12 consecutive frames')
    g = np.asarray(gray, dtype=np.float64)
    features = {}

    def add(name, values):
        v = np.asarray(values, dtype=np.float64).ravel()
        for q, stat in zip(('mean', 'std', 'q10', 'q50', 'q90'),
                           (v.mean(), v.std(), *np.quantile(v, [.1, .5, .9]))):
            features[name + '_' + q] = float(stat)

    # Differences relative to each video's own median: cadence survives global
    # gain/contrast changes and avoids requiring a particular nominal frame rate.
    d = np.abs(np.diff(g, axis=0)).mean((1, 2))
    dn = d / (np.median(d) + 1e-5)
    add('cadence_relative', dn)
    features['cadence_near_duplicate'] = float(np.mean(dn < .15))
    features['cadence_exact_duplicate'] = float(np.mean(d < 1e-5))
    for lag in range(1, 7):
        a, b = dn[:-lag], dn[lag:]
        features['cadence_ac' + str(lag)] = float(np.mean((a-a.mean())*(b-b.mean())) /
                                                   (a.std()*b.std()+1e-6))
    add('motion_abs', d)
    # Temporal tile coherence separates global refresh/exposure variations from
    # independent moving objects; exposure variation remains an explicit risk.
    tiles = g.reshape(len(g), 6, 18, 8, 24).mean((2, 4)).reshape(len(g), -1)
    delta = np.diff(tiles, axis=0)
    add('flicker_coherence', np.abs(delta.mean(1)) / (np.abs(delta).mean(1) + 1e-5))
    add('global_flicker', np.diff(g.mean((1, 2))))
    for axis, name in ((2, 'row'), (1, 'column')):
        profiles = np.log(g.mean(axis=axis) + .03).astype(np.float32)
        smooth = cv2.GaussianBlur(profiles, (0, 1), 4.)
        band = profiles - smooth
        band -= band.mean(axis=0, keepdims=True)  # stationary scene stripes removed
        add(name + '_band', np.sqrt(np.mean(band * band, axis=1)))
        power = np.abs(np.fft.rfft2(band * np.hanning(len(band))[:, None])) ** 2
        power[0] = 0
        power[:, :2] = 0
        total = power.sum() + 1e-10
        features[name + '_peak_fraction'] = float(power.max() / total)
        features[name + '_temporal_peak'] = float(power.sum(1).max() / total)
        features[name + '_spatial_peak'] = float(power.sum(0).max() / total)
        features[name + '_high_spatial_fraction'] = float(power[:, power.shape[1]//2:].sum()/total)
        features[name + '_power'] = float(np.log1p(total))
    rgb = np.asarray(rgb)
    add('color_flicker', np.diff(rgb - rgb.mean(1, keepdims=True), axis=0))
    for key in spatial[0]:
        add(key, [row[key] for row in spatial])
    x = np.asarray(list(features.values()), dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite cue')
    return dict(version=VERSION, features=features, frames=len(g), seconds=time.perf_counter()-start)


def spatial_features(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    channels = dict(luma=gray, chroma=bgr[:, :, 2]-bgr[:, :, 1])
    out = {}
    freq = np.fft.fftfreq(256)
    yy, xx = np.meshgrid(freq, freq, indexing='ij')
    radius = np.sqrt(xx*xx + yy*yy)
    high = (radius > .08) & (radius < .48)
    window = np.outer(np.hanning(256), np.hanning(256))
    for name, channel in channels.items():
        residual = channel - cv2.GaussianBlur(channel, (0, 0), 2.)
        p = np.abs(np.fft.fft2(residual * window)) ** 2
        v = p[high]
        total = v.sum()+1e-9
        out[name+'_spectral_peak'] = float(v.max()/total)
        out[name+'_spectral_top10'] = float(np.partition(v, -10)[-10:].sum()/total)
        out[name+'_spectral_entropy'] = float(-np.sum((v/total)*np.log(v/total+1e-12)))
        out[name+'_row_stripe'] = float(p[high & (np.abs(xx)<.012)].sum()/total)
        out[name+'_col_stripe'] = float(p[high & (np.abs(yy)<.012)].sum()/total)
        out[name+'_high_energy'] = float(p[(radius>.25)&high].sum()/total)
        out[name+'_rms'] = float(np.sqrt(np.mean(residual**2)))
    return out


def predict(features, model):
    """Portable forest inference; no sklearn dependency in the release."""
    x = np.asarray([features[k] for k in model['features']], dtype=np.float32)
    votes = []
    for tree in model['trees']:
        j = 0
        while tree['left'][j] != -1:
            j = tree['left'][j] if x[tree['feature'][j]] <= tree['threshold'][j] else tree['right'][j]
        votes.append(tree['probability'][j])
    return float(np.mean(votes))
