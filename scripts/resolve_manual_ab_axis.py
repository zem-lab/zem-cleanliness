"""Alternative offline feature registration; preserves failed phase-axis evidence."""
import json
from pathlib import Path
import cv2
import numpy as np

OUT = Path(__file__).resolve().parents[1] / 'artifacts/manual_ab_sample2_20260908/axis'
data = json.loads((OUT / 'axis.json').read_text(encoding='utf-8'))
detector = cv2.SIFT_create(nfeatures=7000, contrastThreshold=.015)
features = {name: detector.detectAndCompute(cv2.imread(str(OUT / (name + '.u8.png')), 0), None)
            for name in ['o1', 'xp', 'xm', 'yp', 'ym', 'o2']}
registrations = {}
for name in ['xp', 'xm', 'yp', 'ym', 'o2']:
    k, d = features['o1']; k2, d2 = features[name]
    pairs = cv2.BFMatcher().knnMatch(d, d2, k=2)
    good = [m for m, n in pairs if m.distance < .7 * n.distance]
    a = np.float32([k[m.queryIdx].pt for m in good]); b = np.float32([k2[m.trainIdx].pt for m in good])
    matrix, inlier = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=2, maxIters=5000)
    if matrix is None: raise ValueError('Feature registration failed: ' + name)
    mask = inlier.ravel() > 0
    delta = np.median((b-a)[mask], axis=0)
    residual = np.linalg.norm((b-a)[mask] - delta, axis=1)
    registrations[name] = {'matches': len(good), 'inliers': int(mask.sum()), 'deltaPixel': delta.tolist(),
        'translationResidualP95Px': float(np.percentile(residual, 95)), 'affine': matrix.tolist(),
        'sourceInliers': a[mask].tolist(), 'targetInliers': b[mask].tolist()}
shots = {r['name']: r for r in data['shots']}
ds = np.array([np.subtract(registrations['xp']['deltaPixel'], registrations['xm']['deltaPixel']),
               np.subtract(registrations['yp']['deltaPixel'], registrations['ym']['deltaPixel'])]).T
de = np.array([np.subtract(shots['xp']['postCaptureUm'], shots['xm']['postCaptureUm']),
               np.subtract(shots['yp']['postCaptureUm'], shots['ym']['postCaptureUm'])]).T
g = -de @ np.linalg.inv(ds)
singular = np.linalg.svd(g)[1]
return_error = float(np.linalg.norm(g @ registrations['o2']['deltaPixel']))
valid = bool(all(r['inliers'] >= 6 and r['translationResidualP95Px'] < 3 for r in registrations.values())
             and return_error < 2 and np.all(abs(singular / data['nominalPixelSizeUm'] - 1) < .1))
result = {**data, 'method': 'SIFT ratio 0.7 + RANSAC 2 px; median inlier translation',
    'phaseCorrelationEvidence': 'axis.json', 'phaseCorrelationValid': data['valid'],
    'registrations': registrations, 'pixelToStageMatrix': g.tolist(), 'singularValuesUmPerPx': singular.tolist(),
    'returnErrorUm': return_error, 'valid': valid}
(OUT / 'axis_sift.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'valid': valid, 'matrix': g.tolist(), 'scales': singular.tolist(), 'returnErrorUm': return_error,
    'residualsP95Px': {name: r['translationResidualP95Px'] for name, r in registrations.items()}}, indent=2))
