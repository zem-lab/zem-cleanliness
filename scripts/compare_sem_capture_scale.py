"""Offline feature-scale check of an existing four-shot diagnostic."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def compare(root):
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    sift = cv2.SIFT_create(nfeatures=4000)
    features = []
    for shot in manifest['shots']:
        raw = cv2.imread(str(root / (shot['name'] + '.png')), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise ValueError('Missing capture: ' + shot['name'])
        lo, hi = np.percentile(raw, [.5, 99.5])
        normalized = np.clip((raw.astype(float)-lo)*255/max(hi-lo, 1), 0, 255).astype(np.uint8)
        features.append(sift.detectAndCompute(normalized, None))
    results = []
    k0, d0 = features[0]
    for shot, (keys, desc) in zip(manifest['shots'][1:], features[1:]):
        matches = [] if d0 is None or desc is None else [a for pair in cv2.BFMatcher().knnMatch(d0, desc, k=2)
                                                       if len(pair) == 2 for a, b in [pair] if a.distance < .7*b.distance]
        result = dict(name=shot['name'], matches=len(matches),
                      expectedScaleRatio=shot['requestedMagnification']/manifest['shots'][0]['requestedMagnification'])
        if len(matches) >= 8:
            source = np.float32([k0[m.queryIdx].pt for m in matches])
            target = np.float32([keys[m.trainIdx].pt for m in matches])
            cv2.setRNGSeed(0)
            matrix, mask = cv2.estimateAffinePartial2D(source, target, method=cv2.RANSAC, ransacReprojThreshold=2)
            if matrix is not None:
                inliers = mask.ravel().astype(bool)
                residual = np.linalg.norm(source @ matrix[:, :2].T + matrix[:, 2] - target, axis=1)
                result.update(imageScaleRatio=float(np.linalg.norm(matrix[:, 0])), inliers=int(inliers.sum()),
                              medianResidualPx=float(np.median(residual[inliers])), matrix=matrix.tolist())
        results.append(result)
    output = {'comparisons': results, 'acceptedTransform': False}
    (root / 'image_comparison.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    compare(parser.parse_args().root.resolve())
