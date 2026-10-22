"""Read-only SIFT overlap audit of saved SEM frames versus recorded stage moves."""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_ab_particle_offline import read_image
from backend.cleanliness_core.atomic_json import write_json_atomic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.task/'task.json').read_text(encoding='utf-8'))
    audit = manifest['calibration']['abPointResult']['semRefinement']
    shots = [s for s in audit['captures'] if s.get('status') == 'CAPTURED' and str(s.get('label', '')).startswith('seed')]
    sift = cv2.SIFT_create(nfeatures=4000)
    features = {}
    result = {'acceptedTransform': False, 'kind': 'saved_stage_overlap_audit', 'pairs': []}
    for s in shots:
        im = read_image(Path(s['capture']['rawPath']))
        lo, hi = np.percentile(im, [.5, 99.5])
        im = np.clip((im.astype(float)-lo)*255/max(hi-lo, 1), 0, 255).astype(np.uint8)
        features[s['index']] = sift.detectAndCompute(im, None)
    for i, first in enumerate(shots):
        for second in shots[i+1:]:
            delta = np.subtract(second['captureStageUm'], first['captureStageUm'])
            if first['phase'] != second['phase'] or not 100 < np.linalg.norm(delta) < 1700:
                continue
            k1, d1 = features[first['index']]
            k2, d2 = features[second['index']]
            if d1 is None or d2 is None:
                continue
            good = [a for pair in cv2.BFMatcher().knnMatch(d1, d2, k=2) if len(pair) == 2 for a, b in [pair] if a.distance < .65*b.distance]
            if len(good) < 15:
                continue
            p1 = np.float32([k1[m.queryIdx].pt for m in good])
            p2 = np.float32([k2[m.trainIdx].pt for m in good])
            cv2.setRNGSeed(0)
            matrix, inliers = cv2.estimateAffinePartial2D(p1, p2, method=cv2.RANSAC, ransacReprojThreshold=2)
            if matrix is None:
                continue
            count = int(inliers.sum())
            scale = np.linalg.norm(matrix[:, 0])
            if count < 15 or count/len(good) < .5 or abs(scale-1) > .02:
                continue
            center = np.array(first['capture']['shape'][::-1])/2
            shift = matrix[:, :2]@center+matrix[:, 2]-center
            axis = np.array(audit['pixelAxes'][first['phase']]['pixelToStageMatrix'])
            measured = -axis@shift
            residual = np.linalg.norm(p1@matrix[:, :2].T+matrix[:, 2]-p2, axis=1)[inliers.ravel().astype(bool)]
            result['pairs'].append(dict(first=first['index'], second=second['index'], matches=len(good), inliers=count,
                                        imageScale=float(scale), medianResidualPx=float(np.median(residual)),
                                        recordedStageDeltaUm=delta.tolist(), imageMeasuredStageDeltaUm=measured.tolist(),
                                        disagreementUm=float(np.linalg.norm(measured-delta))))
    write_json_atomic(args.output, result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
