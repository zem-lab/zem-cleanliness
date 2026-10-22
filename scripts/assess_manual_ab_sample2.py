"""Offline shape comparison of manual-AB predictions; no fitting or hardware calls."""
import base64
import json
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/manual_ab_sample2_20260908'


def main():
    task = OUT / 'sample2-large-particles'
    manifest = json.loads((task / 'task.json').read_text(encoding='utf-8'))
    particles = {p['sequence']: p for p in json.loads(Path(manifest['input']['path']).read_text(encoding='utf-8-sig'))['particles']}
    axis_path = OUT / 'axis-particle/axis.json'
    axis = json.loads(axis_path.read_text(encoding='utf-8')) if axis_path.exists() else {}
    rows = []
    records = manifest['calibration']['abPointResult'].get('semRefinement', {}).get('validation', [])
    if not records:
        # Supports read-only inspection while acquisition is still running.
        records = [{'sequence': int(p.stem.split('.')[0]), 'images': {'semDisplayImage': str(p.relative_to(task))}, 'pixelSizeUm': 127000/120/512}
                   for p in sorted((task / 'ab_sem/validation').glob('*.u8.png'))]
    for record in records:
        if record.get('error'):
            rows.append({**record, 'assessment': 'CAPTURE_FAILED'}); continue
        p = particles[record['sequence']]
        im = cv2.imread(str(task / record['images']['semDisplayImage']), 0).astype(np.float32)
        mask = cv2.imdecode(np.frombuffer(base64.b64decode(p['maskDataBase64']), np.uint8), 0)
        mask = (mask > 127).astype(np.float32)
        optical_px = float(np.sqrt(p['particleInfo']['area']/mask.sum()))
        px = record.get('pixelSizeUm', 127000/120/512)
        high = cv2.GaussianBlur(im, (0, 0), .8) - cv2.GaussianBlur(im, (0, 0), 25)
        flat = np.clip(high / np.sqrt(cv2.GaussianBlur(high**2, (0, 0), 25) + 1), -2, 8)
        best = None
        for angle in range(-180, 180, 5):
            for factor in (.8, 1., 1.2):
                scale = optical_px / px * factor
                h, w = mask.shape
                n = int(np.ceil(np.hypot(h, w)*scale)) + 20
                if n >= min(im.shape): continue
                warp = cv2.getRotationMatrix2D(((w-1)/2, (h-1)/2), angle, scale)
                warp[:, 2] += [(n-w)/2, (n-h)/2]
                templ = cv2.GaussianBlur(cv2.warpAffine(mask, warp, (n, n)), (0, 0), .7)
                scores = cv2.matchTemplate(flat, templ, cv2.TM_CCOEFF_NORMED)
                _, score, _, loc = cv2.minMaxLoc(scores)
                if best is None or score > best['score']:
                    best = {'score': score, 'angleDeg': angle, 'scaleFactor': factor,
                            'centerPixel': [loc[0]+(n-1)/2, loc[1]+(n-1)/2]}
        if best is None:
            rows.append({**record, 'assessment': 'TEMPLATE_OUTSIDE_FIELD'}); continue
        delta = np.array(best['centerPixel']) - np.array(im.shape[::-1])/2
        best['offsetPixel'] = delta.tolist()
        best['nominalOffsetUm'] = float(np.linalg.norm(delta)*px)
        if axis.get('valid'):
            best['measuredOffsetUm'] = float(np.linalg.norm(np.array(axis['pixelToStageMatrix']) @ delta))
        best['assessment'] = 'CANDIDATE_REQUIRES_VISUAL_REVIEW' if best['score'] >= .5 else 'UNVERIFIED'
        display = cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        cv2.drawMarker(display, (im.shape[1]//2, im.shape[0]//2), (0, 255, 0), cv2.MARKER_CROSS, 30, 1)
        cv2.circle(display, tuple(np.rint(best['centerPixel']).astype(int)), max(20, round(p['particleInfo']['length']/px*.6)), (0, 0, 255), 2)
        cv2.putText(display, f"#{p['sequence']} score={best['score']:.3f}", (10, 28), 0, .6, (0, 255, 255), 1)
        path = OUT / f"comparison-{p['sequence']:04d}.png"
        cv2.imwrite(str(path), display)
        rows.append({**record, **best, 'comparisonImage': path.name})
        print(p['sequence'], round(best['score'], 3), best['angleDeg'], round(best['nominalOffsetUm'], 1), flush=True)
    result = {'method': 'Offline mask correlation, no refitting; scores are not probabilities',
              'axisValid': axis.get('valid', False), 'rows': rows}
    (OUT / 'assessment.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__': main()
