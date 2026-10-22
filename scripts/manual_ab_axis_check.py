"""Measure image/stage axes independently; never modify the AB transform."""
import json
import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.cleanliness_core.optical_particle_task import ZemMicroscope
from backend.cleanliness_core.atomic_json import write_json_atomic as dump
from scripts.capture_a2081_point import move_converged


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='axis')
    parser.add_argument('--center-x', type=float, default=0.)
    parser.add_argument('--center-y', type=float, default=1973.)
    args = parser.parse_args()
    out = ROOT / 'artifacts/manual_ab_sample2_20260908' / args.name
    out.mkdir(exist_ok=True)
    if (out / 'axis.json').exists(): raise FileExistsError(out)
    logger.remove(); logger.add(str(out / 'device.log'), encoding='utf-8')
    scope = ZemMicroscope('192.168.0.15', 61557,
        {'imageShape': [512, 512], 'magnification': 120, 'fieldSizeUm': 127000/120}, pixel_avg=4)
    result = {'shots': []}
    start = None
    try:
        result['connection'] = scope.connect()
        start = scope.get_position()
        result['startStageUm'] = start
        # A central sample region, well inside the existing safe ROI.
        center = np.array([args.center_x, args.center_y])
        if np.linalg.norm(center - [1013.605, 869.171]) + 300 > 25798.31:
            raise ValueError('Axis measurement region is outside the existing safe ROI')
        frames = {}
        for name, delta in [('o1', [0, 0]), ('xp', [100, 0]), ('xm', [-100, 0]), ('yp', [0, 100]), ('ym', [0, -100]), ('o2', [0, 0])]:
            print('axis ' + name, flush=True)
            target = center + delta
            move = move_converged(scope, tuple(target))
            arr, capture = scope.capture(out / (name + '.png'))
            result['shots'].append({'name': name, 'move': move, 'capture': capture, 'postCaptureUm': scope.get_position()})
            dump(out / 'progress.json', result)
            arr = cv2.GaussianBlur(arr.astype(np.float32), (0, 0), 1.5)
            frames[name] = arr - cv2.GaussianBlur(arr, (0, 0), 25)
        window = cv2.createHanningWindow((512, 512), cv2.CV_32F)
        shifts = {name: cv2.phaseCorrelate(frames['o1'].copy(), frames[name].copy(), window) for name in ['xp', 'xm', 'yp', 'ym', 'o2']}
        rows = {r['name']: r for r in result['shots']}
        ds = np.array([np.subtract(shifts['xp'][0], shifts['xm'][0]), np.subtract(shifts['yp'][0], shifts['ym'][0])]).T
        de = np.array([np.subtract(rows['xp']['postCaptureUm'], rows['xm']['postCaptureUm']), np.subtract(rows['yp']['postCaptureUm'], rows['ym']['postCaptureUm'])]).T
        g = -de @ np.linalg.inv(ds)
        result.update(shifts=shifts, pixelToStageMatrix=g.tolist(),
                      singularValuesUmPerPx=np.linalg.svd(g)[1].tolist(), returnErrorUm=float(np.linalg.norm(g @ shifts['o2'][0])))
        nominal = rows['o1']['capture']['pixelSizeUm']
        result['nominalPixelSizeUm'] = nominal
        result['valid'] = bool(min(s[1] for s in shifts.values()) > .15 and result['returnErrorUm'] < 5
                               and np.all(np.abs(np.linalg.svd(g)[1] / nominal - 1) < .15))
    except Exception as e:
        result.update(valid=False, error=str(e))
    finally:
        if start is not None:
            try: result['restore'] = move_converged(scope, start)
            except Exception as e: result['restoreError'] = str(e)
        try:
            scope.restore_video()
            result['finalStageUm'] = scope.get_position()
            result['hvStatus'] = str(scope.get_hv_status())
        except Exception as e: result['endStateError'] = str(e)
        dump(out / 'axis.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'shots'}, ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
