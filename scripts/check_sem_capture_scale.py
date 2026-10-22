"""Four-shot, explicitly executed protocol/FOV diagnostic; no gun/EDS/Z changes."""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.cleanliness_core.ab_particle_calibration import move_verified, validate_field_metadata
from backend.cleanliness_core.atomic_json import write_json_atomic
from backend.cleanliness_core.optical_particle_analysis import load_calibration
from backend.cleanliness_core.optical_particle_task import ZemMicroscope
from backend.cleanliness_core.particle_refinement import RefinementSettings


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--center', type=float, nargs=2, required=True)
    p.add_argument('--reference-mag', type=int, required=True)
    p.add_argument('--test-mag', type=int, required=True)
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    if not args.execute:
        p.error('Requires explicit --execute and operator microscope availability')
    output = args.output.resolve()
    if output.exists():
        p.error('Use a new evidence directory')
    output.mkdir(parents=True)
    roi = load_calibration(ROOT / 'resources/conf/optical_particle_calibration.json')
    cfg = RefinementSettings()
    center = np.array(args.center)
    if not roi.target_inside_roi(tuple(center)) or not roi.target_inside_roi(tuple(center+cfg.approach_offset_um)):
        p.error('Diagnostic target outside configured ROI')
    logger.remove()
    logger.add(str(output/'device.log'), level='INFO', encoding='utf-8')
    scope = ZemMicroscope(roi.device['host'], roi.device['port'], {}, pixel_avg=7)
    audit = {'kind': 'four_frame_capture_scale_diagnostic', 'shots': [], 'gunStateChanged': False, 'acceptedTransform': False}
    start = connection = None
    def save():
        write_json_atomic(output/'manifest.json', audit)
    def configure(mag):
        scope.imaging = {'magnification': mag, 'imageShape': [1024,1024],
                         'fieldSizeUm': roi.imaging['fieldSizeUm']*roi.imaging['magnification']/mag}
        return scope.connect()
    try:
        connection = scope.connect()
        start = scope.get_position()
        audit.update(connection=connection, startStageUm=list(start), hvStatus=scope.get_hv_status())
        if str(audit['hvStatus']).rsplit('.',1)[-1].lower() != 'open':
            raise RuntimeError('Gun is not open; no gun change performed')
        move_verified(scope, center+cfg.approach_offset_um, cfg)
        move_verified(scope, center, cfg)
        for name, mag, protocol in [('reference_ng',args.reference_mag,'ng'),('test_ng',args.test_mag,'ng'),
                                     ('test_legacy',args.test_mag,'legacy'),('return_ng',args.reference_mag,'ng')]:
            state = configure(mag)
            before = np.array(scope.get_position())
            if np.linalg.norm(before-center) > cfg.stage_error_um:
                raise RuntimeError('Stage no longer at diagnostic center')
            path = output/(name+'.png')
            if protocol == 'ng':
                image, meta = scope.capture(path)
                validate_field_metadata(meta, scope.imaging['fieldSizeUm'])
            else:
                from zemrpc.zem.models import SnapArgs
                image = scope.api.device.snap_simple(SnapArgs(pixel_avg=7, frame_avg=1, size=scope._image_size_enum, save=False))
                if image is None or not cv2.imwrite(str(path), np.asarray(image)):
                    raise RuntimeError('Legacy capture did not return/save an image')
                meta = {'rawPath': str(path), 'shape': list(np.asarray(image).shape), 'note':'Legacy raw pixels; no inferred physical pixel scale'}
            after = np.array(scope.get_position())
            shot = {'name':name,'requestedMagnification':mag,'protocol':protocol,'connection':state,
                    'captureStageUm':after.tolist(),'capture':meta}
            audit['shots'].append(shot)
            save()
            print(f'{name}: captured shape={np.asarray(image).shape}', flush=True)
            if np.linalg.norm(after-before)>cfg.stage_error_um or np.linalg.norm(after-center)>cfg.stage_error_um:
                raise RuntimeError('Stage drift during protocol diagnostic')
        audit['status']='CAPTURED'
    except Exception as exc:
        audit.update(status='FAILED',error=str(exc))
    finally:
        cleanup = {}
        if start is not None:
            try:
                cleanup['stageRestore']=move_verified(scope,start,cfg)
            except Exception as exc:
                cleanup['stageRestoreError']=str(exc)
        if connection and connection.get('magnification'):
            try:
                cleanup['imagingRestore']=configure(connection['magnification'])
                scope.restore_video()
            except Exception as exc:
                cleanup['imagingRestoreError']=str(exc)
        audit['cleanup']=cleanup
        save()
        print(json.dumps({'status':audit.get('status'),'error':audit.get('error'),'cleanup':cleanup},ensure_ascii=False))


if __name__=='__main__':
    main()
