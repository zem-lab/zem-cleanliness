"""Direct SEM survey: plan -> capture -> segment -> automatic point EDS.

The navigation convention comes from zem_particle/tools/auto_capture.py.
This task has no optical input or optical identity/calibration requirements.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from .atomic_json import write_json_atomic
from .optical_particle_analysis import normalize_uint8
from .sem_spatial import ContourIndex


def stamp():
    return datetime.now().astimezone().isoformat()


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_plan(options, bounds):
    bounds = list(map(float, bounds))
    roi = dict(options.get('roi') or {})
    x0, y0, x1, y1 = [float(roi[k]) for k in ('x0', 'y0', 'x1', 'y1')]
    field = float(options.get('fieldUm', 1000))
    overlap = float(options.get('overlap', .1))
    size = int(options.get('imageSize', 1024))
    minimum = float(options.get('minLengthUm', 5))
    x_sign = options.get('pixelXAxisSign', -1)
    if isinstance(x_sign,bool) or x_sign not in {-1,1}:
        raise ValueError('SEM 像素 X 轴方向必须为 -1 或 1')
    if not all(math.isfinite(v) for v in [*bounds, x0, y0, x1, y1, field, overlap, minimum]):
        raise ValueError('区域与扫描参数必须是有限数值')
    if len(bounds) != 4 or not (bounds[0] <= x0 < x1 <= bounds[2] and bounds[1] <= y0 < y1 <= bounds[3]):
        raise ValueError('ROI 必须位于导航图范围内且面积大于零')
    if roi.get('shape') not in {'square', 'circle'}:
        raise ValueError('请选择正方形或圆形 ROI')
    if not math.isclose(x1-x0, y1-y0, rel_tol=1e-6):
        raise ValueError('ROI 的宽高必须相等')
    if field <= 0 or not 0 <= overlap <= .4 or size not in {512, 1024, 2048} or minimum < 0:
        raise ValueError('扫描参数无效：重叠率 0–40%，图像尺寸 512/1024/2048')
    count = max(1, math.ceil(max(0, x1-x0-field)/(field*(1-overlap)))+1)
    if count * count > 10000:
        raise ValueError('单任务最多 10000 个视野，请缩小区域或增大 FOV')
    # Keep the outside fields centered on the ROI edges, and evenly distribute
    # overlap. Field centers must remain inside the navigation envelope.
    centers = np.linspace(x0+field/2, x1-field/2, count) if count > 1 else [(x0+x1)/2]
    ys = np.linspace(y0+field/2, y1-field/2, count) if count > 1 else [(y0+y1)/2]
    tiles = []
    for row, y in enumerate(ys):
        columns = range(count) if row % 2 == 0 else reversed(range(count))
        for col in columns:
            x = float(centers[col])
            if roi['shape'] == 'circle':
                dx = max(abs(x-(x0+x1)/2)-field/2, 0)
                dy = max(abs(y-(y0+y1)/2)-field/2, 0)
                if math.hypot(dx, dy) > (x1-x0)/2:
                    continue
            tiles.append({'id': len(tiles)+1, 'row': row, 'col': col, 'x': x, 'y': float(y), 'status': 'PENDING'})
    return {'roi': {**roi, 'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1}, 'fieldUm': field,
            'overlap': overlap, 'imageSize': size, 'minLengthUm': minimum, 'rows': count, 'pixelXAxisSign':x_sign,
            'areaMm2': (x1-x0)*(y1-y0)/1e6*(math.pi/4 if roi['shape']=='circle' else 1),
            'tiles': tiles, 'tileCount': len(tiles)}


def inside_roi(point, roi):
    x, y = point
    if not (roi['x0'] <= x <= roi['x1'] and roi['y0'] <= y <= roi['y1']):
        return False
    return roi['shape'] != 'circle' or math.hypot(x-(roi['x0']+roi['x1'])/2, y-(roi['y0']+roi['y1'])/2) <= (roi['x1']-roi['x0'])/2


def segment(image, model_path):
    """Reuse the installed YOLO decoder without optical size ranking/filtering.

    Unlike optical matching, a survey must retain large and border particles
    for explicit acceptance/rejection and must not silently truncate results.
    """
    from .cleanliness_detection import _yolo_session, _nms_indices
    gray = normalize_uint8(image)
    h, w = gray.shape
    size = 1024
    session = _yolo_session(Path(model_path))
    tensor = cv2.cvtColor(cv2.resize(gray, (size, size)), cv2.COLOR_GRAY2RGB).transpose(2, 0, 1)[None].astype(np.float32)/255
    outputs = session.run(None, {session.get_inputs()[0].name: tensor})
    prediction, prototypes = np.asarray(outputs[0])[0], np.asarray(outputs[1])[0]
    if prediction.shape[0] < prediction.shape[1]:
        prediction = prediction.T
    channels, ph, pw = prototypes.shape
    classes = prediction.shape[1]-4-channels
    if classes < 1:
        raise ValueError('颗粒模型输出不支持实例分割')
    scores = prediction[:, 4:4+classes].max(axis=1)
    prediction, scores = prediction[scores >= .25], scores[scores >= .25]
    if not len(prediction):
        return []
    boxes = np.concatenate((prediction[:, :2]-prediction[:, 2:4]/2, prediction[:, :2]+prediction[:, 2:4]/2), axis=1).clip(0, size)
    keep = _nms_indices(boxes, scores, .45, 1001)
    if len(keep) > 1000:
        raise ValueError('单视野超过 1000 个颗粒，分割不完整；请减小 FOV')
    masks = 1/(1+np.exp(-np.clip(prediction[keep, 4+classes:] @ prototypes.reshape(channels, -1), -30, 30)))
    result = []
    for index, mask in zip(keep, masks.reshape(-1, ph, pw), strict=True):
        x0, y0, x1, y1 = boxes[index]*[pw/size, ph/size, pw/size, ph/size]
        mask *= ((np.arange(pw)[None, :] >= x0) & (np.arange(pw)[None, :] < x1) & (np.arange(ph)[:, None] >= y0) & (np.arange(ph)[:, None] < y1))
        binary = (cv2.resize(mask, (w, h)) >= .5).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea).reshape(-1, 2).tolist() if contours else []
        result.append({'points': contour, 'confidence': float(scores[index]), 'fragmented': len(contours) != 1})
    return result


def evaluate_shapes(shapes, shape, pixel_size, center, plan, tile_id):
    """Resolve all competing masks BEFORE any EDS call, in the current frame."""
    h, w = shape[:2]
    if not math.isfinite(pixel_size) or pixel_size <= 0 or not np.isfinite(center).all():
        raise ValueError('采图坐标或像素尺寸无效')
    records = []
    for item in shapes:
        record = {'contour': item.get('points', []), 'confidence': item.get('confidence'), 'status': 'REJECTED'}
        points = np.asarray(record['contour'], dtype=np.float32)
        mask = np.zeros((h, w), np.uint8)
        records.append(record)
        if points.ndim != 2 or points.shape[1:] != (2,) or len(points) < 3 or not np.isfinite(points).all():
            record['reason'] = '轮廓不足或坐标无效'
            continue
        if (points < 0).any() or (points[:, 0] >= w).any() or (points[:, 1] >= h).any():
            record['reason'] = '轮廓坐标超出原图'
            continue
        cv2.fillPoly(mask, [points.astype(np.int32)], 1)
        area = float(cv2.contourArea(points))
        if area < 4 or item.get('fragmented'):
            record['reason'] = '轮廓面积不足或分割存在多个不连通区域'
            continue
        moments = cv2.moments(points)
        if moments['m00'] <= 0:
            record['reason'] = '无有效颗粒面积'
            continue
        px, py = moments['m10']/moments['m00'], moments['m01']/moments['m00']
        # Zem20F: positive stage X moves specimen features right in the SEM
        # image, hence a pixel's world X offset has the opposite sign.
        x_sign=plan.get('pixelXAxisSign',-1)
        stage = [center[0]+x_sign*(px-w/2)*pixel_size, center[1]+(py-h/2)*pixel_size]
        axes = cv2.minAreaRect(points)[1]
        record.update(stageUm=stage, lengthUm=max(axes)*pixel_size, widthUm=min(axes)*pixel_size,
                      areaUm2=area*pixel_size**2, equivalentDiameterUm=math.sqrt(4*area/math.pi)*pixel_size,
                      stageContourUm=((points-[w/2, h/2])*[x_sign*pixel_size,pixel_size]+center).tolist())
        if not inside_roi(stage, plan['roi']):
            record['reason'] = '颗粒中心位于 ROI 外'
        elif min(points[:, 0]) <= 1 or min(points[:, 1]) <= 1 or max(points[:, 0]) >= w-2 or max(points[:, 1]) >= h-2:
            record['reason'] = '视野边缘轮廓截断，待补拍'
        elif min(plan['tiles'], key=lambda t: ((t['x']-stage[0])**2+(t['y']-stage[1])**2, t['id']))['id'] != tile_id:
            record['reason'] = '重叠区域颗粒归属相邻视野'
        elif record['lengthUm'] < plan['minLengthUm']:
            record['reason'] = '低于最小颗粒长度'
        else:
            distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
            _, depth, _, safe = cv2.minMaxLoc(distance)
            if depth < 2:
                record['reason'] = '轮廓内没有可靠采谱点'
            else:
                record.update(status='PENDING', safePoint=list(safe))
    # Any mask intersection is ambiguous. Never merge adjacent particles merely
    # because their centers are close, or let ordering choose an EDS target.
    contour_index = ContourIndex(64)
    for i, record in enumerate(records):
        contour_index.add(record['contour'], (i, record))
    for i, record in enumerate(records):
        if record['status'] == 'PENDING' and any(contours_overlap(record['contour'], other['contour']) for j, other in contour_index.candidates(record['contour']) if i != j):
            record.update(status='REJECTED', reason='不同颗粒轮廓重叠，采谱归属不明确')
    return records


def contours_overlap(a, b, scale=1):
    """Rasterize only the shared bounding box; avoid N full-frame masks."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if any(p.ndim != 2 or p.shape[1:] != (2,) or len(p) < 3 or not np.isfinite(p).all() for p in (a, b)):
        return False
    lower = np.maximum(a.min(axis=0), b.min(axis=0))
    upper = np.minimum(a.max(axis=0), b.max(axis=0))
    if np.any(upper < lower):
        return False
    width, height = np.ceil((upper-lower)/scale).astype(int)+3
    if max(width, height) > 4096:
        return True  # malformed or unreasonably large geometry: reject conservatively
    masks = []
    for points in (a, b):
        mask = np.zeros((height, width), np.uint8)
        cv2.fillPoly(mask, [np.rint((points-lower)/scale+1).astype(np.int32)], 1)
        masks.append(mask)
    return bool(np.any(masks[0] & masks[1]))


def write_reports(directory, manifest):
    from .sem_direct_report import build_sem_reports
    return build_sem_reports(directory, manifest)


def verify_recovery_frame(reference, current, old_records, fresh_records):
    """Validate a recaptured frame before retrying a known failed EDS job.

    Never shifts/mutates original coordinates or trusts a high score alone.
    Both whole-frame agreement and a unique complete contour at the exact
    original sampling point are required. Completed particles are not sampled.
    """
    if reference.shape != current.shape:
        raise ValueError('恢复采谱：图像尺寸已变化')
    a,b=normalize_uint8(reference),normalize_uint8(current)
    if float(a.std())<1 or float(b.std())<1:
        raise ValueError('恢复采谱：图像不足以验证原位置')
    score=float(cv2.matchTemplate(b,a,cv2.TM_CCOEFF_NORMED)[0,0])
    if not math.isfinite(score) or score<.85:
        raise ValueError(f'恢复采谱：当前图像与原视野不一致（{score:.3f}）')
    h,w=a.shape
    def mask(points):
        pts=np.asarray(points,np.float32)
        if pts.ndim!=2 or pts.shape[1:]!=(2,) or len(pts)<3 or not np.isfinite(pts).all() or (pts<0).any() or (pts[:,0]>=w).any() or (pts[:,1]>=h).any():
            raise ValueError('恢复采谱：原轮廓坐标无效')
        m=np.zeros((h,w),np.uint8)
        cv2.fillPoly(m,[pts.astype(np.int32)],1)
        return m
    evidence={}
    for old in old_records:
        if old['status'] not in {'FAILED','PENDING'}: continue
        original=mask(old['contour'])
        matches=[]
        for fresh in fresh_records:
            if fresh['status']!='PENDING': continue
            candidate=mask(fresh['contour'])
            union=np.count_nonzero(original|candidate)
            iou=np.count_nonzero(original&candidate)/union if union else 0
            if iou>=.8:
                matches.append((fresh,iou))
        if len(matches)!=1:
            raise ValueError(f"恢复采谱：颗粒 {old['sequence']} 没有唯一匹配轮廓")
        fresh,iou=matches[0]
        point=old.get('safePoint')
        if not point or len(point)!=2 or not np.isfinite(point).all() or not (0<=point[0]<w and 0<=point[1]<h):
            raise ValueError('恢复采谱：采谱点坐标无效')
        depth=cv2.pointPolygonTest(np.asarray(fresh['contour'],np.float32),tuple(map(float,point)),True)
        if depth<2:
            raise ValueError('恢复采谱：原采谱点不在可靠颗粒内部')
        evidence[old['sequence']]={'imageCorrelation':score,'contourIoU':iou,'interiorDistancePx':depth}
    return evidence


def run_task(directory, *, microscope=None, analyzer=None, segmenter=None, progress=lambda *a: None, should_stop=lambda: False):
    from .optical_particle_task import ZemMicroscope
    from .cleanliness_task import _move_stage_checked, _restore_and_maybe_gun_off
    from .eds.client_v2 import EdsV2Error
    directory = Path(directory)
    from .sem_task_store import load_sem_task, SemTaskStore
    manifest = load_sem_task(directory)
    options, plan = manifest['request'], manifest['plan']
    retry_count = options.setdefault('edsRetryCount', 2)
    if isinstance(retry_count,bool) or not isinstance(retry_count,int) or not 0 <= retry_count <= 2:
        raise ValueError('EDS 重试次数必须在 0 到 2 之间')
    verified = build_plan(plan, manifest['navigation']['boundsUm'])
    if [(t['id'], t['x'], t['y']) for t in plan['tiles']] != [(t['id'], t['x'], t['y']) for t in verified['tiles']]:
        raise ValueError('扫描计划坐标不一致')
    if manifest['status'] not in {'DRAFT', 'STOPPED', 'FAILED', 'RUNNING'}:
        raise ValueError('任务已经结束')
    imaging = {'fieldSizeUm': plan['fieldUm'], 'imageShape': [plan['imageSize']]*2, 'responsiveStagePolling': True,
               'magnification': round(manifest['navigation']['magnification']*manifest['navigation']['fieldUm']/plan['fieldUm'])}
    microscope = microscope or ZemMicroscope(options['microscopeHost'], options['microscopePort'], imaging, pixel_avg=options['captureLevel'])
    remote = manifest['model'].get('backend', 'yolo') == 'sam3'
    if manifest['model'].get('backend', 'yolo') not in {'yolo', 'sam3'}:
        raise ValueError('不支持的颗粒分割模型')
    if segmenter is None and remote:
        from .sem_segmentation import segment_sam3
        def segmenter(image, unused_path):
            audit = {}
            try:
                return segment_sam3(image, manifest['model'], audit=audit)
            finally:
                if audit:
                    write_json_atomic(raw_path.with_suffix('.sam3.json'), audit)
    segmenter = segmenter or segment
    start = None
    connected = False
    run_started = time.perf_counter()
    timing = {'startedAt':stamp(), 'stages':{}, 'events':[], 'persistenceSeconds':0.0}
    manifest.setdefault('timingRuns', []).append(timing)
    manifest['reports'] = {}
    manifest['reportState'] = 'DEFERRED'
    manifest.pop('reportError', None)
    store = SemTaskStore(directory, manifest)
    active_tile = None
    previous_by_tile = defaultdict(list)
    for particle in manifest['particles']:
        previous_by_tile[particle['tileId']].append(particle)
    analyzed_index = ContourIndex(plan['fieldUm'])
    indexed_sequences = set()
    def index_analyzed(particle):
        if particle['sequence'] not in indexed_sequences and particle['status'] in {'SUCCEEDED','MEASURED','ACQUIRING'}:
            analyzed_index.add(particle.get('stageContourUm',[]),particle)
            indexed_sequences.add(particle['sequence'])
    for particle in manifest['particles']:
        index_analyzed(particle)
    def measured(name, operation, *args, owner=None, **kwargs):
        begun, wall = time.perf_counter(), stamp()
        outcome = 'SUCCEEDED'
        try:
            return operation(*args, **kwargs)
        except Exception:
            outcome = 'FAILED'
            raise
        finally:
            seconds = time.perf_counter()-begun
            timing['stages'][name] = timing['stages'].get(name,0)+seconds
            event = {'stage':name,'startedAt':wall,'seconds':seconds,'outcome':outcome}
            if owner is not None:
                owner.setdefault('timing',{})[name] = owner.get('timing',{}).get(name,0)+seconds
                event.update(tileId=owner.get('tileId',owner.get('id')),sequence=owner.get('sequence'))
            timing['events'].append(event)
    def save(*, checkpoint=False, particles=()):
        saved_at = time.perf_counter()
        manifest['updatedAt'] = stamp()
        store.save(manifest, active_tile, dirty_particles=particles)
        if checkpoint:
            store.checkpoint(manifest)
        timing['persistenceSeconds'] += time.perf_counter()-saved_at
    try:
        if remote:
            from .sem_segmentation import check_sam3
            manifest['segmentationService'] = measured('segmentationServiceCheck', check_sam3, manifest['model'])
        elif fingerprint(manifest['model']['path']) != manifest['model']['sha256']:
            raise ValueError('颗粒模型已变化，请重新创建任务')
        if should_stop():
            manifest['status'] = 'STOPPED'
            return manifest
        measured('connect', microscope.connect)
        connected = True
        start = microscope.get_position()
        microscope.get_hv_status()
        if options['edsEnabled'] and analyzer is None:
            from .eds.client_v2 import EdsV2Client
            from .energy_spectrum import process_energy_spectrum_particle
            client = EdsV2Client(options['edsHost'], options['edsPort'])
            analyzer = SimpleNamespace(close=client.close, analyze=lambda p, image: process_energy_spectrum_particle(
                client, {'label': str(p['sequence']), 'points': p['contour']}, str(image),
                options['microscopeHost'], options['microscopePort'], target_point=tuple(p['safePoint'])))
        manifest.update(status='RUNNING', error=None)
        save()
        for tile in plan['tiles']:
            active_tile = tile
            if tile['status'] == 'COMPLETED':
                continue
            if should_stop():
                manifest['status'] = 'STOPPED'
                break
            previous_records = previous_by_tile[tile['id']]
            recovering = bool(previous_records) and all(
                p['status'] in {'SUCCEEDED','MEASURED','REJECTED','PENDING'} or
                (p['status']=='FAILED' and p.get('failureStage')=='eds' and
                 (p.get('edsAttempts') or [{}])[-1].get('error',{}).get('terminal') is True)
                for p in previous_records)
            if previous_records and not recovering:
                tile.update(status='INCOMPLETE', error='Unknown acquisition state; no automatic replay')
                for p in previous_records:
                    if p['status'] in {'PENDING','ACQUIRING'}:
                        p.update(status='REJECTED', reason='Interrupted frame; unknown EDS completion')
                save(particles=previous_records)
                continue
            if recovering and all(p['status'] in {'SUCCEEDED','MEASURED','REJECTED'} for p in previous_records):
                tile['status']='COMPLETED'
                save()
                continue
            tile['status'] = 'RUNNING'
            tile['startedAt'] = stamp()
            tile_started = time.perf_counter()
            save()
            progress(tile['id']-1, plan['tileCount'], f"视野 {tile['id']}：移动并拍图")
            if isinstance(microscope, ZemMicroscope):
                microscope.api.sync_scan_field()
                rotation = float(microscope.api.device.m.scan_info.rot_degree)
                if not math.isfinite(rotation) or abs((rotation+180) % 360-180) > .1:
                    raise ValueError('扫描旋转不为 0°，未执行采图或采谱')
            move = measured('move', _move_stage_checked, microscope, (tile['x'], tile['y']), owner=tile)
            tile['moveEvidence'] = move
            actual = [float(move['after'][k]) for k in ('x', 'y')]
            if not np.isfinite(actual).all():
                raise ValueError('样品台坐标回读无效')
            if recovering:
                recovery_index=1
                while (directory/'tiles'/f"{tile['id']:05d}.verify-{recovery_index}.png").exists():
                    recovery_index+=1
                raw_path=directory/'tiles'/f"{tile['id']:05d}.verify-{recovery_index}.png"
            else:
                raw_path=directory/'tiles'/f"{tile['id']:05d}.png"
            progress(tile['id']-1,plan['tileCount'],f"视野 {tile['id']}：拍图")
            image, metadata = measured('capture', microscope.capture, raw_path, owner=tile)
            scale = float(metadata['pixelSizeUm'])
            if tuple(image.shape[:2]) != (plan['imageSize'],)*2 or not math.isfinite(scale) or not math.isclose(scale*image.shape[1], plan['fieldUm'], rel_tol=.002, abs_tol=.5):
                raise ValueError('实际采图尺寸或 FOV 与扫描计划不一致，未执行采谱')
            if recovering:
                if (not tile.get('raw') or fingerprint(directory/tile['raw'])!=tile.get('sha256') or
                        math.dist(actual,tile['actualStageUm'])>2 or
                        not math.isclose(scale,float(tile['capture']['pixelSizeUm']),rel_tol=.002)):
                    raise ValueError('Recovery source/frame coordinates changed')
                shapes=measured('segmentation',segmenter,image,manifest['model'].get('path'),owner=tile)
                fresh=measured('candidateReview',evaluate_shapes,shapes,image.shape,scale,actual,plan,tile['id'],owner=tile)
                reference=cv2.imread(str(directory/tile['raw']),cv2.IMREAD_UNCHANGED)
                write_json_atomic(raw_path.with_suffix('.segmentation.json'),{'sourceSha256':fingerprint(raw_path),'shapes':shapes,'verificationStatus':'PENDING'})
                # Pending candidates have never been sampled: survey them from
                # the NEW frame under the normal segmentation/acceptance gates.
                # Only failed acquisitions require identity correspondence.
                # All earlier successful contours are exclusion regions.
                failed_records=[p for p in previous_records if p['status']=='FAILED']
                proof=verify_recovery_frame(reference,image,failed_records,fresh)
                capture={'raw':raw_path.relative_to(directory).as_posix(),'sha256':fingerprint(raw_path),'capture':metadata,'actualStageUm':actual,'proof':proof}
                tile.setdefault('recoveryCaptures',[]).append(capture)
                write_json_atomic(raw_path.with_suffix('.segmentation.json'),{'sourceSha256':capture['sha256'],'shapes':shapes,'proof':proof})
                records=list(previous_records)
                for p in previous_records:
                    if p['status']=='FAILED':
                        p.setdefault('recoveryEvidence',[]).append({'capture':capture['raw'],**proof[p['sequence']]})
                        p['status']='PENDING'
                    elif p['status']=='PENDING':
                        p.update(status='REJECTED',reason='恢复时以重拍视野重新分割，原未采候选仅保留审计',supersededByCapture=capture['raw'])
                exclusions=[p for p in previous_records if p['status'] in {'SUCCEEDED','MEASURED','PENDING'}]
                for candidate in fresh:
                    if candidate['status']!='PENDING':continue
                    if any(contours_overlap(candidate['contour'],old['contour']) for old in exclusions):continue
                    if any(old['tileId']!=tile['id'] and old['status'] in {'SUCCEEDED','MEASURED'}
                           and contours_overlap(candidate['stageContourUm'],old.get('stageContourUm',[]),scale)
                           for old in analyzed_index.candidates(candidate['stageContourUm'])):continue
                    candidate.update(sequence=len(manifest['particles'])+1,tileId=tile['id'],
                                     sourceCapture=capture['raw'],recoveryDetection=True)
                    manifest['particles'].append(candidate)
                    records.append(candidate)
                save(particles=records)
            else:
                tile.update(capture=metadata, actualStageUm=actual, raw=raw_path.relative_to(directory).as_posix(), sha256=fingerprint(raw_path))
                save()
                progress(tile['id']-1,plan['tileCount'],f"视野 {tile['id']}：分割")
                shapes = measured('segmentation', segmenter, image, manifest['model'].get('path'), owner=tile)
                write_json_atomic(raw_path.with_suffix('.segmentation.json'), {'sourceSha256': tile['sha256'], 'model': manifest['model'], 'shapes': shapes})
                records = measured('candidateReview', evaluate_shapes, shapes, image.shape, scale, actual, plan, tile['id'], owner=tile)
                for p in records:
                    if p['status'] == 'PENDING' and any(
                        old['status'] in {'SUCCEEDED', 'MEASURED', 'ACQUIRING'}
                        and contours_overlap(p['stageContourUm'], old.get('stageContourUm', []), scale)
                        for old in analyzed_index.candidates(p['stageContourUm'])
                    ):
                        p.update(status='REJECTED', reason='与已分析颗粒区域重叠，拒绝重复或归属不明确的采谱')
                overlay = cv2.cvtColor(normalize_uint8(image), cv2.COLOR_GRAY2BGR)
                for p in records:
                    p.update(sequence=len(manifest['particles'])+1, tileId=tile['id'])
                    manifest['particles'].append(p)
                    points = np.asarray(p['contour'], np.float32)
                    if points.ndim == 2 and len(points) >= 3 and np.isfinite(points).all():
                        cv2.polylines(overlay, [points.astype(np.int32)], True, (60, 210, 90) if p['status']=='PENDING' else (70, 140, 240), 1)
                preview = raw_path.with_name(raw_path.stem+'.overlay.jpg')
                cv2.imwrite(str(preview), overlay)
                tile['preview'] = preview.relative_to(directory).as_posix()
                save(particles=records)
            for index, p in enumerate(records):
                if p['status'] != 'PENDING':
                    continue
                progress(tile['id']-1, plan['tileCount'], f"视野 {tile['id']}：颗粒 {index+1}/{len(records)} 自动采谱")
                if not options['edsEnabled']:
                    p['status'] = 'MEASURED'
                else:
                    for attempt_index in range(retry_count+1):
                        try:
                            current = microscope.get_position()
                            if not np.isfinite(current).all() or math.dist(current, actual) > 2:
                                raise ValueError('采谱前样品台位置已改变，拒绝使用旧图像坐标')
                            if attempt_index and isinstance(microscope,ZemMicroscope):
                                microscope.api.sync_stage_and_scan_field()
                                scan = microscope.api.device.m.scan_info
                                if (not math.isclose(float(scan.scan_um),plan['fieldUm'],rel_tol=.002,abs_tol=.5)
                                        or abs((float(scan.rot_degree)+180)%360-180)>.1):
                                    raise ValueError('重试前视野参数已改变，拒绝使用旧图像坐标')
                        except Exception as error:
                            p.update(status='FAILED',failureStage='eds_target_check',reason=str(error))
                            tile['status']='FAILED'
                            save(particles=[p])
                            raise
                        entry={'attempt':len(p.get('edsAttempts',[]))+1,'startedAt':stamp(),'status':'RUNNING'}
                        p.setdefault('edsAttempts',[]).append(entry)
                        p['status'] = 'ACQUIRING'
                        save(particles=[p])
                        try:
                            p['spectrum'] = measured('eds', analyzer.analyze, p, raw_path, owner=p)
                        except EdsV2Error as error:
                            entry.update(status='FAILED',endedAt=stamp(),error=error.to_dict())
                            p.update(status='FAILED',failureStage='eds',reason=str(error))
                            # A known terminal failure may be retried in THIS
                            # unchanged frame. Unknown/timeout/cancelled jobs
                            # must not be replayed and compete for the device.
                            if not error.terminal or attempt_index>=retry_count or should_stop():
                                tile['status']='FAILED'
                                save(particles=[p])
                                raise
                            p['status']='RETRYING'
                            save(particles=[p])
                            progress(tile['id']-1,plan['tileCount'],f"视野 {tile['id']}：颗粒 {p['sequence']} 采谱失败，重试 {attempt_index+1}/{retry_count}")
                            time.sleep(.5)
                            continue
                        except Exception as error:
                            entry.update(status='FAILED',endedAt=stamp(),error={'message':str(error),'terminal':False})
                            p.update(status='FAILED',failureStage='eds',reason=str(error))
                            tile['status']='FAILED'
                            save(particles=[p])
                            raise
                        entry.update(status='SUCCEEDED',endedAt=stamp())
                        p.update(status='SUCCEEDED')
                        p.pop('reason',None)
                        p.pop('failureStage',None)
                        break
                save(particles=[p])
                index_analyzed(p)
            tile['status'] = 'COMPLETED'
            tile['completedAt'] = stamp()
            tile['wallSeconds'] = time.perf_counter()-tile_started
            save(checkpoint=tile['id'] % 25 == 0)
        else:
            manifest['status'] = 'COMPLETED_WITH_FAILURES' if any(t['status'] != 'COMPLETED' for t in plan['tiles']) else 'COMPLETED'
        if manifest['status'].startswith('COMPLETED'):
            manifest['completedAt'] = stamp()
    except Exception as error:
        if 'tile' in locals() and tile.get('status')=='RUNNING':
            tile.update(status='FAILED',error=str(error))
        manifest.update(status='FAILED', error=str(error))
        raise
    finally:
        if connected:
            manifest['cleanup'] = measured('cleanup',_restore_and_maybe_gun_off,microscope,start,restore_position=options['restoreStartPosition'],gun_off=options['gunOffAfterTask'])
        if analyzer is not None:
            try:
                analyzer.close()
            except Exception as error:
                manifest['edsCloseError'] = str(error)
        try:
            save(checkpoint=True)
            if manifest['status'].startswith('COMPLETED'):
                try:
                    measured('report',write_reports,directory,manifest)
                    manifest['reportState'] = 'READY'
                except Exception as error:
                    manifest['reportState'] = 'FAILED'
                    manifest['reportError'] = str(error)
                    raise
            else:
                manifest['reportState'] = 'DEFERRED'
        finally:
            timing.update(endedAt=stamp(),wallSeconds=time.perf_counter()-run_started,status=manifest['status'])
            from .sem_timing_report import write_timing_report
            try:
                write_timing_report(directory,manifest)
            finally:
                try:
                    save(checkpoint=True)
                finally:
                    store.close()
    return manifest
