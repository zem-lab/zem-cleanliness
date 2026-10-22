"""Remote SAM3 adapter for direct SEM surveys; fail closed, no YOLO fallback."""
import base64
import hashlib
import json
import math
import struct
import urllib.request
from urllib.parse import urlsplit

import cv2
import numpy as np

from .optical_particle_analysis import normalize_uint8

DEFAULT_TEXTS = ['white particle', 'bright particle', 'bright white blob', 'flake', 'fiber', 'debris']
DEFAULT_URL = 'http://192.168.0.42:8765'
MAX_RESPONSE_BYTES = 64 * 1024 * 1024


def sam3_config(options):
    url = str(options.get('url', DEFAULT_URL)).strip().rstrip('/')
    parsed = urlsplit(url)
    if parsed.scheme not in {'http', 'https'} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError('SAM3 服务地址必须是有效的 HTTP/HTTPS 地址')
    texts = options.get('texts', DEFAULT_TEXTS)
    if not isinstance(texts, list) or not 1 <= len(texts) <= 20 or any(not isinstance(t,str) or not t.strip() or len(t)>200 for t in texts):
        raise ValueError('SAM3 提示词必须为 1–20 个非空文本')
    score = float(options.get('scoreThreshold', .1))
    if not math.isfinite(score) or not 0 < score <= 1:
        raise ValueError('SAM3 置信度必须大于 0 且不超过 1')
    return {'backend':'sam3', 'url':url, 'texts':[t.strip() for t in texts],
            'scoreThreshold':score, 'maskThreshold':.5, 'maxDetections':1000, 'dedupIoU':.8}


def request_json(url, payload=None):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(url, data=json.dumps(payload).encode() if payload is not None else None,
                                 headers={'Content-Type':'application/json'})
    try:
        with opener.open(req, timeout=120 if payload is not None else 10) as response:
            data=response.read(MAX_RESPONSE_BYTES+1)
            if len(data)>MAX_RESPONSE_BYTES:
                raise ValueError('SAM3 返回超过 64 MB，无法完整处理；请减小 FOV')
            result = json.loads(data)
    except Exception as error:
        raise RuntimeError(f'SAM3 远程请求失败：{error}') from error
    if not isinstance(result, dict) or result.get('error'):
        raise ValueError(f'SAM3 返回错误：{result}')
    return result


def check_sam3(model):
    config = sam3_config(model)
    health = request_json(config['url']+'/health')
    if health.get('ok') is not True or str(health.get('model','')).lower() != 'sam3':
        raise ValueError('SAM3 服务未就绪或模型不匹配')
    return health


def decode_detections(detections, shape, config):
    if not isinstance(detections,list) or len(detections)>=config['maxDetections']:
        raise ValueError('SAM3 返回格式错误或检测数量达到上限，不能确认分割完整')
    items=[]
    for index, det in enumerate(detections):
        score=float(det['score'])
        if not math.isfinite(score) or not config['scoreThreshold'] <= score <= 1:
            raise ValueError('SAM3 返回无效置信度')
        encoded=base64.b64decode(det['mask_png_b64'],validate=True)
        # Check PNG dimensions before allocating a potentially huge image.
        if (len(encoded)<24 or encoded[:8]!=b'\x89PNG\r\n\x1a\n' or encoded[12:16]!=b'IHDR'
                or struct.unpack('>II',encoded[16:24])!=(shape[1],shape[0])):
            raise ValueError('SAM3 掩码尺寸或 PNG 格式与 SEM 原图不一致')
        mask=cv2.imdecode(np.frombuffer(encoded,np.uint8),cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != tuple(shape):
            raise ValueError('SAM3 掩码尺寸与 SEM 原图不一致')
        _,mask=cv2.threshold(mask,127,1,cv2.THRESH_BINARY)
        bounds=cv2.boundingRect(mask)
        x,y,w,h=bounds
        crop=mask[y:y+h,x:x+w].copy()
        contours,_=(cv2.findContours(crop,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE,offset=(x,y))
                    if w and h else ([],None))
        points=max(contours,key=cv2.contourArea).reshape(-1,2).tolist() if contours else []
        item={'points':points,'confidence':score,'fragmented':len(contours)!=1,
              'prompt':det.get('text',det.get('label')), 'sourceIndex':index,'duplicateIndices':[]}
        # Retain only the occupied crop. Exact areas are computed once; most
        # SEM particles occupy a tiny fraction of the original image.
        items.append((item,crop,bounds,int(cv2.countNonZero(crop)) if crop.size else 0))
    kept=[]
    for item,mask,bounds,area in sorted(items,key=lambda x:-x[0]['confidence']):
        duplicate=None
        for old,other,other_bounds,other_area in kept:
            if item['fragmented'] or old['fragmented'] or len(item['points'])<3 or len(old['points'])<3:
                continue
            x,y,w,h=bounds; ox,oy,ow,oh=other_bounds
            left,top,right,bottom=max(x,ox),max(y,oy),min(x+w,ox+ow),min(y+h,oy+oh)
            if right<=left or bottom<=top:
                continue
            intersection=int(np.count_nonzero(mask[top-y:bottom-y,left-x:right-x] &
                                              other[top-oy:bottom-oy,left-ox:right-ox]))
            union=area+other_area-intersection
            if union and intersection/union>=config['dedupIoU']:
                duplicate=old
                break
        if duplicate is None:
            kept.append((item,mask,bounds,area))
        else:
            duplicate['duplicateIndices'].append(item['sourceIndex'])
    return [entry[0] for entry in kept]


def segment_sam3(image, model, audit=None):
    config=sam3_config(model)
    gray=normalize_uint8(image)
    # Service decodes PNG to RGB; a grayscale PNG yields identical RGB pixels
    # without allocating/encoding three duplicate channels on the client.
    success,png=cv2.imencode('.png',gray)
    if not success: raise ValueError('SAM3 输入图像编码失败')
    raw=png.tobytes()
    digest=hashlib.sha256(raw).hexdigest()
    payload={'image_id':hashlib.sha256(raw+json.dumps(config,sort_keys=True).encode()).hexdigest(),
             'image_png_b64':base64.b64encode(raw).decode(), 'texts':config['texts'],
             'score_threshold':config['scoreThreshold'],'mask_threshold':config['maskThreshold'],
             'max_detections':config['maxDetections']}
    response=request_json(config['url']+'/segment_text',payload)
    if audit is not None:
        audit.update(inputPngSha256=digest,config=config,response=response)
    return decode_detections(response.get('detections'),gray.shape,config)
