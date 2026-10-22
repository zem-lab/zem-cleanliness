# Frozen pre-optimization compatibility baseline, not human particle truth.
import base64
import math
import cv2
import numpy as np

def decode_detections(detections, shape, config):
    if not isinstance(detections,list) or len(detections)>=config['maxDetections']:
        raise ValueError('SAM3 返回格式错误或检测数量达到上限，不能确认分割完整')
    items=[]
    for index, det in enumerate(detections):
        score=float(det['score'])
        if not math.isfinite(score) or not config['scoreThreshold'] <= score <= 1:
            raise ValueError('SAM3 返回无效置信度')
        mask=cv2.imdecode(np.frombuffer(base64.b64decode(det['mask_png_b64'],validate=True),np.uint8),cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != tuple(shape):
            raise ValueError('SAM3 掩码尺寸与 SEM 原图不一致')
        mask=(mask>127).astype(np.uint8)
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        points=max(contours,key=cv2.contourArea).reshape(-1,2).tolist() if contours else []
        item={'points':points,'confidence':score,'fragmented':len(contours)!=1,
              'prompt':det.get('text',det.get('label')), 'sourceIndex':index,'duplicateIndices':[]}
        items.append((item,mask))
    kept=[]
    for item,mask in sorted(items,key=lambda x:-x[0]['confidence']):
        duplicate=None
        for old,other in kept:
            union=np.count_nonzero(mask|other)
            if union and np.count_nonzero(mask&other)/union>=config['dedupIoU']:
                # Do not hide fragmented or otherwise invalid shapes via dedup.
                if not item['fragmented'] and not old['fragmented'] and len(item['points'])>=3 and len(old['points'])>=3:
                    duplicate=old
                    break
        if duplicate is None:
            kept.append((item,mask))
        else:
            duplicate['duplicateIndices'].append(item['sourceIndex'])
    return [item for item,_ in kept]
