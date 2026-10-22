"""One-time provenance-preserving import; never overwrites regression fixtures.

Selection uses historical metadata only, before current-algorithm replay.
Historical acceptance is a compatibility baseline, not human identity truth.
"""
import hashlib
import json
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
TASKS=Path.home()/'Documents/ZEM Cleanliness/tasks'
OUT=ROOT/'backend/tests/fixtures/matching_expansion_20260918'
MODEL=ROOT/'resources/models/cleanliness/particle_yolo11n_seg_reann_20260811_best.onnx'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def task(tid):
    folder=TASKS/tid
    return folder,json.loads((folder/'task.json').read_text(encoding='utf-8'))


def entries(folder,manifest):
    result=[]
    for record in manifest.get('particles',[]):
        for attempt in record.get('captureAttempts',[]):
            if not attempt.get('rawPath') or not (folder/attempt['rawPath']).is_file():continue
            status=(attempt.get('match') or {}).get('status',attempt.get('status'))
            if not status or status=='FAILED':continue
            result.append((record,attempt,status))
    return result


def diverse_successes(rows,count,forced=()):
    candidates=[r for r in rows if r[2] in {'MATCHED_HIGH','MATCHED_MEDIUM'}]
    # One capture per particle in the diversity pool.
    by_seq={}
    for r in candidates:by_seq.setdefault(r[0]['sequence'],r)
    candidates=list(by_seq.values());selected=[]
    for seq in forced:
        if seq in by_seq:selected.append(by_seq[seq])
    def audit(r):return r[1].get('maskVerification') or {}
    def center(r):return np_center(r[1].get('match') or {})
    keys=[lambda r:float(r[0].get('equivalentDiameterUm') or 0),
          lambda r:-float(r[0].get('lengthUm') or 0),
          lambda r:float(audit(r).get('maskIoU',1)),
          lambda r:float(audit(r).get('margin',2)),
          lambda r:-sum((v-256)**2 for v in center(r)),
          lambda r:-float(r[0].get('lengthUm') or 0)/max(float(r[0].get('widthUm') or 1),1)]
    ordered=[sorted(candidates,key=lambda r:(key(r),r[0]['rank'])) for key in keys]
    while len(selected)<count:
        before=len(selected)
        for ordered_rows in ordered:
            for row in ordered_rows:
                if row[0]['sequence'] not in {r[0]['sequence'] for r in selected}:
                    selected.append(row);break
            if len(selected)>=count:break
        if len(selected)==before:raise RuntimeError('Insufficient distinct historical successes')
    return selected


def np_center(match):
    return (match.get('best') or {}).get('centerPx') or (256,256)


def main():
    if OUT.exists():raise RuntimeError('Do not overwrite an existing corpus; review changes separately')
    existing=set()
    for name,file in [('mask_matching_sample2','baseline.json'),('template_identity_cases','cases.json'),('human_confirmed_20260918','cases.json')]:
        data=json.loads((ROOT/'backend/tests/fixtures'/name/file).read_text(encoding='utf-8'))
        existing.update(c['imageSha256'] for c in data['cases'])
    inputs={}
    for path in TASKS.glob('*/task.json'):
        d=json.loads(path.read_text(encoding='utf-8'));source=Path((d.get('input')or{}).get('path','_missing'))
        if source.is_file():inputs.setdefault((d.get('input')or{}).get('sha256'),source)
    plans=[]
    for slug,tid in [
        ('sample7','82599546-2f1d-40a8-b986-7fca42d99133'),
        ('sample2_stable','a2883068-edf5-4da3-8ed4-fbeb6eeaabad'),
        ('sample2_low_signal','9d7fac90-0900-4312-ae7b-15c0a89255da'),
        ('legacy_without_mask','f22e1c47-08a5-4843-8087-dcbd2a8f6a88'),
    ]:
        folder,manifest=task(tid)
        rows=[r for r in entries(folder,manifest) if sha(folder/r[1]['rawPath']) not in existing]
        selected=[]
        if slug=='sample7':
            selected=diverse_successes(rows,20)
            selected += [r for r in rows if (r[1].get('maskVerification')or{}).get('status')=='NO_MATCH']
        elif slug=='sample2_stable':
            # Include EDS failures whose particle matching actually succeeded.
            eds_sequences=[r[0]['sequence'] for r in rows if r[0]['status']=='FAILED' and r[2] in {'MATCHED_HIGH','MATCHED_MEDIUM'}]
            selected=diverse_successes(rows,10,forced=eds_sequences[:2])
            selected += [r for r in rows if r[0]['sequence'] in {36,63,37,173,232,50,82,178} and r[2].startswith('FAILED')]
        elif slug=='sample2_low_signal':
            selected=[r for r in rows if r[2] in {'MATCHED_HIGH','MATCHED_MEDIUM'}
                      or (r[1].get('maskVerification')or{}).get('status')=='AMBIGUOUS'
                      or r[0]['sequence'] in {16,172}]
        else:
            selected=diverse_successes(rows,4)
            selected += [r for r in rows if r[0]['sequence'] in {250,170,8,47} and r[1].get('attempt')==1 and r[2].startswith('FAILED')]
        plans.append((slug,folder,manifest,selected))
    OUT.mkdir()
    corpus={'schemaVersion':1,'labelMeaning':'Compatibility behavior only; no new human identity annotations.',
        'creationPolicy':'Cases selected from historical metadata before replay. Expected results are historical matching outcomes, never task/EDS success. The importer refuses overwrite.',
        'model':str(MODEL.relative_to(ROOT)).replace('\\','/'),'modelSha256':sha(MODEL),
        'sources':[],'cases':[]}
    for slug,folder,manifest,selected in plans:
        source=inputs[manifest['input']['sha256']]
        assert sha(source)==manifest['input']['sha256']
        original=json.loads(source.read_text(encoding='utf-8-sig'))
        sequences={r[0]['sequence'] for r in selected}
        subset={**original,'particles':[p for p in original['particles'] if p['sequence'] in sequences]}
        by_seq={p['sequence']:p for p in subset['particles']}
        destination=OUT/slug;(destination/'sem').mkdir(parents=True)
        write(destination/'optical.json',subset)
        settings=folder/'.cleanliness-settings.json'
        settings_data=json.loads(settings.read_text(encoding='utf-8')) if settings.is_file() else {}
        seg=(settings_data.get('calibration')or{}).get('candidateSegmentation')or{}
        parameters={k:seg[k] for k in ['inputSize','confidenceThreshold','iouThreshold','maskThreshold','maxDetections'] if k in seg}
        portable={'backend':'yolo','modelPath':corpus['model'],**parameters}
        source_meta={'id':slug,'sourceTaskId':folder.name,'sourceTaskSha256':sha(folder/'task.json'),
            'sourceInputSha256':manifest['input']['sha256'],'recordedInputPath':manifest['input']['path'],
            'resolvedInputPath':str(source),'inputRelocatedByMatchingFingerprint':str(source)!=manifest['input']['path'],
            'optical':f'{slug}/optical.json','opticalSubsetSha256':sha(destination/'optical.json'),
            'settingsSha256':sha(settings) if settings.is_file() else None,'historicalSegmentation':seg,
            'replaySegmentation':portable,'pixelSizeUm':manifest['baselineImaging']['pixelSizeUm']}
        corpus['sources'].append(source_meta)
        for record,attempt,status in selected:
            particle=by_seq[record['sequence']];assert particle['particleId']==record['particleId']
            raw=folder/attempt['rawPath'];fingerprint=sha(raw)
            assert fingerprint not in existing,'Duplicate image would inflate corpus size'
            existing.add(fingerprint)
            filename=f"{record['sequence']:04d}_{attempt['attempt']}.png"
            target=destination/'sem'/filename;shutil.copy2(raw,target)
            audit=attempt.get('maskVerification')
            accepted=status in {'MATCHED_HIGH','MATCHED_MEDIUM'}
            best=(attempt.get('match')or{}).get('best')
            center=(audit or {}).get('selectedCenterPx') or (best or {}).get('centerPx') if accepted else None
            minimum_iou=max(.5,float(audit['maskIoU'])-.03) if accepted and audit and 'maskIoU' in audit else None
            if not particle.get('maskDataBase64'):category='legacy_no_mask_acceptance' if accepted else 'legacy_physical_or_shape_rejection'
            elif accepted:category='matched_despite_eds_failure' if record['status']=='FAILED' else 'accepted_behavior'
            elif (audit or {}).get('status')=='AMBIGUOUS':category='distinct_competition'
            elif (audit or {}).get('status')=='MATCH':category='insufficient_final_contour'
            elif (audit or {}).get('score') is not None:category='below_template_score'
            else:category='template_does_not_fit'
            corpus['cases'].append({'caseId':f"{slug}-rank{record['rank']}-seq{record['sequence']}-capture{attempt['attempt']}",
                'source':slug,'rank':record['rank'],'sequence':record['sequence'],'attempt':attempt['attempt'],'particleId':record['particleId'],
                'image':f'{slug}/sem/{filename}','imageSha256':fingerprint,'labelKind':'historical_behavior_only','category':category,
                'historicalTaskStatus':record['status'],'historicalMatch':attempt.get('match'),
                'historicalMaskVerification':audit,'expectedAccepted':accepted,'expectedDecisionStatus':status,
                'expectedCenterPx':center,'centerTolerancePx':3.0,'minimumMaskIoU':minimum_iou,
                'hasCustomerMask':bool(particle.get('maskDataBase64'))})
    write(OUT/'cases.json',corpus)
    print(json.dumps({'cases':len(corpus['cases']),'sources':len(corpus['sources']),
        'accepted':sum(c['expectedAccepted'] for c in corpus['cases']),
        'rejected':sum(not c['expectedAccepted'] for c in corpus['cases']),
        'bySource':{s['id']:sum(c['source']==s['id'] for c in corpus['cases']) for s in corpus['sources']}}))


if __name__=='__main__':main()
