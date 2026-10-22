"""Offline ECD-filtered comparison using saved SEM pixels and task settings."""
import argparse
import base64
import hashlib
import html
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.replay_historical_particle_matching import no_network
import cv2
import numpy as np
from loguru import logger
from backend.cleanliness_core.optical_particle_analysis import load_optical_export,normalize_uint8
from backend.cleanliness_core.cleanliness_detection import build_sem_anchor_candidates,CandidateSegmentationConfig,CandidateGate
from backend.cleanliness_core.cleanliness_task import _select_mask_guided_candidate
from backend.cleanliness_core.html_to_pdf_edge import html_to_pdf_via_edge


def image_url(image):
    ok,data=cv2.imencode('.png',image)
    if not ok:raise ValueError('PNG encoding failed')
    return 'data:image/png;base64,'+base64.b64encode(data).decode()


def decode(encoded,mode):
    return cv2.imdecode(np.frombuffer(base64.b64decode(encoded.split(',')[-1]),np.uint8),mode)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--min-ecd',type=float,default=80)
    parser.add_argument('--ranks',type=int,nargs='+',help='Optionally restrict to original report ranks')
    args=parser.parse_args(); logger.remove()
    root=args.task.resolve(); out=args.output.resolve()
    if out.exists() or out.is_relative_to(root):parser.error('Use a fresh output outside the task directory')
    out.mkdir(parents=True)
    task_bytes=(root/'task.json').read_bytes()
    manifest=json.loads(task_bytes.decode('utf-8-sig'))
    export=load_optical_export(Path(manifest['input']['path']))
    assert export.sha256==manifest['input']['sha256']
    selected=[r for r in manifest['particles'] if r['equivalentDiameterUm']>args.min_ecd and (not args.ranks or r['rank'] in args.ranks)]
    allowed={r['sequence'] for r in manifest['particles'] if not args.ranks or r['rank'] in args.ranks}
    expected={p.sequence for p in export.particles if p.equivalent_diameter_um>args.min_ecd and p.sequence in allowed}
    assert {r['sequence'] for r in selected}==expected
    assert len(selected)==len(expected)
    settings_path=root/(manifest.get('settingsSnapshot') or '.cleanliness-settings.json')
    settings=json.loads(settings_path.read_text(encoding='utf-8-sig'))['calibration']
    segmentation=CandidateSegmentationConfig.from_dict(settings['candidateSegmentation'],settings_path=settings_path)
    if segmentation.backend=='sam3':raise ValueError('Remote segmentation is prohibited')
    gate=CandidateGate.from_dict(settings.get('candidateGate'))
    pixel=manifest['baselineImaging']['pixelSizeUm']
    cards=[];results=[]
    for i,r in enumerate(selected,1):
        p=export.by_sequence[r['sequence']]
        saved_path=r['images'].get('semRawImage')
        shot=next((a for a in reversed(r.get('captureAttempts') or []) if a.get('rawPath') and (not saved_path or a['rawPath']==saved_path)),{})
        raw_path=root/(saved_path or shot['rawPath'])
        raw_bytes=raw_path.read_bytes()
        raw=cv2.imdecode(np.frombuffer(raw_bytes,np.uint8),-1)
        background=cv2.cvtColor(normalize_uint8(raw),cv2.COLOR_GRAY2BGR)
        old=background.copy();new=background.copy()
        previous=(shot.get('match') or r['match'])['best']; box=previous['boxPx'];center=previous['centerPx']
        cv2.rectangle(old,tuple(box[:2]),tuple(box[2:]),(40,40,240),2)
        cv2.drawMarker(old,tuple(round(x) for x in center),(40,40,240),cv2.MARKER_CROSS,22,2)
        row=dict(rank=r['rank'],sequence=r['sequence'],particleId=r['particleId'],ecdUm=r['equivalentDiameterUm'],
                 sourceImage=str(raw_path),imageSha256=hashlib.sha256(raw_bytes).hexdigest(),oldCenterPx=center)
        try:
            candidates=build_sem_anchor_candidates(p,raw,pixel_size_um=pixel,stage_center_um=(0,0),segmentation=segmentation,gate=gate)
            chosen,decision,audit=_select_mask_guided_candidate(raw,p,pixel,candidates,stage_center_um=(0,0),segmentation=segmentation,gate=gate)
            cv2.polylines(new,[np.asarray(chosen.candidate.contour,np.int32)],True,(240,160,0),2)
            cv2.drawMarker(new,tuple(round(x) for x in chosen.pixel_xy),(240,160,0),cv2.MARKER_CROSS,22,2)
            if audit and 'centerPixel' in audit:
                cv2.drawMarker(new,tuple(round(x) for x in audit['centerPixel']),(0,170,255),cv2.MARKER_TILTED_CROSS,20,2)
            row.update(status=decision.status,accepted=decision.matched,newCenterPx=list(chosen.pixel_xy),
                       centerChangeUm=float(np.linalg.norm(np.subtract(center,chosen.pixel_xy)))*pixel,
                       reason=decision.reason,maskVerification=audit)
        except Exception as exc:
            row.update(status='REPLAY_FAILED',accepted=False,reason=str(exc),maskVerification=None)
        results.append(row)
        optical=decode(p.image_data_base64,1);mask=decode(p.mask_data_base64,0)
        label='通过算法复核 · 身份仍待确认' if row['accepted'] else '未通过复核 · 不应据此采谱'
        change=f"{row['centerChangeUm']:.1f} μm" if 'centerChangeUm' in row else '—'
        audit=row['maskVerification'] or {}
        score=f"{audit['score']:.3f}" if 'score' in audit else '—'
        margin=f"{audit['margin']:.3f}" if 'margin' in audit else '—'
        cards.append(f'''<section class="particle"><header><b>原报告 Rank {r['rank']} · sequence #{r['sequence']}</b><span>{i} / {len(selected)}</span></header>
<p class="id">ID {html.escape(r['particleId'])}　|　ECD {r['equivalentDiameterUm']:.1f} μm　|　光镜尺寸 {r['lengthUm']:.1f} × {r['widthUm']:.1f} μm</p>
<p class="status">{label}</p><div class="grid"><div class="optical"><figure><figcaption>光镜原图</figcaption><img src="{image_url(optical)}"></figure><figure><figcaption>客户原始掩码</figcaption><img src="{image_url(mask)}"></figure></div>
<figure><figcaption>旧匹配：红框及红十字</figcaption><img class="sem" src="{image_url(old)}"></figure><figure><figcaption>新候选：蓝色轮廓及十字</figcaption><img class="sem" src="{image_url(new)}"></figure></div>
<p class="metrics">新旧位置差 {change}　|　模板相关分数 {score}　|　第二候选差距 {margin}</p>
<p class="reason">{html.escape(row['reason'])}</p><footer>蓝色：SEM 候选轮廓与中心；橙色叉号：掩码模板定位。拒绝候选也展示。未重新采图或采谱。</footer></section>''')
        (out/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
        print(f'{i}/{len(selected)} rank={r["rank"]} {row["status"]}',flush=True)
    counts=dict(Counter(r['status'] for r in results));accepted=sum(r['accepted'] for r in results)
    summary=dict(taskId=manifest['taskId'],sourceSha256=export.sha256,filter=f'ECD > {args.min_ecd} um',
                 selected=len(selected),accepted=accepted,rejected=len(selected)-accepted,counts=counts,
                 notice='Offline research result. Acceptance is not confirmed matching accuracy; no new EDS.')
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    style='''@page{size:A4 landscape;margin:8mm}*{box-sizing:border-box}body{font:14px/1.45 "Microsoft YaHei",sans-serif;color:#183046;margin:0;background:white}.cover,.particle{break-after:page;padding:12px}.particle:last-child{break-after:auto}header{display:flex;justify-content:space-between;background:#153e52;color:white;padding:10px;font-size:18px}.id{font-size:12px;margin:9px 0}.status{font-weight:bold;padding:5px;background:#edf3f7;margin:6px 0}.grid{display:grid;grid-template-columns:170px 1fr 1fr;gap:12px}figure{margin:0;border:1px solid #d5dfe7}figcaption{text-align:center;background:#edf3f7;padding:5px;font-weight:bold}img{display:block;object-fit:contain;width:100%}.optical img{height:152px}.optical figure+figure{margin-top:10px}.sem{height:350px}.metrics{font-weight:bold;margin:10px 0 4px}.reason,footer{font-size:11px;margin:4px 0}footer{color:#617384}.cover{height:650px;padding:36px}h1{font-size:30px}.cover p{font-size:17px;line-height:1.8}.cover code{font-size:13px}@media screen{body{max-width:1120px;margin:auto;background:#eef2f5}.particle,.cover{background:white;margin:20px 0}}'''
    report_title='指定颗粒修复验证' if args.ranks else f'ECD > {args.min_ecd:g} μm 颗粒离线重匹配'
    selection_note=('指定原报告 Rank：'+', '.join(map(str,args.ranks))) if args.ranks else f'等效直径严格大于 {args.min_ecd:g} μm'
    cover=f'''<section class="cover"><h1>{html.escape(report_title)}</h1><p>来源任务包含 {len(manifest['particles'])} 条颗粒记录<br>任务：<code>{html.escape(manifest['taskId'])}</code></p><p>{selection_note}；共 <b>{len(selected)}</b> 个颗粒，保持原报告 Rank 顺序。</p><p>本轮：<b>{accepted}</b> 个通过算法复核，<b>{len(selected)-accepted}</b> 个未通过。<br>这是算法结果分布，不是匹配准确率。新位置需要人工核对。</p><p>使用客户原始掩码、历史设置中的本地 {segmentation.backend.upper()} 分割和物理尺度模板复核。相关分数、不同目标差距和轮廓 IoU 门槛未放宽。</p><p>每页含光镜原图、原始掩码和同一张 SEM 图上的新旧候选。原记录可能是拒绝项，不代表原位置曾验收成功。未通过的新候选也展示。</p><p>全程离线；未连接电镜，未重新采集 EDS，未修改旧报告。</p></section>'''
    document='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta http-equiv="Content-Security-Policy" content="default-src \'none\'; img-src data:; style-src \'unsafe-inline\'"><title>颗粒重匹配验证</title><style>'+style+'</style></head><body>'+cover+''.join(cards)+'</body></html>'
    report=out/'comparison.html';report.write_text(document,encoding='utf-8')
    html_to_pdf_via_edge(report,out/'comparison.pdf')
    assert (root/'task.json').read_bytes()==task_bytes
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
