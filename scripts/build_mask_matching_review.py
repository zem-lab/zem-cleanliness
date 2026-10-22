"""Build an offline review page; new candidate identities remain unconfirmed."""
import argparse
import base64
import html
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from backend.cleanliness_core.optical_particle_analysis import load_optical_export, normalize_uint8


def png(array):
    ok,data=cv2.imencode('.png',array)
    if not ok:raise ValueError('PNG encoding failed')
    return 'data:image/png;base64,'+base64.b64encode(data).decode()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replay-root',type=Path,required=True)
    p.add_argument('--task-root',type=Path,default=Path.home()/'Documents/ZEM Cleanliness/tasks')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists() or args.output.resolve().is_relative_to(args.task_root.resolve()):
        p.error('Use a fresh output outside the task root')
    rows=[json.loads(x) for x in (args.replay_root/'replay.jsonl').read_text(encoding='utf-8').splitlines()]
    audits=json.loads((args.replay_root/'identity_gate.json').read_text(encoding='utf-8'))
    cards=[]
    for r in rows:
        if not r['knownWrong']:continue
        m=json.loads((args.task_root/r['task']/'task.json').read_text(encoding='utf-8-sig'))
        export=load_optical_export(Path(m['input']['path']))
        assert export.sha256==r['source']
        particle=export.by_sequence[r['sequence']]
        mask=cv2.imdecode(np.frombuffer(base64.b64decode(particle.mask_data_base64.split(',')[-1]),np.uint8),0)
        optical=cv2.imdecode(np.frombuffer(base64.b64decode(particle.image_data_base64.split(',')[-1]),np.uint8),1)
        raw=cv2.imdecode(np.frombuffer(Path(r['image']).read_bytes(),np.uint8),-1)
        sem=cv2.cvtColor(normalize_uint8(raw),cv2.COLOR_GRAY2BGR)
        previous=r['previous']['best']['centerPx']
        current=r['mask']['center']
        for point,color in [(previous,(50,50,240)),(current,(240,170,0))]:
            cv2.drawMarker(sem,tuple(round(x) for x in point),color,cv2.MARKER_CROSS,24,2)
        audit=next(a for a in audits if a['task']==r['task'] and a['sequence']==r['sequence'])
        status='候选通过两种算法复核，身份仍待人工确认' if audit['status']=='MATCHED_HIGH' else '存在歧义或证据不足：不允许采谱'
        figures=''.join(f'<figure><img src="{png(image)}" alt="{title}"><figcaption>{title}</figcaption></figure>' for title,image in [('光镜原图',optical),('客户原始掩码',mask),('历史 SEM 原图上的候选位置',sem)])
        cards.append(f'<section><h2>Rank {r["rank"]} · sequence {r["sequence"]}</h2><p>ID {html.escape(particle.particle_id)}</p><p>{status}</p><div class="grid">{figures}</div><p>红十字：用户指出的历史错误位置；蓝十字：掩码参与后新选位置。标记仅表示算法坐标，不代表人工确认。</p><details><summary>复核数值</summary><pre>{html.escape(json.dumps(audit,ensure_ascii=False,indent=2))}</pre></details></section>')
    text='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src data:; style-src 'unsafe-inline'"><title>历史颗粒匹配离线复核</title><style>body{font:16px/1.6 system-ui,"Microsoft YaHei";background:#f3f5f7;color:#183046;margin:0}main{max-width:1200px;margin:auto;padding:24px}section{background:white;padding:24px;margin:24px 0;border:1px solid #ccd6dd;border-radius:12px}.grid{display:grid;grid-template-columns:1fr 1fr 2fr;gap:16px}figure{margin:0}img{width:100%;height:360px;object-fit:contain;background:#eee}pre{white-space:pre-wrap;overflow-wrap:anywhere}@media(max-width:700px){.grid{grid-template-columns:1fr}img{height:300px}}</style><main><h1>历史颗粒匹配离线复核 · 2026-09-09</h1><p>仅使用已保存数据，没有连接电镜、重新采图或修改历史报告。此页是算法研究复核页，不是新的检测报告。</p><p>原始掩码已参与形状比较；严格模板复核仍是研究方案。三个旧错误位置均不再入选，但不能据此声称三个颗粒已经全部找对。</p>'''+''.join(cards)+'</main></html>'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(text,encoding='utf-8')
    print(args.output)


if __name__=='__main__':main()
