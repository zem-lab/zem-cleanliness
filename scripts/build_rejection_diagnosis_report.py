"""Render algorithmic peak evidence, without judging particle identities visually."""
import base64
import html
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from backend.cleanliness_core.optical_particle_analysis import load_optical_export,normalize_uint8


def data(image):
    ok,raw=cv2.imencode('.png',image)
    if not ok:raise ValueError('PNG encode failed')
    return 'data:image/png;base64,'+base64.b64encode(raw).decode()


def main():
    root=Path(sys.argv[1]);out=root/'report.html'
    if out.exists():raise FileExistsError(out)
    rows=[r for r in json.loads((root/'diagnostics.json').read_text(encoding='utf-8')) if r['attempt']==2]
    task=Path(rows[0]['image']).parents[1]
    manifest=json.loads((task/'task.json').read_text(encoding='utf-8-sig'))
    export=load_optical_export(Path(manifest['input']['path']))
    assert export.sha256==manifest['input']['sha256']
    cards=[];table=[]
    for r in rows:
        rank=r['rank'];p=export.by_sequence[r['sequence']]
        raw=cv2.imdecode(np.frombuffer(Path(r['image']).read_bytes(),np.uint8),-1)
        image=cv2.cvtColor(normalize_uint8(raw),cv2.COLOR_GRAY2BGR)
        overlay=image.copy();yolo=image.copy()
        for key,color in [('primary',(255,180,0)),('runner',(0,150,255))]:
            mask=cv2.imdecode(np.frombuffer((root/f'rank{rank}_attempt2_{key}.png').read_bytes(),np.uint8),0)
            contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay,contours,-1,color,1)
            cv2.drawMarker(overlay,tuple(round(x) for x in r[key]['centerPixel']),color,cv2.MARKER_CROSS,16,1)
        ys=[c for c in r['candidateOverlaps'] if c['method']=='yolo']
        for c in ys:
            b=c['box'];cv2.rectangle(yolo,tuple(b[:2]),tuple(b[2:]),(0,240,100),1)
            cv2.putText(yolo,f"Y{c['index']}",tuple(b[:2]),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,240,100),1)
        mask=cv2.imdecode(np.frombuffer(base64.b64decode(p.mask_data_base64.split(',')[-1]),np.uint8),0)
        optical=cv2.imdecode(np.frombuffer(base64.b64decode(p.image_data_base64.split(',')[-1]),np.uint8),1)
        yiou=max((c['primary']['iou'] for c in ys),default=0)
        ciou=max((c['primary']['iou'] for c in r['candidateOverlaps'] if c['method']=='classic'),default=0)
        if rank==8:reason='分数不足；细化角度到 8° 后得分由 0.484 提高至 0.716。YOLO 也有完整轮廓不足的问题。'
        elif rank in (73,111,118):reason='两个模板假设位于分离位置；需要进一步的方向、尺度和邻域身份约束，不能简单取消歧义检查。'
        elif rank==42:reason='两模板有较大重叠，但完整形状一致性也较弱。重复假设与分割/形变问题需要分开处理。'
        else:reason='第二峰与主峰的投影高度重叠；仅按短半径排重会把同一目标的邻近高分位置当成竞争目标。'
        table.append(f"<tr><td>{rank}</td><td>{r['sequence']}</td><td>{r['primary']['score']:.3f}</td><td>{r['primary']['margin']:.3f}</td><td>{r['centerDistancePx']:.2f} / {r['radiusPx']:.2f}</td><td>{r['templateOverlap']['iou']:.3f}</td><td>{yiou:.3f} / {ciou:.3f}</td></tr>")
        cards.append(f'''<section><h2>Rank {rank} · sequence #{r['sequence']}</h2><p>{reason}</p>
<div class="grid"><div><figure><figcaption>光镜原图</figcaption><img class="small" src="{data(optical)}"></figure><figure><figcaption>客户原始掩码</figcaption><img class="small" src="{data(mask)}"></figure></div><figure><figcaption>主峰（青）/第二峰（橙）的模板投影</figcaption><img src="{data(overlay)}"></figure><figure><figcaption>独立运行 YOLO 的候选框</figcaption><img src="{data(yolo)}"></figure></div>
<p>两模板中心距离 {r['centerDistancePx']:.2f}px，旧排重半径 {r['radiusPx']:.2f}px；投影 IoU {r['templateOverlap']['iou']:.3f}。YOLO 候选 {r['yoloCount']} 个。</p>
<p class="note">青/橙轮廓是光镜掩码投影，不是 SEM 分割真值。绿色为本地 YOLO 输出；这些图用于人工复核，不能把算法候选直接当作已确认颗粒。</p></section>''')
    document='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src data:; style-src 'unsafe-inline'"><title>失败颗粒原因诊断</title><style>body{font:16px/1.7 system-ui,"Microsoft YaHei";background:#eef2f5;color:#183046;margin:0}main{max-width:1300px;margin:auto;padding:25px}section{background:white;padding:24px;margin:24px 0;border:1px solid #ced9e0;border-radius:10px}.grid{display:grid;grid-template-columns:180px 1fr 1fr;gap:14px}figure{margin:0;background:#f5f7f9}img{width:100%;height:410px;object-fit:contain}.small{height:178px}figcaption{padding:6px;font-size:12px;text-align:center}table{border-collapse:collapse;width:100%;font-size:13px}td,th{border:1px solid #ccd7df;padding:8px}.note{color:#607384;font-size:13px}@media(max-width:750px){.grid{grid-template-columns:1fr}img{height:360px}}</style></head><body><main><h1>12 个失败颗粒的原因诊断</h1><p>2026-09-10 · 来源任务 8e3a30d2-6001-4b59-ae88-a158c37456ca。仅重放保存图片，未连接电镜、未修改历史记录或线上算法。</p><section><h2>结论</h2><p>11 个在模板歧义门槛处停止，1 个分数不足；不是统一的 YOLO 无检测。逐帧重放的分数和差距与原记录一致。7 个高重叠近邻峰支持重复峰排重存在问题；Rank 42 还混有轮廓一致性问题；3 个的第二峰确实在另一位置；Rank 8 对角度采样敏感。</p><p>SAM3 可改善分割候选，但仅更换分割器不会改变当前提前执行的模板分数/歧义门槛。早期项目找到 5 条默认文本提示词及低阈值批量脚本；本次 SAM3 服务健康检查连接被拒绝，未获得任何 SAM3 实测结果。</p><p>下表“分割重合”是候选对模板主峰投影的最大 IoU，不是人工标注准确率。</p><table><thead><tr><th>Rank</th><th>sequence</th><th>主峰分数</th><th>领先差距</th><th>峰距离 / 半径 px</th><th>两模板 IoU</th><th>YOLO / 经典分割重合</th></tr></thead><tbody>'''+''.join(table)+'</tbody></table></section>'+''.join(cards)+'</main></body></html>'
    out.write_text(document,encoding='utf-8');print(out)


if __name__=='__main__':main()
