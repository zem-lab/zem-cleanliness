"""Build a standalone, reviewable report of the offline shape experiment."""
import argparse
import base64
import html
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]


def annotated(row):
    image=Image.open(ROOT/row['image']).convert('RGB')
    draw=ImageDraw.Draw(image)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',15)
    reference=np.asarray(row['referenceCenterPx'])
    for item in row['shapeComparison']['candidates']:
        correct=np.linalg.norm(np.asarray(item['centerPx'])-reference)<12
        label='A' if correct else 'B'
        color='#49ed8f' if correct else '#ffb452'
        points=[tuple(point) for point in item['observedContour']]
        draw.line(points+[points[0]],fill=color,width=2)
        x,y=item['centerPx']
        draw.rectangle((x-35,y-35,x+35,y+35),outline=color,width=1)
        tx=min(max(5,x-35),385);ty=max(3,y-58)
        draw.text((tx,ty),f"{label}  {item['shapeIoU']:.4f}",font=font,fill=color,stroke_width=1,stroke_fill='#111111')
    stream=io.BytesIO();image.save(stream,format='PNG')
    return 'data:image/png;base64,'+base64.b64encode(stream.getvalue()).decode()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--robustness',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=json.loads(args.results.read_text(encoding='utf-8'))
    perturb=json.loads(args.robustness.read_text(encoding='utf-8'))
    assert result['summary']['frames']==77, 'Report requires the completed real-image replay'
    assert len(perturb)==16
    latest=[r for r in result['cases'] if r['corpus']=='human_confirmed_20260918']
    originals=[r for r in result['cases'] if r['corpus']!='human_confirmed_20260918']
    mismatches=[r for r in originals if r['historicalAccepted'] and not r['sameObjectAsReference']]
    accepted=[r for r in originals if r['historicalAccepted']]
    same=sum(r['sameObjectAsReference'] for r in accepted)
    rows=[]; figures=[]
    for row in latest:
        candidates=row['shapeComparison']['candidates']
        best,second=candidates[:2]
        rows.append(f"<tr><td>{row['rank']}</td><td>{row['attempt']}</td><td>A</td><td>{best['shapeIoU']:.4f}</td><td>{second['shapeIoU']:.4f}</td><td>{row['shapeComparison']['scoreMargin']:.4f}</td></tr>")
        if row['attempt']==2:
            figures.append(f'<figure><h3>报告颗粒 {row["rank"]} · 第 2 次采图</h3><img src="{annotated(row)}" alt="颗粒 {row["rank"]} 的 A、B 外形比较"><figcaption>绿线 A：用户确认的目标；橙线 B：另一颗粒。分数为直接外形重合度。</figcaption></figure>')
    negative=[r for r in originals if r['corpus']=='template_identity_cases' and r['rank']==118]
    negative_text='；'.join(f"第 {r['attempt']} 帧外形最高分 {r['shapeComparison']['candidates'][0]['shapeIoU']:.4f}，领先 {r['shapeComparison']['scoreMargin']:.4f}" for r in negative)
    mismatch_rows=''.join(f"<tr><td>{html.escape(r['corpus'])}</td><td>{r['rank']}</td><td>{r['attempt'] or '-'}</td><td>{'无完整可用轮廓' if not r['shapeComparison']['candidates'] else '排名首位与旧参考位置不一致'}</td></tr>" for r in mismatches)
    stability_min=min(r['scoreMargin'] for r in perturb)
    document='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>颗粒 48、125：外形优化离线试验</title>
<style>body{margin:0;background:#f3f6fa;color:#172b42;font:16px/1.7 "Microsoft YaHei",sans-serif}main{max-width:1060px;margin:auto;padding:32px 20px 70px}p{overflow-wrap:anywhere}h1{font-size:28px}h2{font-size:21px}h3{font-size:17px}.panel{background:white;padding:22px;margin:18px 0;border:1px solid #dbe4ef;border-radius:12px}.result{border-left:5px solid #21825e}.limit{border-left:5px solid #b46b13}.pair{display:grid;grid-template-columns:1fr 1fr;gap:20px}figure{margin:0;min-width:0}img{width:100%;height:auto}figcaption,.muted{font-size:13px;color:#53677d}table{width:100%;border-collapse:collapse;font-size:14px;table-layout:fixed}th,td{padding:9px 5px;text-align:left;border-bottom:1px solid #e4eaf1;overflow-wrap:anywhere}summary{cursor:pointer}code{overflow-wrap:anywhere;font-size:13px}a{color:#145ab1}.bad{color:#954d0c}@media(max-width:650px){.pair{grid-template-columns:1fr}main{padding:18px 12px}.panel{padding:15px}h1{font-size:23px}table{font-size:12px}}</style></head><body><main>
<h1>颗粒 48、125：外形优化离线试验</h1>
<p>范围：先试“更认真地比较实际外形”，同时检查旧案例。2026-09-18。</p>
<section class="panel result"><h2>新案例有改善，正式采谱规则尚未切换</h2>
<p>两颗粒共 <b>4 张原始采图，外形比较都把 A 排第一</b>；另外 16 张轻微成像变化图也都保持 A 第一。</p>
<p>外形比较是独立的离线试验，没有接入正式任务、没有改旧门槛、没有控制设备或重新采谱。当前安装版仍会在这两例上拒绝。</p></section>
<section class="panel"><h2>这次具体改了什么</h2><p>将客户原始掩码与实际分割出来的 SEM 外形直接比较，减少内部亮暗对分数的影响。每个候选都采用同样的旋转、镜像、物理尺寸范围和细化搜索；没有硬编码颗粒编号或 A 的位置。</p><p>另外修复了半像素对齐时二值轮廓被缩小的计算问题，并补充对应测试。</p></section>
<section class="panel"><h2>两颗粒的结果</h2><div class="pair">__FIGURES__</div>
<table><thead><tr><th>报告颗粒</th><th>采图</th><th>排名首位</th><th>A 外形分数</th><th>B 外形分数</th><th>A − B</th></tr></thead><tbody>__NEW_ROWS__</tbody></table>
<p class="muted">这些是新的外形重合分数，不是原来的明暗相关分数，更不是身份正确率。第 2 帧有用户直接身份确认，第 1 帧是同任务重复采集支持证据。</p></section>
<section class="panel"><h2>小幅成像变化检查</h2><p>对每张原图分别进行：对比度变为 0.8 并加 10、固定种子标准差 2 的噪声、σ=0.5 的轻微模糊，以及水平 +1 / 垂直 −1 像素移动。</p><p><b>16 / 16 仍将 A 排第一。</b> 最小领先差为 __MIN_MARGIN__；125 的一次 1 像素移动会使领先差缩小，因此仍不能把单次分差当成可靠采谱许可。</p><p class="muted">这些是派生的稳定性检查，不是新增的人工标注样本。</p></section>
<section class="panel limit"><h2>为什么没有直接替换正式算法</h2><p>旧的 73 帧中有 62 帧曾被正式算法接受。若只用全图 YOLO 外形排名，<b>__SAME__ / 62</b> 排名首位仍在旧参考对象附近，另有 <b>__MISMATCHES__ 帧</b>位置不一致或没有完整可用轮廓。这不是准确率统计，但足以说明直接替换会改变旧行为。</p>
<p>旧竞争负例 Rank 118：__NEGATIVE__。它原本仍应拒绝，可见“外形第一名领先”本身不能证明已经排除竞争。</p>
<p><b>结论：保留正式算法。外形信息可以继续研究为额外复核证据，不能单独决定选谁或能否采谱。</b></p>
<details><summary>展开查看与旧接受基线不一致的帧</summary><table><thead><tr><th>测试集</th><th>报告编号</th><th>采图</th><th>观察</th></tr></thead><tbody>__MISMATCH_ROWS__</tbody></table></details>
<p class="muted">旧接受状态不是独立人工身份真值。以上检查区分“行为变了”和“身份已确认错误”，不把二者混为一谈。</p></section>
<section class="panel"><h2>防退化检查</h2><ul><li>现有后端完整回归：317 通过、1 跳过、2 个已知误拒绝 XFAIL。旧的 73 帧回归均通过。</li><li>新增外形试验测试：22 通过，覆盖真实 A/B、两个相同目标、无目标、轮廓不足、边缘截断、错误坐标、错误像素尺度、取消和禁止试验结果直接授权 EDS。</li><li>匹配专项命令已补齐两套旧真实图、新人工确认案例、外形试验及流程保护，共收集 186 项。</li><li>没有覆盖旧基线，没有将现有两个生产误拒绝写成“已经修好”。</li></ul></section>
<section class="panel"><h2>证据与复现</h2><p>原图、光镜掩码、模型指纹、用户确认以及逐帧结果均保留。</p><p><code>python scripts/evaluate_shape_comparison.py --output artifacts/新目录</code></p><p><code>python scripts/check_shape_comparison_robustness.py --output artifacts/另一个新目录</code></p><p><code>npm run test:matching</code></p><p class="muted">结果文件：__RESULT_PATH__。本报告不是安装版验证或发布记录。</p></section>
</main></body></html>'''
    replacements={'__FIGURES__':''.join(figures),'__NEW_ROWS__':''.join(rows),
        '__MIN_MARGIN__':f'{stability_min:.4f}','__SAME__':str(same),'__MISMATCHES__':str(len(mismatches)),
        '__NEGATIVE__':negative_text,'__MISMATCH_ROWS__':mismatch_rows,'__RESULT_PATH__':html.escape(str(args.results.resolve()))}
    for key,value in replacements.items():document=document.replace(key,value)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(document,encoding='utf-8')
    print(args.output.resolve())


if __name__=='__main__':main()
