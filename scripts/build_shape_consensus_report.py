"""Standalone evidence report for the guarded shape-consensus improvement."""
import argparse
import base64
import hashlib
import html
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--evidence',type=Path,required=True)
parser.add_argument('--robustness-result',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--tests-passed',type=int,required=True)
args=parser.parse_args()
data=json.loads(args.evidence.read_text(encoding='utf-8'))
assert data['summary']=={'cases':6,'accepted':4,'preservedRejections':2}
for name,digest in data['codeSha256'].items():
    assert hashlib.sha256((ROOT/'backend/cleanliness_core'/name).read_bytes()).hexdigest()==digest,name
augmented=json.loads(args.robustness_result.read_text(encoding='utf-8'))
assert augmented['summary']=={'cases':22,'accepted':20,'preservedRejections':2}
fixture=ROOT/'backend/tests/fixtures/human_confirmed_20260918'
spec=json.loads((fixture/'cases.json').read_text(encoding='utf-8'))
optical=json.loads((fixture/'optical.json').read_text(encoding='utf-8'))
byseq={p['sequence']:p for p in optical['particles']}
rechecked=[]
for row in augmented['cases']:
    if not row['corpus'].startswith('derived_'):continue
    selected=np.asarray(row['selectedCenterPx'])
    reviewed=np.asarray(row['maskVerification']['shapeConsensus']['targets'][0]['yoloCenterPx'])
    tolerance=max(8.,byseq[row['sequence']]['particleInfo']['width']/spec['pixelSizeUm']*.25)
    distance=float(np.linalg.norm(selected-reviewed))
    assert row['accepted'] and distance<=tolerance
    rechecked.append({'condition':row['corpus'],'rank':row['rank'],'attempt':row['attempt'],
                      'distancePx':distance,'maximumDifferencePx':tolerance,'satisfiesFinalCenterGuard':True})
assert len(rechecked)==16
guard_record={'sourceEvidenceSha256':hashlib.sha256(args.robustness_result.read_bytes()).hexdigest(),
              'finalTaskCodeSha256':data['codeSha256']['cleanliness_task.py'],
              'meaning':'Recheck the exact added final-center predicate against saved full-pipeline perturbation results; this is not a new capture or a repeated full pipeline run.',
              'cases':rechecked}
args.output.parent.mkdir(parents=True,exist_ok=True)
(args.output.parent/'final_center_guard_recheck.json').write_text(json.dumps(guard_record,ensure_ascii=False,indent=2),encoding='utf-8')

def uri(im):
    f=io.BytesIO();im.save(f,format='PNG');return 'data:image/png;base64,'+base64.b64encode(f.getvalue()).decode()

def figure(row):
    im=Image.open(row['sourceImage']).convert('RGB');draw=ImageDraw.Draw(im)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
    x,y=row['selectedCenterPx'];box=row['decision']['best']['boxPx']
    draw.rectangle(box,outline='#49ef89',width=2)
    draw.line([(x-5,y),(x+5,y)],fill='#49ef89',width=2);draw.line([(x,y-5),(x,y+5)],fill='#49ef89',width=2)
    draw.text((max(3,x-40),max(3,box[1]-28)),'已选中 A',font=font,fill='#49ef89',stroke_width=1,stroke_fill='#111111')
    bx,by=row['maskVerification']['runnerUp']['centerPixel']
    draw.rectangle((bx-27,by-27,bx+27,by+27),outline='#ffbf69',width=2)
    draw.text((min(max(3,bx-35),410),max(3,by-54)),'竞争 B',font=font,fill='#ffbf69',stroke_width=1,stroke_fill='#111111')
    return '<figure><h3>颗粒 '+str(row['rank'])+' · 第 2 次采图</h3><img src="'+uri(im)+'" alt="最终匹配选中 A，B 未被选择"><figcaption>绿框与十字：最终选中位置；橙框：竞争位置 B。</figcaption></figure>'

new=[c for c in data['cases'] if c['corpus']=='human_confirmed_20260918']
negative=[c for c in data['cases'] if c['corpus']=='template_identity_cases']
rows=''.join(f"<tr><td>{c['rank']}</td><td>{c['attempt']}</td><td>匹配失败</td><td class='ok'>匹配成功，选 A</td><td>{c['maskVerification']['maskIoU']:.4f}</td></tr>" for c in new)
neg_rows=''.join(f"<tr><td>旧任务 118 / {c['attempt']}</td><td>{c['maskVerification']['distinctHypothesisCount']} 个竞争位置</td><td>仍拒绝</td></tr>" for c in negative)
details=[]
for c in new:
    review=c['maskVerification']['shapeConsensus'];targets=review['targets']
    details.append(f"<tr><td>{c['rank']} / {c['attempt']}</td><td>{targets[0]['yoloShapeIoU']:.4f} / {targets[1]['yoloShapeIoU']:.4f}</td><td>{targets[0]['grayscaleShapeIoU']:.4f} / {targets[1]['grayscaleShapeIoU']:.4f}</td><td>{c['maskVerification']['consensusCenterDifferencePx']:.3f} px</td></tr>")
page='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>颗粒 48、125：第二轮优化结果</title><style>
body{margin:0;background:#f3f6fa;color:#193149;font:16px/1.7 "Microsoft YaHei",sans-serif}main{max-width:1080px;margin:auto;padding:28px 20px 70px}h1{font-size:28px}h2{font-size:21px}h3{font-size:17px}.card{background:white;border:1px solid #dce4eb;border-radius:12px;margin:18px 0;padding:22px}.success{border-left:5px solid #19845b}.notice{border-left:5px solid #c28429}.pair{display:grid;grid-template-columns:1fr 1fr;gap:20px}figure{margin:0;min-width:0}img{width:100%;height:auto}figcaption,.note{font-size:13px;color:#51677b}p,td,code{overflow-wrap:anywhere}table{border-collapse:collapse;width:100%;table-layout:fixed;font-size:14px}td,th{padding:9px 5px;text-align:left;border-bottom:1px solid #e4eaf0}.ok{color:#14724b}summary{cursor:pointer}code{font-size:13px}@media(max-width:650px){.pair{grid-template-columns:1fr}main{padding:16px 12px}.card{padding:16px}h1{font-size:23px}table{font-size:12px}}</style></head><body><main>
<h1>颗粒 48、125：第二轮优化结果</h1>
<section class="card success"><h2>源码回放已匹配成功，选中你确认的 A</h2><p>4 张原图全部通过最终匹配；旧的 73 帧回归保持原结果，真正分不清的旧案例继续拒绝。</p><p>这次将外形作为额外复核，不再用它替换全部匹配。当前安装版尚未重新打包，历史报告和实际 EDS 数据没有改写。</p></section>
<section class="card"><h2>这次怎么改的</h2><ol><li>原来已经能匹配成功的颗粒，继续走原流程。</li><li>仅在完整搜索得到两个分离的竞争位置时，对 A、B 都提取可靠轮廓。</li><li>YOLO 和直接从 SEM 明暗取出的轮廓都支持原第一候选明显领先，才继续往下检查。</li><li>最后的轮廓必须仍在同一颗粒上，并通过原有轮廓和采谱安全点要求。</li></ol><p>竞争颗粒漏检、轮廓残缺、两种结果不一致、位置换了，都会保留拒绝。</p></section>
<section class="card"><h2>两颗粒的最终选择</h2><div class="pair">__FIGURES__</div><table><thead><tr><th>颗粒</th><th>采图</th><th>修改前</th><th>修改后</th><th>最终轮廓 IoU</th></tr></thead><tbody>__ROWS__</tbody></table><p class="note">第 2 帧由用户直接确认 A；第 1 帧是同任务重复采集。人工确认的是对象身份，不是逐像素轮廓。</p></section>
<section class="card"><h2>防止把其他颗粒改坏</h2><ul><li>旧真实图 73 帧：62 个接受、11 个拒绝保持原有基线；旧接受位置与轮廓保护断言未放宽。</li><li>16 张轻微亮度、噪声、模糊和移动变化图：完整匹配都选 A；后补的最终位置约束对这些保存结果复核也全部满足。</li><li>测试覆盖：两个同样相似的目标、无目标、漏掉竞争者、轮廓不足、坐标错误、分割异常、取消，以及拒绝后 EDS 调用为零。</li><li>即使外形复核成功，最终轮廓不合格或位置不一致，也不能进入 EDS。</li></ul><table><thead><tr><th>保留的负例</th><th>搜索情况</th><th>当前结果</th></tr></thead><tbody>__NEG_ROWS__</tbody></table><p>验证合计 __TESTS__ 项通过，1 项因缺少旧历史图片跳过。原来的两项已知误拒绝已转为正常通过测试，不再用 XFAIL 掩盖。</p><p class="note">旧基线是兼容性证据，不是全部人工身份真值；成像变化图也是派生测试，不是新增人工标注。</p></section>
<section class="card"><details><summary>展开查看具体复核证据</summary><table><thead><tr><th>颗粒 / 采图</th><th>YOLO 外形 A / B</th><th>灰度外形 A / B</th><th>最终中心与复核中心差</th></tr></thead><tbody>__DETAILS__</tbody></table><p>两种轮廓来自同一张 SEM 图，不能当成两次独立采图或身份概率。新门槛：YOLO 外形重合度至少 0.85；两种分割对每个对象的重合度至少 0.75；两路各自领先至少 0.07。原模板分数及其差距保留，最终轮廓 IoU 仍至少 0.5。</p><p>原始 0.65 模板分数门槛没有降低；原模板唯一性规则仍保留。新增复核的依据单独记入 shapeConsensus。</p></details></section>
<section class="card notice"><h2>交付范围</h2><p>源码和回归测试已修改；尚未打包、安装或在电镜上重新执行。全程使用保存图片和模拟设备测试。</p><p>下一次发布必须另行完成冻结 EXE 的运行时自检与安装界面验证，不能用本次源码测试替代。</p></section>
</main></body></html>'''
for key,value in {'__FIGURES__':''.join(figure(c) for c in new if c['attempt']==2),'__ROWS__':rows,
                  '__NEG_ROWS__':neg_rows,'__TESTS__':str(args.tests_passed),'__DETAILS__':''.join(details)}.items():page=page.replace(key,value)
args.output.write_text(page,encoding='utf-8')
print(args.output.resolve())
