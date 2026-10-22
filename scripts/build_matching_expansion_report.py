"""Reviewable corpus inventory and unhidden real-image safety counterexamples."""
import base64
import hashlib
import html
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/matching_expansion_20260918'
FIXTURES=ROOT/'backend/tests/fixtures'
expansion=json.loads((FIXTURES/'matching_expansion_20260918/cases.json').read_text(encoding='utf-8'))
legacy=json.loads((FIXTURES/'matching_expansion_20260918/legacy_version_review.json').read_text(encoding='utf-8'))
controls=json.loads((FIXTURES/'real_matching_controls_20260918/cases.json').read_text(encoding='utf-8'))
human=json.loads((FIXTURES/'human_confirmed_20260918/cases.json').read_text(encoding='utf-8'))
probe=json.loads((OUT/'real_negative_probe.json').read_text(encoding='utf-8'))
font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)


def encode(im):
    stream=io.BytesIO();im.save(stream,format='PNG')
    return 'data:image/png;base64,'+base64.b64encode(stream.getvalue()).decode()


figures=[]
for rank in [48,125]:
    source=next(c for c in human['cases'] if c['reportRank']==rank and c['reviewStatus']=='human_confirmed')
    control=next(c for c in controls['cases'] if c['rank']==rank and c['condition']=='target_erased_competitor_remains')
    result=next(c for c in probe if c['rank']==rank and c['condition']=='target_erased_competitor_remains')
    original=Image.open(FIXTURES/'human_confirmed_20260918'/source['image']).convert('RGB')
    changed=Image.open(FIXTURES/'real_matching_controls_20260918'/control['image']).convert('RGB')
    assert np.array_equal(np.asarray(changed),np.asarray(Image.open(OUT/result['image']).convert('RGB')))
    assert result['accepted'] and result['audit']['status']=='MATCH'
    d=ImageDraw.Draw(original);x,y=source['correctTarget']['referenceCenterPx'];bx,by=source['wrongTarget']['referenceCenterPx']
    d.rectangle((x-35,y-35,x+35,y+35),outline='#4deb8a',width=2)
    d.text((x-35,y-61),'正确目标 A',font=font,fill='#4deb8a',stroke_width=1,stroke_fill='#111111')
    d.rectangle((bx-25,by-25,bx+25,by+25),outline='#ffbf66',width=2)
    d.text((min(bx-28,400),by-51),'错误颗粒 B',font=font,fill='#ffbf66',stroke_width=1,stroke_fill='#111111')
    d=ImageDraw.Draw(changed);d.rectangle(control['derivation']['replaceXyxy'],outline='#69b9ff',width=2)
    d.text((x-42,y-60),'A 已移除',font=font,fill='#69b9ff',stroke_width=1,stroke_fill='#111111')
    d.rectangle(result['decision']['best']['boxPx'],outline='#ff6767',width=3)
    d.text((min(bx-35,380),by-59),'程序错误接受 B',font=font,fill='#ff6767',stroke_width=1,stroke_fill='#111111')
    figures.append(f'<h3>来源：报告颗粒 {rank}</h3><div class="pair"><figure><img src="{encode(original)}" alt="原始采图中正确目标 A 与错误颗粒 B"><figcaption>原始采图，A 的身份由用户确认。</figcaption></figure><figure><img src="{encode(changed)}" alt="派生反例中 A 被移除但程序错误接受 B"><figcaption>派生反例，仅将 A 区域换成同图背景，B 像素不变。</figcaption></figure></div>')
source_rows=''.join(f"<tr><td>{html.escape(s['id'])}</td><td>{sum(c['source']==s['id'] for c in expansion['cases'])}</td><td>{html.escape(s['sourceTaskId'])}</td></tr>" for s in expansion['sources'])
legacy_rows=''.join(f"<tr><td>{c['rank']} / {c['sequence']}</td><td>{'接受' if c['historicalExpected']['accepted'] else '拒绝'}</td><td>{'接受' if c['committedExpected']['accepted'] else '拒绝'}</td><td>{c['historicalToCommittedDistancePx']:.2f} px</td></tr>" for c in legacy['cases'] if c['differenceAlreadyPresentInCommittedCode'])
page='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>回归集扩充与新发现</title><style>
body{margin:0;background:#f4f6fa;color:#193047;font:16px/1.7 "Microsoft YaHei",sans-serif}main{max-width:1100px;margin:auto;padding:28px 20px 60px}h1{font-size:28px}h2{font-size:21px}h3{font-size:17px}.card{background:white;border:1px solid #dfe5ec;border-radius:12px;padding:22px;margin:20px 0}.issue{border-left:5px solid #c8463a}.pair{display:grid;grid-template-columns:1fr 1fr;gap:18px}figure{margin:0;min-width:0}img{width:100%;height:auto}figcaption,.note{font-size:13px;color:#526b7e}table{width:100%;table-layout:fixed;border-collapse:collapse;font-size:14px}th,td{padding:10px 5px;border-bottom:1px solid #e1e8ee;text-align:left;overflow-wrap:anywhere}p,code{overflow-wrap:anywhere}summary{cursor:pointer}code{font-size:13px}@media(max-width:650px){.pair{grid-template-columns:1fr}main{padding:16px 12px}.card{padding:16px}h1{font-size:23px}table{font-size:12px}}</style></head><body><main>
<h1>回归集已扩充，并发现两个必须修复的反例</h1>
<section class="card"><h2>新增了什么</h2><ul><li>真实原始采图：77 → <b>155 张</b>，新增 78 张，按图片指纹去重。</li><li>另加 <b>16 张派生控制图</b>：8 张应保持 A 的扰动图、8 张应拒绝的反例；不混算成原始采图或人工标签。</li><li>新增错误坐标检查，以及真实图配合假设备验证拒绝后 EDS 调用为零。</li><li>原来因旧图片缺失而跳过的 YOLO 检查，改用仓库内固定模型和真实图片，现已实际执行通过。</li></ul>
<table><thead><tr><th>新增来源</th><th>原始帧数</th><th>来源任务</th></tr></thead><tbody>__SOURCES__</tbody></table><p class="note">4 个任务不等于 4 个独立样本；重复采图不等于新增人工身份真值。</p></section>
<section class="card issue"><h2>发现的明确漏洞：A 不在时，会把 B 当成目标</h2><p>两个反例均基于用户确认 A 正确、B 非目标的真实图。仅移除 A 后，原有“唯一高分位置”路径仍接受 B。预期应拒绝，当前实际接受，因此测试明确失败。</p>__FIGURES__
<p>两个场景在假设备任务流程中还各触发了 1 次 EDS 分析调用。匹配拒绝与 EDS 拦截合计 <b>4 条断言失败</b>，未操作真实设备。</p><p><b>没有 XFAIL、没有跳过、没有改成接受。</b> 默认匹配回归命令会报错，提醒问题尚未解决。本轮未修改生产算法掩盖失败。</p><p class="note">这些是可复现的派生安全反例，不是假称原始采集时就没有 A；生成区域、背景来源和全部图片指纹均保存。</p></section>
<section class="card"><h2>其余验证结果</h2><ul><li>新真实图套件：80 项通过（78 帧回放及 2 项数据/版本审计；版本核对后重跑相关项）。</li><li>真实图派生控制套件：23 项通过、上述 4 项失败。</li><li>替换原跳过项的 YOLO 实测检查：1 项通过。</li></ul><p>本轮相关用例合计 <b>104 项通过、4 项失败</b>。真实图首次回放中的 6 个旧版本差异，已完成下面的版本审计并单独留档，不能与这两个已确认反例混淆。</p></section>
<section class="card"><h2>没有把旧版本结果当成人工正确答案</h2><p>70 张有原始掩码的新真实图保持历史结果。8 张无掩码旧图中，6 张与历史记录不同；用此前已提交版本 <code>7cb7544</code> 的原选择函数回放，也出现同样的差异。</p><p>因此保留所有旧结果，单独记录该提交的兼容基线及旧/新中心，不用当前运行输出自动覆盖期望。两套结果的颗粒身份仍未人工确认。</p><details><summary>查看 6 个已留档的版本差异</summary><table><thead><tr><th>旧报告号 / sequence</th><th>历史记录</th><th>已提交版本</th><th>所选中心差</th></tr></thead><tbody>__LEGACY__</tbody></table></details>
<p class="note">历史任务整体 FAILED 也不等于匹配失败：两个 EDS 失败但匹配成功的案例已分别保留正确的匹配状态。</p></section>
<section class="card"><h2>运行与证据</h2><p><code>npm run test:matching</code></p><p>该入口已包含新套件；当前预期因上述两个尚未修复的安全反例报错。</p><p>原始采图：<code>backend/tests/fixtures/matching_expansion_20260918</code><br>派生图：<code>backend/tests/fixtures/real_matching_controls_20260918</code></p><p>原图、原始光镜掩码、模型与输入指纹均保留；无掩码旧数据明确标识，不补造掩码。</p><p class="note">本轮只扩充测试和记录发现，没有操作电镜、采 EDS、覆盖历史报告或打包安装。</p></section>
</main></body></html>'''
for key,value in {'__FIGURES__':''.join(figures),'__SOURCES__':source_rows,'__LEGACY__':legacy_rows}.items():page=page.replace(key,value)
(OUT/'report.html').write_text(page,encoding='utf-8')
print(OUT/'report.html')
