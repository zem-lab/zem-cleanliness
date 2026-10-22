"""Build a self-contained HTML report from sample 2 manual AB evidence."""
import base64
import html
import io
import json
from pathlib import Path
from datetime import datetime
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/manual_ab_sample2_20260908'


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def uri(path): return 'data:image/png;base64,' + base64.b64encode(Path(path).read_bytes()).decode()
def esc(value): return html.escape(str(value))
def download(path, label):
    return f'<a class="download" download="{esc(path.name)}" href="data:application/json;base64,{base64.b64encode(path.read_bytes()).decode()}">{label} ↓</a>'


def main():
    task = OUT / 'sample2-large-particles'
    manifest = read(task / 'task.json')
    first = read(OUT / 'sample2-manual-ab/task.json')
    assessment = read(OUT / 'assessment.json')
    axis = read(OUT / 'axis-particle/axis.json')
    optical = json.loads(Path(manifest['input']['path']).read_text(encoding='utf-8-sig'))
    particles = {p['sequence']: p for p in optical['particles']}
    ab = manifest['calibration']['abPointResult']
    refinement = ab['semRefinement']
    transform = refinement['refinedTransform']
    matrix = np.array(transform['matrix']); translation = np.array(transform['translation'])
    reviewed_path = OUT / 'visual_review.json'
    reviewed = read(reviewed_path) if reviewed_path.exists() else {'confirmedSequences': []}
    confirmed = [r for r in assessment['rows'] if r['sequence'] in reviewed['confirmedSequences'] and r.get('measuredOffsetUm') is not None]
    errors = [r['measuredOffsetUm'] for r in confirmed]
    mean = f'{np.mean(errors):.1f}' if errors else '待核实'
    maximum = f'{max(errors):.1f}' if errors else '待核实'
    predictions = []
    for p in particles.values():
        target = matrix @ [p['abPosition']['x'], p['abPosition']['y']] + translation
        predictions.append({'sequence': p['sequence'], 'stageUm': target.tolist()})
    prediction_path = OUT / 'manual_ab_predictions.json'
    prediction_path.write_text(json.dumps({'input': manifest['input'], 'transform': transform,
        'acceptedForAutomaticAnalysis': False, 'note': 'Manual AB predictions only; independent image offsets must be reviewed.',
        'particles': predictions}, ensure_ascii=False, indent=2), encoding='utf-8')
    cards, rows = [], []
    for r in assessment['rows']:
        seq = r['sequence']; p = particles[seq]
        ok = seq in reviewed['confirmedSequences']
        status = '形貌对应已核对' if ok else '未确认匹配'
        metric = f"{r['measuredOffsetUm']:.1f} μm" if ok and r.get('measuredOffsetUm') is not None else '不计入误差统计'
        sem = task / r['images']['semDisplayImage'] if r['images'].get('semDisplayImage') else None
        pair = f'<figure><img class="zoom" src="data:image/png;base64,{p["imageDataBase64"]}" alt="颗粒 {seq} 光镜 ROI"><figcaption>光镜原始 ROI · {p["particleInfo"]["length"]:.1f} μm</figcaption></figure>'
        if sem:
            img = Image.open(sem).convert('RGB')
            center = r.get('centerPixel', [256, 256]); radius = max(50, p['particleInfo']['length'] / r.get('pixelSizeUm', 2.067) * .7)
            crop = img.crop((max(0, int(center[0]-radius)), max(0, int(center[1]-radius)), min(img.width, int(center[0]+radius)), min(img.height, int(center[1]+radius))))
            memory = io.BytesIO(); crop.save(memory, format='PNG')
            pair += f'<figure><img class="zoom" src="data:image/png;base64,{base64.b64encode(memory.getvalue()).decode()}" alt="颗粒 {seq} SEM 候选局部"><figcaption>SEM 候选局部 · {status}</figcaption></figure>'
        full = OUT / r['comparisonImage'] if r.get('comparisonImage') else sem
        button = f'<button class="frame" data-image="{uri(full)}" data-caption="颗粒 #{seq}：绿色十字为预测中心，红圈为相关匹配候选；{status}">查看完整视场 ↗</button>' if full else ''
        cards.append(f'<article class="particle"><div class="particle-head"><h3>颗粒 #{seq}</h3><b>{metric}</b></div><div class="pair">{pair}</div><div class="particle-foot"><span>相关分数 {r.get("score", 0):.3f} · {status}</span>{button}</div></article>')
        rows.append(f'<tr><td>#{seq}</td><td>{p["particleInfo"]["length"]:.1f}</td><td>{r.get("score",0):.3f}</td><td>{r.get("postCaptureErrorUm",r.get("moveErrorUm",0)):.2f}</td><td>{metric}</td><td>{status}</td></tr>')
    first_records = first['calibration']['abPointResult']['semRefinement']['validation']
    first_rows = ''.join(f'<tr><td>#{r["sequence"]}</td><td>{r["lengthUm"]:.1f}</td><td>{esc(r.get("error") or "已拍摄；仅作补充证据，未计入定量统计")}</td></tr>' for r in first_records)
    formula = '\n'.join(f'{axis_name} = {matrix[i,0]:.12f} × abX {matrix[i,1]:+.12f} × abY {translation[i]:+.6f}' for i,axis_name in enumerate(['X','Y']))
    dots = ''.join(f'<circle cx="{(p["stageUm"][0]+24000)/48000*680+30:.2f}" cy="{(24000-p["stageUm"][1])/48000*430+25:.2f}" r="1.3" fill="#bbc9dc"/>' for p in predictions)
    for r in assessment['rows']:
        target = r.get('targetStageUm')
        if target:
            x=(target['x']+24000)/48000*680+30; y=(24000-target['y'])/48000*430+25
            dots += f'<circle cx="{x}" cy="{y}" r="5" fill="#2866aa"/><text x="{x+8}" y="{y-7}" font-size="14" fill="#173450">#{r["sequence"]}</text>'
    downloads = ''.join([download(prediction_path, '800 个颗粒预测坐标与 AB 矩阵'), download(OUT/'assessment.json', '独立实拍评估'), download(task/'task.json', '完整任务记录'), download(OUT/'axis-particle/axis.json', '实测像素轴')])
    final = axis.get('finalStageUm', [])
    tests = read(OUT / 'verification.json')
    summary = {'confirmedCount': len(confirmed), 'total': len(assessment['rows']), 'meanErrorUm': float(np.mean(errors)) if errors else None,
               'maxErrorUm': max(errors) if errors else None, 'axisValid': axis.get('valid'),
               'automaticAnalysisAccepted': False, 'finalStageUm': final, 'hvStatus': axis.get('hvStatus'), 'tests': tests}
    (OUT / 'report_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    doc = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>样品 2｜手动 AB 标定与电镜实测报告</title>
<style>
:root{{font-family:Inter,"Microsoft YaHei",sans-serif;color:#172b43;background:#eef2f6;line-height:1.65}}*{{box-sizing:border-box}}body{{margin:0}}main{{max-width:1160px;margin:auto;padding:36px 28px 70px}}header{{background:#173751;color:white;padding:36px;border-radius:18px}}.eyebrow{{color:#a5cced;letter-spacing:2px;font-size:12px}}h1{{font-size:32px;margin:8px 0}}h2{{font-size:22px;margin:0 0 18px}}h3{{margin:0;font-size:17px}}p{{margin:10px 0}}.intro{{color:#d4e3ef;max-width:840px}}.pills{{display:flex;flex-wrap:wrap;gap:8px;margin-top:22px}}.pill{{background:#ffffff17;border:1px solid #ffffff30;padding:5px 12px;border-radius:30px;font-size:13px}}section{{margin-top:24px;background:white;padding:28px;border-radius:16px;border:1px solid #dce4ec}}.metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:22px 0}}.metric{{padding:20px;background:white;border:1px solid #dce4ec;border-radius:12px}}.metric b{{display:block;font-size:29px;color:#1a537f}}.metric span{{font-size:13px;color:#64778a}}.notice{{padding:16px 20px;background:#fff4df;border-left:4px solid #d89d2b;border-radius:6px}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}.particle{{border:1px solid #d9e3ed;border-radius:12px;overflow:hidden}}.particle-head{{display:flex;justify-content:space-between;padding:14px 18px;background:#f4f7fb}}.particle-head b{{font-size:16px;color:#a86516}}.pair{{display:grid;grid-template-columns:1fr 1fr;gap:10px;padding:14px}}figure{{margin:0}}.pair img{{width:100%;height:190px;object-fit:contain;background:#f0f2f4;cursor:zoom-in}}figcaption{{font-size:11px;color:#637588;margin-top:6px}}.particle-foot{{padding:10px 15px;border-top:1px solid #e6ecf2;font-size:11px;display:flex;align-items:center;justify-content:space-between;gap:8px}}button,.download{{border:1px solid #bcd0e3;border-radius:6px;background:#f3f8ff;color:#20577f;padding:7px 10px;cursor:pointer;font:inherit;text-decoration:none}}.downloads{{display:flex;gap:10px;flex-wrap:wrap;margin-top:18px}}table{{width:100%;border-collapse:collapse;font-size:13px}}td,th{{padding:11px;text-align:left;border-bottom:1px solid #e1e8ee}}th{{background:#f1f5f9}}.table-wrap{{overflow-x:auto}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#142e43;color:#d9eefc;padding:20px;border-radius:10px;font-size:14px}}.muted{{color:#65788c;font-size:13px}}.full{{width:100%;height:auto;border:1px solid #e0e7ee;border-radius:8px}}code{{overflow-wrap:anywhere}}dialog{{border:0;border-radius:12px;max-width:94vw;max-height:95vh;padding:20px}}dialog::backdrop{{background:#102235dc}}dialog img{{display:block;max-width:85vw;max-height:75vh;object-fit:contain;margin:auto}}dialog button{{float:right}}footer{{text-align:center;color:#708293;font-size:12px;margin-top:30px}}@media(max-width:700px){{main{{padding:14px}}header,section{{padding:20px}}h1{{font-size:25px}}.metrics{{grid-template-columns:1fr 1fr}}.grid{{grid-template-columns:1fr}}.pair img{{height:150px}}.particle-foot{{flex-wrap:wrap}}}}
</style><main><header><div class="eyebrow">ZEPTOOLS · SEM CALIBRATION · 2026.09.08</div><h1>样品 2｜手动 AB 标定与电镜实测</h1><p class="intro">在 Tab2 直接输入操作员提供的 A/B 样品台坐标，复用原有 AB 两点变换与颗粒实拍流程。通过真实电镜验证输入、移台、采图与形貌对应，完整保留偏差与失败证据。</p><div class="pills"><span class="pill">软件功能验证通过</span><span class="pill">192.168.0.15 · 真机实拍</span><span class="pill">未自动确认标定 / 未进行 EDS</span></div></header>
<div class="metrics"><div class="metric"><b>{len(confirmed)} / {len(assessment['rows'])}</b><span>完成形貌核对的定量验证点</span></div><div class="metric"><b>{mean} <small>μm</small></b><span>平均颗粒偏移 · 冻结 AB 矩阵</span></div><div class="metric"><b>{maximum} <small>μm</small></b><span>最大颗粒偏移</span></div><div class="metric"><b>56.176 <small>mm</small></b><span>人工 AB 基线长度</span></div></div>
<div class="notice"><strong>结论：手动 AB 功能可用，直接 AB 定位仍存在颗粒偏移。</strong><p>软件正确接收坐标并使用原有 AB 法移台实拍；两点约束本身不能修正光镜全样品的局部偏差。上面的偏移来自图像核对，不是样品台回读误差。对精确居中或小颗粒分析，应在人工审核后决定是否进一步精校准；本报告没有把颗粒匹配结果回填到 AB 矩阵。</p></div>
<section><h2>01 · 输入与实际使用的变换</h2><div class="table-wrap"><table><tr><th>参考点</th><th>X（mm）</th><th>Y（mm）</th><th>内部 X / Y（μm）</th></tr><tr><td>A</td><td>-20.100</td><td>-18.890</td><td>-20100 / -18890</td></tr><tr><td>B</td><td>20.380</td><td>20.060</td><td>20380 / 20060</td></tr></table></div><p class="muted">用户原文中 A 点后一项“B点 -18.89”按 A 点 Y = -18.89 mm 理解。光镜数据含 {len(particles)} 个颗粒，参考基线 {transform['sourceBaselineUm']/1000:.6f} mm，缩放 {transform['scale']:.9f}。源坐标为 abPosition，所有公式变量单位均为 μm。</p><pre id="ab-code">{formula}</pre><p>光镜参考点先解析到 AB 坐标系，再使用原有手性推断与两点变换。A/B 映射在数值精度内吻合；两点拟合的零残差仅说明约束满足，不代表实际颗粒定位零误差。</p><details><summary>输入文件与可追溯信息</summary><p><code>{esc(manifest['input']['path'])}</code></p><p class="muted">SHA-256：<code>{esc(manifest['input']['sha256'])}</code></p></details></section>
<section><h2>02 · 软件新增入口与测试</h2><img class="full zoom" src="{uri(OUT/'tab2-manual-input.png')}" alt="Tab2 手动 AB 标定界面截图"><p class="muted">界面自动化测试截图，使用同一组样品 2 输入值；界面测试中的设备请求被拦截，实机证据来自下面的独立真实调用。</p><p>新增「手动 AB 标定」模式：毫米单位输入、完整性与数值范围校验、提交确认、直接建立 AB 变换、原有实拍验证、重新打开时恢复坐标与模式。保存接口不连接设备；实拍后仍需审核。</p><div class="table-wrap"><table><tr><th>检查</th><th>结果</th></tr><tr><td>前端 TypeScript / Vite 构建</td><td>通过</td></tr><tr><td>后端回归与手动 AB 专项</td><td>{esc(tests['backend'])}</td></tr><tr><td>Playwright：空值 / 越界、取消、mm 传递、提交顺序、重开任务</td><td>通过，0 浏览器错误</td></tr><tr><td>原有 AB 流程复用与不调用自动找孔</td><td>通过专项回归</td></tr></table></div></section>
<section><h2>03 · 分布与独立验证方式</h2><div class="grid"><div><svg viewBox="0 0 740 490" role="img" aria-label="800 个颗粒预测位置和 8 个验证点"><rect x="20" y="15" width="700" height="450" rx="12" fill="#f6f8fb"/>{dots}</svg></div><div><p>灰点为全部颗粒的 AB 预测台坐标，蓝点为本轮验证点。采用长度 ≥150 μm 的筛选条件，在符合条件的颗粒中按空间分散选择 8 点；矩阵从人工 A/B 直接得到，在拍摄验证图之前已固定。</p><p>图像以原始预测位置为中心拍摄，没有先把候选居中再报告误差。离线掩膜相关只用于找对应形貌，并经逐图核对；相关分数不是概率。像素到台坐标转换另行通过 ±100 μm 实际移台测量。</p><p class="muted">实测像素轴有效：{esc(axis.get('valid'))}；返回原点图像误差：{axis.get('returnErrorUm',float('nan')):.3f} μm；实测尺度主值：{esc([round(x,4) for x in axis.get('singularValuesUmPerPx',[])])} μm/px。台坐标回读误差与颗粒偏移分别列出。初次在周期滤膜区域的全图相关失败，未采用；本表使用已确认的大颗粒附近复测结果，原始失败证据保留在 axis 目录。</p></div></div><div class="table-wrap"><table><tr><th>颗粒</th><th>长度 μm</th><th>相关分数</th><th>采图台误差 μm</th><th>图像颗粒偏移</th><th>核对</th></tr>{''.join(rows)}</table></div></section>
<section><h2>04 · 光镜 / SEM 逐颗证据</h2><p class="muted">点击完整视场查看：绿色十字为 AB 预测中心；红圈为离线匹配候选。局部裁图用于检查形貌，偏移始终相对于完整图像的预测中心计算。</p><div class="grid">{''.join(cards)}</div></section>
<section><h2>05 · 首轮记录、设备结束状态与限制</h2><p>首轮从全部 800 个颗粒中按空间分散选取 8 点，其中含 8.4–46.4 μm 的小颗粒，形貌核对不充分，未计入本报告的定量均值。首张 #764 发生设备端“拍照超时”，其余 7 张返回图像。首轮证据保留，没有以第二轮覆盖。</p><details><summary>展开首轮全部记录</summary><div class="table-wrap"><table><tr><th>颗粒</th><th>长度 μm</th><th>结果</th></tr>{first_rows}</table></div></details><p>两轮实拍及像素轴测量均执行了返回起点。末次回读台坐标（μm）：<strong>{esc(final)}</strong>；电子枪状态回读：<strong>{esc(axis.get('hvStatus'))}</strong>。本次未开关电子枪，保持用户开枪状态；未运行 EDS、未点击确认标定。</p><p class="muted">手动 AB 的精度取决于输入参考点、光镜坐标以及样品安装几何；本次实测仅代表样品 2 与本次装夹。未确认的图像不填入误差统计，也不视为零误差。相关匹配是离线核对工具，未改变产品运行时的 AB 算法。</p><div class="downloads">{downloads}</div></section><footer>生成时间 {datetime.now().astimezone().isoformat(timespec='seconds')} · 所有图片与 JSON 下载均嵌入 HTML，可离线打开与分享。</footer></main>
<dialog id="lightbox"><button onclick="document.getElementById('lightbox').close()">关闭大图</button><p id="caption"></p><img id="full-image" alt="完整实拍证据"></dialog><script>const box=document.getElementById('lightbox');function show(src,caption){{document.getElementById('full-image').src=src;document.getElementById('caption').textContent=caption;box.showModal()}}document.querySelectorAll('.frame').forEach(b=>b.addEventListener('click',()=>show(b.dataset.image,b.dataset.caption)));document.querySelectorAll('img.zoom').forEach(i=>i.addEventListener('click',()=>show(i.src,i.alt)));box.addEventListener('click',e=>{{if(e.target===box)box.close()}});</script></html>'''
    path = OUT / '样品2_手动AB标定与电镜实测报告.html'
    path.write_text(doc, encoding='utf-8')
    print(path)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__': main()
