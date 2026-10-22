// One UI smoke check using an in-memory desktop bridge. No hardware/network API.
const { chromium } = require('playwright')
const fs = require('node:fs')
const assert = require('node:assert/strict')
;(async () => {
  const browser = await chromium.launch({ headless: true, channel: 'chrome', args: ['--no-proxy-server'] })
  const page = await browser.newPage({ viewport: { width: 1440, height: 1100 } })
  const errors = []
  page.on('pageerror', e => errors.push(e.message))
  await page.addInitScript(() => {
    const nav = `<svg xmlns="http://www.w3.org/2000/svg" width="600" height="600"><defs><pattern id="g" width="25" height="25" patternUnits="userSpaceOnUse"><rect width="25" height="25" fill="#343d45"/><circle cx="10" cy="12" r="2" fill="#a3adb6"/></pattern></defs><rect width="600" height="600" fill="#1b2530"/><circle cx="300" cy="300" r="245" fill="url(#g)" stroke="#687480" stroke-width="12"/><text x="300" y="35" text-anchor="middle" fill="white" font-size="14">SIMULATED NAVIGATION — UI CHECK ONLY</text></svg>`
    let task = { taskId: 'ui-only', mode: 'sem_direct', status: 'DRAFT', input: { workpieceName: 'UI check' }, particles: [], request: { edsEnabled: true, gunOffAfterTask: true }, navigation: { boundsUm: [0, 0, 6000, 6000], imageShape: [600, 600], path: 'navigation.png', fieldUm: 1000, magnification: 127 }, previewDataUrl: 'data:image/svg+xml,'+encodeURIComponent(nav) }
    window.zemCleanliness = {
      getRuntime: async () => ({ backendBaseUrl: 'http://unused.invalid', version: 'prototype', logDirectory: '', packaged: false }),
      log: async () => {}, openPath: async () => {},
      request: async (method, route, body) => {
        if (route === '/tasks') return { tasks: [] }
        if (route === '/devices') return { devices: [] }
        if (route === '/models') return { models: [] }
        if (route === '/sem/navigation') return structuredClone(task)
        if (route.startsWith('/sem/active/')) return { job: null }
        if (route.includes('/sem/progress')) return {cursor:'mock:1',reset:true,task:structuredClone(task),tiles:structuredClone(task.plan.tiles)}
        if (route === '/jobs/mock-job/cancel') { task.status='STOPPED'; task.reportState='DEFERRED'; task.reports={}; return {} }
        if (route.endsWith('/sem/report')) { window.__reportRequested=true; return { id:'report-job',taskId:task.taskId,operation:'sem_report',status:'running',message:'生成报告' } }
        if (route === '/jobs/report-job') { task.reportState='READY'; task.reports={pdf:'sem_report.pdf'}; return {id:'report-job',taskId:task.taskId,operation:'sem_report',status:'completed',message:'报告已生成'} }
        if (route.endsWith('/sem/plan')) {
          window.__segmentation = body.segmentation
          task.model = body.segmentation
          const tiles = []
          for (let y=body.roi.y0+500; y<body.roi.y1; y+=850) for(let x=body.roi.x0+500; x<body.roi.x1; x+=850) tiles.push({ id: tiles.length+1, x, y, status: 'PENDING' })
          task.plan = { ...body, areaMm2: (body.roi.x1-body.roi.x0)**2/1e6, tiles, tileCount: tiles.length }
          return structuredClone(task)
        }
        if (route.endsWith('/sem/start')) { window.__semStarted = true; task.status='RUNNING'; return { id: 'mock-job', taskId: task.taskId, operation: 'sem_analysis', status: 'running', progress: 0, total: task.plan.tileCount, message: '模拟作业，仅检查界面' } }
        if (route === '/jobs/mock-job') return { id: 'mock-job', taskId: task.taskId, operation:'sem_analysis', status: task.status==='STOPPED'?'completed':'running', message: '模拟作业，仅检查界面', progress: 0, total: task.plan.tileCount }
        if (route === '/tasks/ui-only') return structuredClone(task)
        throw Error('Unexpected mock route: '+method+' '+route)
      }
    }
  })
  try {
    await page.goto(process.env.SEM_E2E_URL || 'http://127.0.0.1:5192', { waitUntil: 'domcontentloaded' })
    await page.getByRole('button', { name: '新建检测任务', exact: true }).click()
    await page.getByRole('button', { name: /直接 SEM 分析/ }).click()
    await page.getByRole('button', { name: '获取导航图', exact: true }).first().click()
    const canvas = page.getByLabel('拖动绘制扫描区域')
    await canvas.waitFor()
    const rect = await canvas.boundingBox()
    await page.mouse.move(rect.x+rect.width*.2, rect.y+rect.height*.2)
    await page.mouse.down()
    await page.mouse.move(rect.x+rect.width*.75, rect.y+rect.height*.75, { steps: 5 })
    await page.mouse.up()
    await page.locator('label').filter({ hasText: '颗粒分割模型' }).locator('select').selectOption('sam3')
    assert.equal(await page.getByLabel('SAM3 服务地址', { exact: true }).inputValue(), 'http://192.168.0.42:8765')
    assert.equal(await page.getByLabel('SAM3 置信度阈值').inputValue(), '0.1')
    await page.getByLabel('SAM3 置信度阈值').fill('0.05')
    await page.getByRole('button', { name: '预览扫描网格' }).click()
    await page.waitForFunction(() => document.querySelectorAll('.semGrid').length > 0)
    assert.equal((await page.evaluate(() => window.__segmentation)).scoreThreshold, 0.05)
    assert.equal((await page.evaluate(() => window.__segmentation)).texts.length, 6)
    await page.locator('label').filter({ hasText: '颗粒分割模型' }).locator('select').selectOption('yolo')
    assert.equal(await page.getByLabel('SAM3 服务地址', { exact: true }).count(), 0)
    await page.locator('label').filter({ hasText: '颗粒分割模型' }).locator('select').selectOption('sam3')
    assert(await page.getByRole('button', { name: '开始自动分析' }).isEnabled())
    fs.mkdirSync('artifacts/sem-direct', { recursive: true })
    await page.screenshot({ path: 'artifacts/sem-direct/sam3-switch.png', fullPage: true })
    await page.getByRole('button', { name: '开始自动分析' }).click()
    await page.getByRole('button', { name: '暂停扫描' }).waitFor()
    assert(await page.evaluate(() => window.__semStarted))
    assert(await page.locator('label').filter({ hasText: '颗粒分割模型' }).locator('select').isDisabled())
    assert(await page.getByRole('button',{name:'生成当前结果报告'}).isDisabled())
    await page.getByRole('button',{name:'暂停扫描'}).click()
    await page.waitForFunction(() => !Array.from(document.querySelectorAll('button')).find(b=>b.textContent.includes('生成当前结果报告'))?.disabled)
    assert.equal(await page.getByRole('button',{name:'打开正式 PDF 报告'}).count(),0)
    await page.getByRole('button',{name:'生成当前结果报告'}).click()
    await page.getByRole('button',{name:'打开正式 PDF 报告'}).waitFor()
    assert(await page.evaluate(()=>window.__reportRequested))
    assert.deepEqual(errors, [])
    console.log('Time optimization UI: running export lock, deferred report, explicit export passed; no hardware calls')
  } finally { await browser.close() }
})().catch(e => { console.error(e); process.exitCode=1 })
