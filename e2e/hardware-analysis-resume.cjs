const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const { _electron: electron } = require('playwright')

const root = path.resolve(__dirname, '..')
const markerPath = process.env.ZEM_ACCEPTANCE_TEST_FILE
const taskId = process.env.ZEM_ACCEPTANCE_TASK_ID
const screenshotRoot = path.join(root, '.tmp', 'hardware-acceptance')

function log(message) {
  const line = `${new Date().toISOString()} ${message}`
  fs.mkdirSync(screenshotRoot, { recursive: true })
  fs.appendFileSync(path.join(screenshotRoot, 'acceptance.log'), `${line}\n`)
  console.log(message)
}

async function main() {
  assert.ok(markerPath && fs.existsSync(markerPath), 'ZEM_ACCEPTANCE_TEST_FILE is required')
  assert.ok(taskId, 'ZEM_ACCEPTANCE_TASK_ID is required')
  const options = JSON.parse(fs.readFileSync(markerPath, 'utf8'))
  const app = await electron.launch({ args: ['.'], cwd: root, env: { ...process.env, ZEM_ACCEPTANCE_TEST_FILE: markerPath } })
  const errors = []
  try {
    const page = await app.firstWindow()
    page.on('console', (message) => { if (message.type() === 'error') errors.push(message.text()) })
    page.on('pageerror', (error) => errors.push(error.message))
    page.on('requestfailed', (request) => errors.push(`${request.method()} ${request.url()} ${request.failure()?.errorText || ''}`))

    await page.getByRole('heading', { name: '首页' }).waitFor({ timeout: 60000 })
    await page.getByText('后端已连接').waitFor({ timeout: 60000 })
    await page.getByRole('button', { name: '电镜已连接' }).waitFor({ timeout: 60000 })
    const taskRow = page.locator('.tableRow').filter({ hasText: taskId })
    await taskRow.waitFor({ timeout: 60000 })
    await taskRow.click()
    await page.getByRole('heading', { name: '检测运行' }).waitFor({ timeout: 60000 })
    const resumeButton = page.getByRole('button', { name: /继续剩余 \d+ 个/ })
    await resumeButton.waitFor({ timeout: 60000 })
    await page.screenshot({ path: path.join(screenshotRoot, '04a-stale-analysis-recovery.png'), fullPage: true })
    log(`analysis_recovery_ui=ok task=${taskId}`)

    await resumeButton.click()
    await page.getByText(/检测完成，报告已生成|有异常/).waitFor({ timeout: 3600000 })
    const manifest = await page.evaluate((id) => window.zemCleanliness.request('GET', `/tasks/${id}`), taskId)
    assert.ok(['COMPLETED', 'COMPLETED_WITH_FAILURES'].includes(manifest.status), manifest.status)
    assert.equal(manifest.selection.count, options.expectedParticles)
    assert.equal(manifest.particles.length, options.expectedParticles)
    assert.equal(manifest.acceptanceTest?.fakeEds, true)
    assert.ok(manifest.analysisResumeHistory?.length >= 1, 'Interrupted run was not resumed')
    assert.ok(manifest.particles.every((item) => item.energySpectrum?.mock === true || item.status === 'FAILED'))
    assert.ok(manifest.reports?.pdf && manifest.reports?.csv && manifest.reports?.html, JSON.stringify(manifest.reports))
    for (const kind of ['pdf', 'csv', 'html']) {
      const reportPath = path.join(manifest.taskDirectory, manifest.reports[kind])
      assert.ok(fs.existsSync(reportPath), `Missing ${kind} report: ${reportPath}`)
      assert.ok(fs.statSync(reportPath).size > 0, `Empty ${kind} report: ${reportPath}`)
    }
    const html = fs.readFileSync(path.join(manifest.taskDirectory, manifest.reports.html), 'utf8')
    const csv = fs.readFileSync(path.join(manifest.taskDirectory, manifest.reports.csv), 'utf8')
    assert.match(html, /验收模式 · 模拟能谱/)
    assert.match(html, /不代表真实材料成分/)
    assert.match(html, /Mock EDS \(acceptance test\)/)
    assert.match(csv, /edsDevice/)
    assert.match(csv, /Mock EDS \(acceptance test\)/)
    await page.screenshot({ path: path.join(screenshotRoot, '04-analysis-complete.png'), fullPage: true })
    log(`analysis=done status=${manifest.status} succeeded=${manifest.particles.filter((item) => item.status === 'SUCCEEDED').length} failed=${manifest.particles.filter((item) => item.status !== 'SUCCEEDED').length}`)

    await page.getByRole('button', { name: '报告记录' }).click()
    await page.getByRole('heading', { name: '报告记录' }).waitFor()
    const reportRow = page.locator('.reportsTable .tableRow').filter({ hasText: taskId })
    await reportRow.waitFor({ timeout: 60000 })
    await reportRow.getByRole('button', { name: 'PDF' }).waitFor()
    await reportRow.getByRole('button', { name: 'CSV' }).waitFor()
    await page.screenshot({ path: path.join(screenshotRoot, '05-report-list.png'), fullPage: true })
    log(`report_ui=ok task=${taskId}`)

    fs.writeFileSync(path.join(screenshotRoot, 'result.json'), JSON.stringify({ taskId, manifest, errors }, null, 2))
    assert.deepEqual(errors, [])
    log('customer_flow=complete')
  } finally {
    await app.close()
  }
}

main().catch((error) => {
  console.error(error)
  try { log(`failure=${error.stack || error}`) } catch {}
  process.exitCode = 1
})
