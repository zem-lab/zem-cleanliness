const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const { _electron: electron } = require('playwright')

const root = path.resolve(__dirname, '..')
const markerPath = process.env.ZEM_ACCEPTANCE_TEST_FILE
const screenshotRoot = path.join(root, '.tmp', 'hardware-acceptance')

function log(message) {
  const line = `${new Date().toISOString()} ${message}`
  fs.appendFileSync(path.join(screenshotRoot, 'acceptance.log'), `${line}\n`)
  console.log(message)
}

async function main() {
  assert.ok(markerPath && fs.existsSync(markerPath), 'ZEM_ACCEPTANCE_TEST_FILE is required')
  const options = JSON.parse(fs.readFileSync(markerPath, 'utf8'))
  assert.ok(fs.existsSync(options.opticalJson), `Missing optical JSON: ${options.opticalJson}`)
  fs.mkdirSync(screenshotRoot, { recursive: true })

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
    const acceptance = await page.evaluate(() => window.zemCleanliness.getAcceptanceTest())
    assert.equal(acceptance.fakeEds, true)
    assert.equal(acceptance.opticalJson, options.opticalJson)
    log('preflight=ok')

    await page.getByRole('button', { name: '新建检测任务', exact: true }).click()
    await page.getByRole('heading', { name: '导入与筛选' }).waitFor()
    const loaded = await page.evaluate(async (inputPath) => window.zemCleanliness.request('POST', '/optical/preview', {
      inputPath,
      filters: { minLengthUm: 224.8, maxLengthUm: 731.3 },
    }), options.opticalJson)
    assert.equal(loaded.selectedCount, options.expectedParticles)

    // Use the same UI state transitions a customer sees after file selection.
    await app.evaluate(({ dialog }, selectedPath) => {
      dialog.showOpenDialog = async () => ({ canceled: false, filePaths: [selectedPath] })
    }, options.opticalJson)
    await page.getByRole('button', { name: /请选择光镜颗粒 JSON/ }).click()
    await page.getByText('389 个有效颗粒').waitFor({ timeout: 60000 })
    await page.getByLabel('最小值').fill('224.8')
    await page.getByLabel('最大值').fill('731.3')
    await page.getByRole('button', { name: '应用筛选' }).click()
    await page.getByText(`已选 ${options.expectedParticles} / 389`).waitFor({ timeout: 60000 })
    await page.screenshot({ path: path.join(screenshotRoot, '01-import-filter.png'), fullPage: true })
    log(`import_filter=ok particles=${options.expectedParticles}`)

    await page.getByRole('button', { name: /下一步：坐标标定/ }).click()
    await page.getByRole('heading', { name: '坐标标定' }).waitFor({ timeout: 60000 })
    await page.getByRole('button', { name: 'AB 点法' }).click()
    await page.getByRole('button', { name: '获取导航图' }).click()
    await page.getByAltText('完整 SEM 导航图').waitFor({ timeout: 120000 })
    await page.getByText(/2 个候选孔/).waitFor({ timeout: 120000 })
    await page.screenshot({ path: path.join(screenshotRoot, '02-navigation.png'), fullPage: true })
    log('navigation=ok')

    await page.getByRole('button', { name: '开始 SEM 精定位' }).click()
    await page.getByText('SEM 精定位与验证拍摄完成，请逐张对比后确认').waitFor({ timeout: 900000 })
    await page.getByText('光镜 / SEM 对比验收').waitFor()
    const evidence = await page.locator('.abEvidenceCard').count()
    assert.ok(evidence > 0, 'No calibration evidence was generated')
    await page.locator('.abEvidenceList').screenshot({ path: path.join(screenshotRoot, '03-calibration-evidence.png') })
    log(`calibration_refine=ok evidence=${evidence}`)

    await page.getByRole('button', { name: /对比一致，确认标定/ }).click()
    await page.getByRole('heading', { name: '检测运行' }).waitFor({ timeout: 120000 })
    await page.getByRole('button', { name: '开始检测' }).click()
    await page.getByText('检测完成，报告已生成').waitFor({ timeout: 3600000 })
    const taskId = await page.locator('.runHeader').locator('span').first().textContent().then((value) => String(value).split(' · ')[0])
    const manifest = await page.evaluate((id) => window.zemCleanliness.request('GET', `/tasks/${id}`), taskId)
    assert.ok(['COMPLETED', 'COMPLETED_WITH_FAILURES'].includes(manifest.status), manifest.status)
    assert.equal(manifest.selection.count, options.expectedParticles)
    assert.equal(manifest.particles.length, options.expectedParticles)
    assert.equal(manifest.acceptanceTest?.fakeEds, true)
    assert.ok(manifest.particles.every((item) => item.energySpectrum?.mock === true || item.status === 'FAILED'))
    assert.ok(manifest.reports?.pdf && manifest.reports?.csv && manifest.reports?.html, JSON.stringify(manifest.reports))
    await page.screenshot({ path: path.join(screenshotRoot, '04-analysis-complete.png'), fullPage: true })
    log(`analysis=done status=${manifest.status} succeeded=${manifest.particles.filter((item) => item.status === 'SUCCEEDED').length} failed=${manifest.particles.filter((item) => item.status !== 'SUCCEEDED').length}`)

    await page.getByRole('button', { name: '报告记录' }).click()
    await page.getByRole('heading', { name: '报告记录' }).waitFor()
    const row = page.locator('.reportsTable .tableRow').filter({ hasText: taskId })
    await row.waitFor({ timeout: 60000 })
    await row.getByRole('button', { name: 'PDF' }).waitFor()
    await row.getByRole('button', { name: 'CSV' }).waitFor()
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
