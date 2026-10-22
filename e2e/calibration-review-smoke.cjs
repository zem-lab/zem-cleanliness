const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const { _electron: electron } = require('playwright')

const root = path.resolve(__dirname, '..')
const opticalJson = 'C:\\path\\to\\optical-export.json'
const screenshotRoot = path.join(root, '.tmp', 'calibration-review-ui')

async function main() {
  assert.equal(fs.existsSync(opticalJson), true, `Missing formal optical export: ${opticalJson}`)
  fs.mkdirSync(screenshotRoot, { recursive: true })
  const app = await electron.launch({ args: ['.'], cwd: root })

  try {
    await app.evaluate(({ dialog }, selectedPath) => {
      dialog.showOpenDialog = async () => ({ canceled: false, filePaths: [selectedPath] })
    }, opticalJson)

    const page = await app.firstWindow()
    const errors = []
    page.on('console', (message) => { if (message.type() === 'error') errors.push(message.text()) })
    page.on('pageerror', (error) => errors.push(error.message))

    await page.getByRole('heading', { name: '首页' }).waitFor()
    await page.getByText('后端已连接').waitFor({ timeout: 60000 })

    await page.getByRole('button', { name: '新建检测任务', exact: true }).click()
    await page.getByRole('button', { name: /请选择光镜颗粒 JSON/ }).click()
    await page.getByText('389 个有效颗粒').waitFor({ timeout: 60000 })
    const mapMetrics = await page.locator('.particleMapCanvas').evaluate((element) => {
      const canvas = element.getBoundingClientRect()
      const circle = element.querySelector('circle')?.getBoundingClientRect()
      return { canvasWidth: canvas.width, canvasHeight: canvas.height, circleWidth: circle?.width || 0, circleHeight: circle?.height || 0 }
    })
    assert.ok(Math.abs(mapMetrics.canvasWidth - mapMetrics.canvasHeight) < 1, JSON.stringify(mapMetrics))
    assert.ok(Math.abs(mapMetrics.circleWidth - mapMetrics.circleHeight) < 0.15, JSON.stringify(mapMetrics))
    await page.screenshot({ path: path.join(screenshotRoot, 'particle-map.png'), fullPage: true })

    await page.getByRole('button', { name: '首页' }).click()
    const reviewTask = await page.evaluate(async () => {
      const summaries = await window.zemCleanliness.request('GET', '/tasks')
      for (const summary of summaries.tasks) {
        if (summary.status !== 'CALIBRATION_PENDING_REVIEW') continue
        const manifest = await window.zemCleanliness.request('GET', `/tasks/${summary.id}`)
        if (manifest.calibration?.strategy === 'ab_points' && manifest.calibration?.abPointResult?.semRefinement?.validation?.length) return summary
      }
      return null
    })
    assert.ok(reviewTask, 'No completed AB calibration review task was found')
    await page.locator('.tableRow').filter({ hasText: reviewTask.id }).click()
    await page.getByRole('heading', { name: '坐标标定' }).waitFor()
    await page.getByText('光镜 / SEM 对比验收').waitFor()
    await page.getByText('已取得 2 / 2 组光镜与 SEM 对比证据').waitFor()

    const navigation = await page.locator('.navigationViewport').evaluate((element) => {
      const viewport = element.getBoundingClientRect()
      const image = element.querySelector('img')
      return { width: viewport.width, height: viewport.height, naturalWidth: image?.naturalWidth || 0, naturalHeight: image?.naturalHeight || 0 }
    })
    assert.equal(navigation.naturalWidth, 1920)
    assert.equal(navigation.naturalHeight, 1080)
    assert.ok(Math.abs(navigation.width / navigation.height - 1920 / 1080) < 0.01, JSON.stringify(navigation))
    assert.equal(await page.getByText(/H1/).count(), 0)
    await page.locator('.navigationOverlay text').filter({ hasText: 'A 点（候选孔 1）' }).waitFor()
    await page.locator('.navigationOverlay text').filter({ hasText: 'B 点（候选孔 2）' }).waitFor()
    assert.equal(await page.locator('.abEvidenceCard').count(), 2)
    assert.equal(await page.locator('.comparisonImage img').count(), 4)
    const loadedImages = await page.locator('.comparisonImage img').evaluateAll((images) => images.map((image) => ({ complete: image.complete, width: image.naturalWidth, height: image.naturalHeight })))
    assert.ok(loadedImages.every((image) => image.complete && image.width > 0 && image.height > 0), JSON.stringify(loadedImages))
    await page.getByText('100%').waitFor()
    await page.getByRole('button', { name: /对比一致，确认标定/ }).waitFor()
    await page.getByRole('button', { name: /对比不一致，重新精定位/ }).waitFor()
    await page.screenshot({ path: path.join(screenshotRoot, 'ab-calibration-review.png'), fullPage: true })
    await page.locator('.abEvidenceList').scrollIntoViewIfNeeded()
    await page.locator('.abEvidenceList').screenshot({ path: path.join(screenshotRoot, 'ab-evidence-comparison.png') })

    assert.deepEqual(errors, [])
    console.log('particle_map_square=ok')
    console.log('particle_circle_round=ok')
    console.log('navigation_full_1920x1080=ok')
    console.log('navigation_labels_chinese=ok')
    console.log('refinement_progress=100%')
    console.log('optical_sem_evidence=2')
    console.log('hardware_actions=none')
  } finally {
    await app.close()
  }
}

main().catch((error) => {
  console.error(error)
  process.exitCode = 1
})
