const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const { _electron: electron } = require('playwright')

const root = path.resolve(__dirname, '..')
const testRoot = path.join(root, '.tmp', 'electron-smoke')
const sample = path.join(root, 'backend', 'tests', 'fixtures', 'sample_cleanliness_report', 'optical.json')

async function main() {
  fs.mkdirSync(testRoot, { recursive: true })
  const app = await electron.launch({
    args: ['.'],
    cwd: root,
    env: {
      ...process.env,
      ZEM_TASK_ROOT: path.join(testRoot, 'tasks'),
    },
  })

  try {
    const page = await app.firstWindow()
    const consoleErrors = []
    page.on('console', (message) => {
      if (message.type() === 'error') consoleErrors.push(message.text())
    })
    page.on('pageerror', (error) => consoleErrors.push(error.message))

    await page.getByRole('heading', { name: '首页' }).waitFor()
    await page.getByText('后端已连接').waitFor()
    assert.equal(await page.getByText('今日统计').count(), 0)
    assert.equal(await page.getByText('设备状态').count(), 0)
    assert.equal(await page.getByText('最多执行').count(), 0)
    assert.equal(await page.getByText('任务结束后自动关枪').count(), 0)
    assert.equal(await page.getByText('暂无进行中的任务').count(), 1)

    await page.getByRole('button', { name: '系统设置' }).click()
    await page.getByRole('heading', { name: '系统设置' }).waitFor()
    assert.equal(await page.getByText('就绪', { exact: true }).count(), 3)
    await page.getByText('运行日志目录').waitFor()

    const runtime = await page.evaluate(() => window.zemCleanliness.getRuntime())
    assert.ok(runtime.logDirectory)

    const preview = await page.evaluate(async (inputPath) => window.zemCleanliness.request('POST', '/optical/preview', {
      inputPath,
      filters: { minLengthUm: 0 },
    }), sample)
    assert.equal(preview.particleCount, 5)
    const electronLog = path.join(runtime.logDirectory, 'electron.jsonl')
    const backendLog = path.join(runtime.logDirectory, 'backend.jsonl')
    assert.ok(fs.existsSync(electronLog))
    assert.ok(fs.existsSync(backendLog))
    assert.match(fs.readFileSync(electronLog, 'utf8'), /api_request_finished/)
    assert.match(fs.readFileSync(backendLog, 'utf8'), /http_request_finished/)

    await page.screenshot({ path: path.join(testRoot, 'desktop-smoke.png'), fullPage: true })
    assert.deepEqual(consoleErrors, [])
    console.log('electron_window=ok')
    console.log('empty_state=ok')
    console.log('removed_options=ok')
    console.log('models=3_ready')
    console.log('optical_preview=5')
    console.log('structured_logs=ok')
  } finally {
    await app.close()
  }
}

main().catch((error) => {
  console.error(error)
  process.exitCode = 1
})
