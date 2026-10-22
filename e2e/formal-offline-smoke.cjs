const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const { _electron: electron } = require('playwright')

const root = path.resolve(__dirname, '..')
const testRoot = path.join(root, '.tmp', 'formal-offline-ui')
const taskRoot = path.join(testRoot, `tasks-${process.pid}-${Date.now()}`)
const opticalJson = 'C:\\path\\to\\optical-export.json'

async function main() {
  assert.equal(fs.existsSync(opticalJson), true, `Missing formal optical export: ${opticalJson}`)
  fs.mkdirSync(testRoot, { recursive: true })
  const packagedExecutable = process.env.ZEM_PACKAGED_EXE
  if (packagedExecutable) assert.equal(fs.existsSync(packagedExecutable), true, `Missing packaged application: ${packagedExecutable}`)
  const app = await electron.launch({
    ...(packagedExecutable ? { executablePath: packagedExecutable } : { args: ['.'] }),
    cwd: packagedExecutable ? path.dirname(packagedExecutable) : root,
    env: {
      ...process.env,
      ZEM_TASK_ROOT: taskRoot,
    },
  })

  try {
    await app.evaluate(({ dialog }, selectedPath) => {
      dialog.showOpenDialog = async () => ({ canceled: false, filePaths: [selectedPath] })
    }, opticalJson)

    const page = await app.firstWindow()
    const consoleErrors = []
    page.on('console', (message) => {
      if (message.type() === 'error') consoleErrors.push(message.text())
    })
    page.on('pageerror', (error) => consoleErrors.push(error.message))

    await page.getByRole('heading', { name: '首页' }).waitFor()
    await page.getByText('后端已连接').waitFor({ timeout: 60000 })
    await page.getByRole('button', { name: /^电镜(已连接|忙碌|离线|未知)$/ }).waitFor({ timeout: 60000 })
    await page.getByRole('button', { name: /^能谱(已连接|忙碌|离线|未知)$/ }).waitFor({ timeout: 60000 })
    await page.getByRole('button', { name: '新建检测任务', exact: true }).click()
    await page.getByRole('heading', { name: '导入与筛选' }).waitFor()
    await page.getByRole('button', { name: /请选择光镜颗粒 JSON/ }).click()
    await page.getByText('389 个有效颗粒').waitFor({ timeout: 60000 })
    await page.getByLabel('最小值').fill('70')
    await page.getByLabel('最大值').fill('500')
    await page.getByRole('button', { name: '应用筛选' }).click()
    await page.getByText('已选 288 / 389').waitFor({ timeout: 60000 })
    await page.getByRole('button', { name: /下一步：坐标标定/ }).click()
    await page.getByRole('heading', { name: '坐标标定' }).waitFor({ timeout: 60000 })

    const tasks = await page.evaluate(() => window.zemCleanliness.request('GET', '/tasks'))
    assert.equal(tasks.tasks.length, 1)
    const manifest = await page.evaluate((taskId) => window.zemCleanliness.request('GET', `/tasks/${taskId}`), tasks.tasks[0].id)
    assert.equal(manifest.status, 'DRAFT')
    assert.equal(manifest.input.particleCount, 389)
    assert.equal(manifest.selection.count, 288)
    assert.equal(manifest.request.gunOffAfterTask, undefined)

    await page.getByRole('button', { name: /导入与筛选/ }).click()
    await page.getByRole('heading', { name: '导入与筛选' }).waitFor()
    assert.equal(await page.getByLabel('最小值').inputValue(), '70')
    assert.equal(await page.getByLabel('最大值').inputValue(), '500')
    await page.getByLabel('最小值').fill('80')
    const selectedBeforeRevision = await page.getByText(/^已选 \d+ \/ \d+$/).textContent()
    await page.getByRole('button', { name: '应用筛选' }).click()
    await page.waitForFunction((previous) => Array.from(document.querySelectorAll('.sectionNote')).some((element) => /^已选 \d+ \/ \d+$/.test(element.textContent || '') && element.textContent !== previous), selectedBeforeRevision)
    await page.getByRole('button', { name: /下一步：坐标标定/ }).click()
    await page.getByRole('heading', { name: '坐标标定' }).waitFor()
    const revisedTasks = await page.evaluate(() => window.zemCleanliness.request('GET', '/tasks'))
    const revisedManifest = await page.evaluate((taskId) => window.zemCleanliness.request('GET', `/tasks/${taskId}`), tasks.tasks[0].id)
    assert.equal(revisedTasks.tasks.length, 1)
    assert.equal(revisedTasks.tasks[0].id, tasks.tasks[0].id)
    assert.equal(revisedManifest.revision, 1)
    assert.notEqual(revisedManifest.selection.count, 288)
    assert.deepEqual(consoleErrors, [])

    await page.screenshot({ path: path.join(testRoot, 'formal-optical-calibration.png'), fullPage: true })
    console.log('formal_json_ui=ok')
    console.log('formal_particles=389')
    console.log('filtered_particles=288')
    console.log('draft_persistence=ok')
    console.log('step_back_revision=ok')
    console.log('duplicate_task_prevented=ok')
    console.log('device_status_rendered=ok')
    console.log('hardware_actions=none')
  } finally {
    await app.close()
  }
}

main().catch((error) => {
  console.error(error)
  process.exitCode = 1
})
