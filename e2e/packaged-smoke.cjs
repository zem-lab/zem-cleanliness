const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const { spawnSync } = require('node:child_process')
const { _electron: electron } = require('playwright')

const root = path.resolve(__dirname, '..')
const executable = process.env.ZEM_PACKAGED_EXE || path.join(root, 'release', 'win-unpacked', 'ZEM Cleanliness.exe')
const unpackedRoot = path.dirname(executable)
const testRoot = path.join(root, '.tmp', 'packaged-smoke', `${Date.now()}-${process.pid}`)
const expectedVersion = JSON.parse(fs.readFileSync(path.join(root, 'package.json'), 'utf8')).version

async function main() {
  assert.equal(fs.existsSync(executable), true, `Missing packaged application: ${executable}`)
  fs.mkdirSync(testRoot, { recursive: true })
  const dependencyCheck = spawnSync(path.join(unpackedRoot, 'resources', 'backend', 'zem-cleanliness-backend.exe'), ['--check-runtime'], {
    env: { ...process.env, ZEM_TASK_ROOT: path.join(testRoot, 'runtime-check-tasks') },
    encoding: 'utf8', windowsHide: true, timeout: 60000,
  })
  assert.equal(dependencyCheck.status, 0, dependencyCheck.stderr || dependencyCheck.stdout)
  assert.match(dependencyCheck.stdout, /"runtimeCheck": "passed"/)
  assert.match(dependencyCheck.stdout, /"largeParticleCenterCheck": "passed"/)
  fs.writeFileSync(path.join(testRoot, 'runtime-check.txt'), dependencyCheck.stdout, 'utf8')
  console.log('packaged_driver_dependencies=ok')
  const app = await electron.launch({
    executablePath: executable,
    args: [`--user-data-dir=${path.join(testRoot, 'user-data')}`],
    cwd: unpackedRoot,
    env: {
      ...process.env,
      ZEM_TASK_ROOT: path.join(testRoot, 'tasks'),
      ZEM_MCP_SIMULATED: '1',
    },
  })

  try {
    const page = await app.firstWindow({ timeout: 60000 })
    const consoleErrors = []
    page.on('console', (message) => {
      if (message.type() === 'error') consoleErrors.push(message.text())
    })
    page.on('pageerror', (error) => consoleErrors.push(error.message))

    await page.getByRole('heading', { name: '首页' }).waitFor({ timeout: 60000 })
    await page.getByText('后端已连接').waitFor({ timeout: 60000 })
    const runtime = await page.evaluate(() => window.zemCleanliness.getRuntime())
    assert.equal(runtime.packaged, true)
    assert.equal(runtime.version, expectedVersion)
    const mcp = await page.evaluate(async (base) => (await fetch(base + '/chat/runtime')).json(), runtime.backendBaseUrl)
    assert.equal(mcp.mcp.mode, 'simulated')
    await page.getByRole('button', { name: '系统设置' }).click()
    await page.getByRole('heading', { name: '系统设置' }).waitFor()
    assert.equal(await page.getByText('就绪', { exact: true }).count(), 3)
    assert.equal(await page.getByLabel('拍照档位', { exact: true }).inputValue(), '7')
    assert.equal(await page.getByLabel('候选颗粒分差阈值', { exact: true }).inputValue(), '0.07')
    assert.equal(await page.getByLabel('模板相关分数阈值', { exact: true }).inputValue(), '0.65')
    assert.equal(await page.getByLabel('轮廓重叠', { exact: true }).inputValue(), '0.5')
    assert.equal(await page.getByLabel('拍照张数', { exact: true }).inputValue(), '2')
    await page.getByRole('switch', { name: '检测结束后自动关枪', exact: true }).waitFor()
    await page.getByRole('switch', { name: '显示光镜/电镜坐标', exact: true }).waitFor()
    await page.getByRole('switch', { name: '开工前状态检查', exact: true }).waitFor()
    await page.screenshot({ path: path.join(testRoot, 'packaged-smoke.png'), fullPage: true })
    assert.deepEqual(consoleErrors, [])
    console.log('packaged_app=ok')
    console.log('packaged_backend=connected')
    console.log('packaged_models=3_ready')
    console.log(`packaged_version=${runtime.version}`)
    console.log(`packaged_evidence=${testRoot}`)
  } finally {
    await app.close()
  }
}

main().catch((error) => {
  console.error(error)
  process.exitCode = 1
})
