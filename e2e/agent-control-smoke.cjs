// 防退化：ZEM_AGENT_CONTROL=1 启动时显示「正在被 Agent 控制」徽标与淡蓝描边；
// 正常启动（无该环境变量）时必须完全不出现，且两种模式都无 console 错误。
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const { _electron: electron } = require('playwright')

const root = path.resolve(__dirname, '..')
const testRoot = path.join(root, '.tmp', 'agent-control-smoke')

async function launchApp(agentControlled) {
  fs.mkdirSync(testRoot, { recursive: true })
  const env = { ...process.env, ZEM_TASK_ROOT: path.join(testRoot, 'tasks') }
  if (agentControlled) env.ZEM_AGENT_CONTROL = '1'
  else delete env.ZEM_AGENT_CONTROL
  const app = await electron.launch({ args: ['.'], cwd: root, env })
  const page = await app.firstWindow()
  const consoleErrors = []
  page.on('console', (message) => {
    if (message.type() === 'error') consoleErrors.push(message.text())
  })
  page.on('pageerror', (error) => consoleErrors.push(error.message))
  await page.getByRole('heading', { name: '首页' }).waitFor()
  await page.getByText('后端已连接').waitFor()
  return { app, page, consoleErrors }
}

async function main() {
  {
    const { app, page, consoleErrors } = await launchApp(true)
    await page.getByText('当前软件正在被 Agent 控制').waitFor()
    assert.equal(await page.locator('.agentControlBadge').isVisible(), true)
    assert.equal(await page.locator('.agentControlFrame').isVisible(), true)
    const runtime = await page.evaluate(() => window.zemCleanliness.getRuntime())
    assert.equal(runtime.agentControlled, true)
    await page.screenshot({ path: path.join(testRoot, 'agent-mode.png'), fullPage: true })
    assert.deepEqual(consoleErrors, [])
    await app.close()
  }
  {
    const { app, page, consoleErrors } = await launchApp(false)
    assert.equal(await page.locator('.agentControlBadge').count(), 0)
    assert.equal(await page.locator('.agentControlFrame').count(), 0)
    const runtime = await page.evaluate(() => window.zemCleanliness.getRuntime())
    assert.notEqual(runtime.agentControlled, true)
    assert.deepEqual(consoleErrors, [])
    await app.close()
  }
  console.log('agent_control_indicator=ok (徽标+蓝框仅在 ZEM_AGENT_CONTROL=1 时显示，两种启动均无 console 错误)')
}

main().catch((error) => {
  console.error(error)
  process.exit(1)
})
