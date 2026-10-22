/**
 * Vibe 模式 e2e 动作库：给 codex cli 调用的原子动作。
 *
 * 用法（在仓库根目录）：
 *   node e2e/vibe-actions.cjs detect --json "<光镜颗粒 JSON 绝对路径>" [--min <um>] [--max <um>] [--task-root <dir>]
 *
 * detect：完整链路 = 启动软件 → 导入 JSON → （可选）筛选 → AB 点标定 →
 *         SEM 精定位 → 确认标定 → 开始检测 → 等待报告生成。
 * 前提：电镜与能谱在线。该动作会真实驱动设备。
 *
 * 成功：stdout 最后一行输出 JSON 摘要，退出码 0。
 * 失败：stderr 输出 JSON {ok:false, error, screenshot}，退出码 1。
 * 过程截图保存在 .tmp/vibe-actions/<时间戳>/，出错时附在输出里便于回看。
 */

const fs = require('node:fs')
const path = require('node:path')
const { _electron: electron } = require('playwright')

const root = path.resolve(__dirname, '..')

function parseArgs(argv) {
  const args = { _: [] }
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index]
    if (!token.startsWith('--')) { args._.push(token); continue }
    const key = token.slice(2)
    const next = argv[index + 1]
    if (next === undefined || next.startsWith('--')) { args[key] = true; continue }
    args[key] = next
    index += 1
  }
  return args
}

function output(payload, toStderr = false) {
  const line = JSON.stringify(payload)
  if (toStderr) process.stderr.write(`${line}\n`)
  else console.log(line)
}

async function main() {
  const args = parseArgs(process.argv.slice(2))
  const action = args._[0]
  if (!action || action === 'help' || args.help) {
    output({ ok: true, usage: 'node e2e/vibe-actions.cjs detect --json <path> [--min <um>] [--max <um>] [--task-root <dir>]', actions: ['detect'] })
    return
  }
  if (action !== 'detect') throw new Error(`未知动作：${action}（支持：detect）`)
  const opticalJson = String(args.json || '')
  if (!opticalJson) throw new Error('缺少 --json <光镜颗粒 JSON 路径>')
  if (!fs.existsSync(opticalJson)) throw new Error(`JSON 文件不存在：${opticalJson}`)

  const stamp = new Date().toISOString().replace(/[:.]/g, '-')
  const shotRoot = path.join(root, '.tmp', 'vibe-actions', stamp)
  fs.mkdirSync(shotRoot, { recursive: true })
  const taskRoot = args['task-root']
    ? path.resolve(String(args['task-root']))
    : path.join(root, '.tmp', 'vibe-tasks', `vibe-${stamp}`)
  fs.mkdirSync(taskRoot, { recursive: true })

  const app = await electron.launch({
    args: ['.'],
    cwd: root,
    env: { ...process.env, ZEM_TASK_ROOT: taskRoot },
  })
  const errors = []
  try {
    const page = await app.firstWindow()
    page.on('console', (message) => { if (message.type() === 'error') errors.push(message.text()) })
    page.on('pageerror', (error) => errors.push(error.message))

    // ---- 启动与设备自检 ----
    await page.getByRole('heading', { name: '首页' }).waitFor({ timeout: 60000 })
    await page.getByText('后端已连接').waitFor({ timeout: 60000 })
    try {
      await page.getByRole('button', { name: '电镜已连接' }).waitFor({ timeout: 60000 })
    } catch {
      throw new Error('电镜未连接，无法运行检测。请先在软件顶栏确认电镜在线后重试')
    }
    await page.getByRole('button', { name: '能谱已连接' }).waitFor({ timeout: 60000 })
    await page.screenshot({ path: path.join(shotRoot, '01-home.png'), fullPage: true })

    // ---- 新建任务并导入 JSON ----
    await page.getByRole('button', { name: '新建检测任务', exact: true }).click()
    await page.getByRole('heading', { name: '导入与筛选' }).waitFor()
    await app.evaluate(({ dialog }, selectedPath) => {
      dialog.showOpenDialog = async () => ({ canceled: false, filePaths: [selectedPath] })
    }, opticalJson)
    await page.getByRole('button', { name: /请选择光镜颗粒 JSON/ }).click()
    await page.getByText(/\d+ 个有效颗粒/).waitFor({ timeout: 60000 })
    if (args.min || args.max) {
      if (args.min) await page.getByLabel('最小值').fill(String(args.min))
      if (args.max) await page.getByLabel('最大值').fill(String(args.max))
      await page.getByRole('button', { name: '应用筛选' }).click()
      await page.getByText(/^已选 \d+ \/ \d+$/).waitFor({ timeout: 60000 })
    }
    await page.screenshot({ path: path.join(shotRoot, '02-import-filter.png'), fullPage: true })

    // ---- AB 点标定 ----
    await page.getByRole('button', { name: /下一步：坐标标定/ }).click()
    await page.getByRole('heading', { name: '坐标标定' }).waitFor({ timeout: 60000 })
    await page.getByRole('button', { name: 'AB 点法' }).click()
    await page.getByRole('button', { name: '获取导航图' }).click()
    await page.getByAltText('完整 SEM 导航图').waitFor({ timeout: 120000 })
    await page.screenshot({ path: path.join(shotRoot, '03-navigation.png'), fullPage: true })

    // ---- SEM 精定位与确认 ----
    await page.getByRole('button', { name: '开始 SEM 精定位' }).click()
    await page.getByText('SEM 精定位与验证拍摄完成，请逐张对比后确认').waitFor({ timeout: 900000 })
    await page.getByText('光镜 / SEM 对比验收').waitFor()
    await page.screenshot({ path: path.join(shotRoot, '04-calibration-evidence.png'), fullPage: true })
    await page.getByRole('button', { name: /对比一致，确认标定/ }).click()

    // ---- 检测与报告 ----
    await page.getByRole('heading', { name: '检测运行' }).waitFor({ timeout: 120000 })
    await page.getByRole('button', { name: '开始检测' }).click()
    await page.getByText('检测完成，报告已生成').waitFor({ timeout: 3600000 })
    const taskId = await page.locator('.runHeader').locator('span').first().textContent()
      .then((value) => String(value).split(' · ')[0])
    const manifest = await page.evaluate((id) => window.zemCleanliness.request('GET', `/tasks/${id}`), taskId)
    if (!['COMPLETED', 'COMPLETED_WITH_FAILURES'].includes(manifest.status)) {
      throw new Error(`检测任务状态异常：${manifest.status} ${manifest.error || ''}`)
    }
    const reports = manifest.reports || {}
    if (!reports.pdf || !fs.existsSync(reports.pdf)) {
      throw new Error(`PDF 报告未生成：${JSON.stringify(reports)}`)
    }
    await page.screenshot({ path: path.join(shotRoot, '05-analysis-complete.png'), fullPage: true })

    output({
      ok: true,
      action: 'detect',
      taskId,
      status: manifest.status,
      particleCount: (manifest.particles || []).length,
      succeeded: (manifest.particles || []).filter((item) => item.status === 'SUCCEEDED').length,
      failed: (manifest.particles || []).filter((item) => item.status !== 'SUCCEEDED').length,
      reports,
      taskDirectory: path.join(taskRoot, taskId),
      screenshots: shotRoot,
      consoleErrors: errors,
    })
  } catch (error) {
    const screenshot = path.join(shotRoot, 'failure.png')
    try { await (await app.firstWindow()).screenshot({ path: screenshot, fullPage: true }) } catch {}
    output({ ok: false, error: String((error && error.message) || error), screenshot: fs.existsSync(screenshot) ? screenshot : undefined }, true)
    process.exitCode = 1
  } finally {
    await app.close()
  }
}

main().catch((error) => {
  output({ ok: false, error: String((error && error.message) || error) }, true)
  process.exitCode = 1
})
