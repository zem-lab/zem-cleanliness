// UI-only completion semantics. All backend requests are mocked; no hardware.
const assert = require('node:assert/strict');
const { chromium } = require('playwright');

async function main() {
  const browser = await chromium.launch({ headless: true, executablePath: 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe' });
  try {
    for (const mode of ['partial', 'fatal', 'partial-with-error', 'closed', 'unconfirmed', 'preflight-blocked']) {
      const context = await browser.newContext();
      const page = await context.newPage();
      let finished = false;
      const result = () => ({ taskId: 'completion-test', status: !finished ? 'CALIBRATION_ACCEPTED' : mode === 'fatal' ? 'FAILED' : 'COMPLETED_WITH_FAILURES',
        input: { workpieceName: 'Completion test', particleCount: 3 }, selection: { count: 3 },
        calibration: { strategy: 'ab_points', review: { accepted: true } },
        reports: finished && ['partial','closed'].includes(mode) ? { pdf: 'report.pdf' } : null,
        error: finished && !['partial','closed'].includes(mode) ? 'actual device or report failure' : null,
        cleanup: finished && ['closed','unconfirmed'].includes(mode) ? { gunOffRequested:true,gunOffVerified:mode==='closed',gunOffError:mode==='unconfirmed'?'HV timeout':null } : {},
        particles: finished ? [{ sequence: 1, status: 'SUCCEEDED' },
          { sequence: 2, status: 'FAILED', error: 'mask ambiguous' },
          { sequence: 3, status: 'FAILED', error: 'EDS timeout', edsAttempts: [{ status: 'FAILED' }] }] : [] });
      await page.route('http://127.0.0.1:8765/**', async route => {
        const pathname = new URL(route.request().url()).pathname;
        const headers = { 'Access-Control-Allow-Origin': '*', 'Access-Control-Allow-Headers': '*', 'Access-Control-Allow-Methods': '*' };
        if (route.request().method() === 'OPTIONS') return route.fulfill({ status: 204, headers });
        let body = {};
        if (pathname === '/tasks') body = { tasks: [{ id: 'completion-test', status: result().status, input: result().input }] };
        else if (pathname === '/tasks/completion-test') body = result();
        else if (pathname === '/tasks/completion-test/analysis') body = { id: 'completion-job', status: 'queued' };
        else if (pathname === '/jobs/completion-job') {
          if(mode==='preflight-blocked')body={id:'completion-job',status:'failed',error:'开工检查未通过：电子枪未开启',preflight:{enabled:true,passed:false,items:[{id:'gun',label:'电子枪状态',status:'FAIL',detail:'请在电镜端确认开枪'}]}};
          else {finished = true; body = { id: 'completion-job', status: 'completed', result: result() };}
        }
        else if (pathname === '/devices') body = { devices: [] };
        else if (pathname === '/models') body = { models: [] };
        else if (!['/health', '/settings'].includes(pathname)) throw new Error(`Unexpected request ${pathname}`);
        await route.fulfill({ json: body, headers });
      });
      await page.goto('http://127.0.0.1:5174');
      await page.locator('.tableRow').filter({ hasText: 'completion-test' }).click();
      await page.getByRole('button', { name: '开始检测', exact: true }).click();
      if(mode==='preflight-blocked') {
        await page.getByText('电子枪状态 · 阻止启动',{exact:true}).waitFor();
        assert.match(await page.locator('.toast.error').innerText(),/开工检查未通过/);
        assert.equal(await page.locator('.semImage img').count(),0);
      } else if (['partial','closed'].includes(mode)) {
        await page.locator('.completionSummary').waitFor();
        assert.match(await page.locator('.completionSummary').innerText(), /成功 1 个，定位\/采图未通过 1 个，能谱失败 1 个。报告已生成/);
        assert.equal(await page.locator('.toast.error, .persistentError').count(), 0);
        if(mode==='closed')assert.match(await page.locator('.completionSummary').innerText(), /电子枪已关闭（设备反馈确认）/);
        await page.getByText('查看 2 个失败颗粒及原因', { exact: true }).click();
        assert.match(await page.locator('details').innerText(), /mask ambiguous/);
        assert.match(await page.locator('details').innerText(), /EDS timeout/);
      } else {
        await page.locator('.toast.error').waitFor();
        assert.match(await page.locator('.toast.error').innerText(), /actual device or report failure/);
        assert.equal(await page.locator('.completionSummary').count(), 0);
        if(mode==='unconfirmed')assert.match(await page.locator('.verificationGuide').innerText(), /自动关枪未确认/);
      }
      console.log(`${mode}=passed`);
      await context.close();
    }
  } finally { await browser.close(); }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
