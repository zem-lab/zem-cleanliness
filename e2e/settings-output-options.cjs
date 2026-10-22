// Mocked backend UI contract: no hardware traffic.
const assert = require('node:assert/strict');
const { chromium } = require('playwright');
(async () => {
  const browser = await chromium.launch({ headless:true, executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe' });
  try {
    const page = await browser.newPage();
    const pageErrors=[];const consoleErrors=[];
    page.on('pageerror',error=>pageErrors.push(String(error)));
    page.on('console',message=>{if(message.type()==='error')consoleErrors.push(message.text())});
    let settings = {microscopeHost:'127.0.0.1',microscopePort:61557,edsHost:'127.0.0.1',edsPort:6200,
      edsEnabled:true,restoreStartPosition:true,gunOffAfterTask:true,showReportCoordinates:true,preflightEnabled:true,captureLevel:7,candidateScoreMargin:0.07,templateMatchMinScore:0.65,contourOverlapMin:0.5,maxCaptureAttempts:2,taskRoot:'test'};
    let saved = 0;
    await page.route('http://127.0.0.1:8765/**', async route => {
      const req=route.request(), pathname=new URL(req.url()).pathname;
      const headers={'Access-Control-Allow-Origin':'*','Access-Control-Allow-Headers':'*','Access-Control-Allow-Methods':'*'};
      if(req.method()==='OPTIONS')return route.fulfill({status:204,headers});
      let body={};
      if(pathname==='/settings') {
        if(req.method()==='POST'){settings=req.postDataJSON();saved++;}
        body=settings;
      } else if(pathname==='/tasks')body={tasks:[]};
      else if(pathname==='/devices')body={devices:[]};
      else if(pathname==='/models')body={models:[]};
      else throw Error(`Unexpected route ${pathname}`);
      await route.fulfill({json:body,headers});
    });
    await page.goto('http://127.0.0.1:5174');
    await page.getByRole('button',{name:'系统设置',exact:true}).click();
    const gun=page.getByRole('switch',{name:'检测结束后自动关枪',exact:true});
    const coordinates=page.getByRole('switch',{name:'显示光镜/电镜坐标',exact:true});
    const preflight=page.getByRole('switch',{name:'开工前状态检查',exact:true});
    const captureLevel=page.getByLabel('拍照档位',{exact:true});
    const candidateScoreMargin=page.getByLabel('候选颗粒分差阈值',{exact:true});
    const templateMatchMinScore=page.getByLabel('模板相关分数阈值',{exact:true});
    const contourOverlapMin=page.getByLabel('轮廓重叠',{exact:true});
    const maxCaptureAttempts=page.getByLabel('拍照张数',{exact:true});
    await gun.waitFor();
    assert.equal(await gun.getAttribute('aria-checked'),'true');
    assert.equal(await coordinates.getAttribute('aria-checked'),'true');
    assert.equal(await preflight.getAttribute('aria-checked'),'true');
    assert.equal(await captureLevel.inputValue(),'7');
    assert.equal(await candidateScoreMargin.inputValue(),'0.07');
    assert.equal(await templateMatchMinScore.inputValue(),'0.65');
    assert.equal(await contourOverlapMin.inputValue(),'0.5');
    assert.equal(await maxCaptureAttempts.inputValue(),'2');
    await captureLevel.selectOption('4');
    await candidateScoreMargin.fill('0.09');
    await templateMatchMinScore.fill('0.55');
    await contourOverlapMin.fill('0.4');
    await maxCaptureAttempts.selectOption('3');
    await gun.click();await coordinates.click();await preflight.click();
    await page.getByRole('button',{name:'保存设置',exact:true}).click();
    await page.getByText('设置已保存',{exact:true}).waitFor();
    assert(saved>0 && settings.gunOffAfterTask===false && settings.showReportCoordinates===false && settings.preflightEnabled===false && settings.captureLevel===4 && settings.candidateScoreMargin===0.09 && settings.templateMatchMinScore===0.55 && settings.contourOverlapMin===0.4 && settings.maxCaptureAttempts===3);
    await page.getByRole('button',{name:'首页',exact:true}).click();
    const settingsReload=page.waitForResponse(response=>new URL(response.url()).pathname==='/settings' && response.request().method()==='GET');
    await page.getByRole('button',{name:'系统设置',exact:true}).click();
    await settingsReload;
    assert.equal(await gun.getAttribute('aria-checked'),'false');
    assert.equal(await coordinates.getAttribute('aria-checked'),'false');
    assert.equal(await preflight.getAttribute('aria-checked'),'false');
    assert.equal(await page.getByLabel('拍照档位',{exact:true}).inputValue(),'4');
    assert.equal(await page.getByLabel('候选颗粒分差阈值',{exact:true}).inputValue(),'0.09');
    assert.equal(await page.getByLabel('模板相关分数阈值',{exact:true}).inputValue(),'0.55');
    assert.equal(await page.getByLabel('轮廓重叠',{exact:true}).inputValue(),'0.4');
    assert.equal(await page.getByLabel('拍照张数',{exact:true}).inputValue(),'3');
    assert.match(await page.locator('body').innerText(),/默认展示光镜原图、原始光镜掩码和 SEM 图/);
    assert.deepEqual(pageErrors,[]);assert.deepEqual(consoleErrors,[]);
    console.log('settings defaults, shared capture level, candidate score margin, persistence, and always-on optical mask explanation=passed');
  } finally {await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1});
