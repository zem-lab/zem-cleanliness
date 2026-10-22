"""Run the frozen MCP bridge and backend with isolated simulated devices."""
import asyncio
from datetime import timedelta
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / '.tmp/mcp-frozen-protocol' / time.strftime('%Y%m%d-%H%M%S')
OUT.mkdir(parents=True, exist_ok=True)
EXE = ROOT / 'backend-dist/zem-cleanliness-backend.exe'
env = {**os.environ, 'ZEM_TASK_ROOT':str(OUT/'tasks'), 'ZEM_LOG_DIR':str(OUT/'logs'),
       'ZEM_RESOURCE_ROOT':str(ROOT/'resources'),
       'ZEM_MCP_TOKEN':'isolated-frozen-test-only', 'PYTHONUTF8':'1'}
process = subprocess.Popen([str(EXE),'--mcp-simulated','--port','0','--mcp-state-dir',str(OUT/'mcp')],
                           env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')
try:
    lines = queue.Queue()
    threading.Thread(target=lambda: lines.put(process.stdout.readline()),daemon=True).start()
    ready = json.loads(lines.get(timeout=30))
    env['ZEM_MCP_BACKEND_URL'] = 'http://127.0.0.1:' + str(ready['port'])
    env['ZEM_MCP_CALLER_ID'] = 'frozen-protocol-test'
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    bridge_log = (OUT/'bridge-stderr.log').open('w', encoding='utf-8')
    async def check():
        async with stdio_client(StdioServerParameters(command=str(EXE),args=['--mcp-stdio'],env=env), errlog=bridge_log) as (read,write):
            async with ClientSession(read,write,read_timeout_seconds=timedelta(seconds=30)) as session:
                await session.initialize()
                tools = (await session.list_tools()).tools
                assert len(tools) == 16
                async def call(name,args):
                    result = await session.call_tool(name,args)
                    assert not result.isError, result
                    return result.structuredContent
                caps = await call('sem_get_capabilities',{})
                assert caps['mode'] == 'simulated'
                snapshot = await call('sem_get_snapshot',{})
                assert snapshot['state_text']['gun_state'] == '开启'
                catalog = await call('sem_list_rpc',{})
                assert len(catalog['commands']) == 115
                photo_average = next(e for e in catalog['commands'] if e['command']=='SetAvLenth')
                assert photo_average['completion']=='acknowledgement_only'
                assert 'SnapAvPointGear' in photo_average['reason']
                job = await call('sem_rpc_execute',{'command':'SetMag','values':[5000],'request_id':'freeze-write-1'})
                result = await call('sem_wait_job',{'job_id':job['job_id'],'timeout_seconds':10})
                assert result['status'] == 'completed'
                assert (await call('sem_rpc_read',{'command':'GetMag'}))['values'] == [5000]
                capture = await call('sem_capture',{'request_id':'freeze-capture-1'})
                image_job = await call('sem_wait_job',{'job_id':capture['job_id'],'timeout_seconds':10})
                assert image_job['status'] == 'completed', image_job
                picture = await session.call_tool('sem_get_image',{'image_id':image_job['result']['image']['image_id']})
                assert not picture.isError and any(c.type == 'image' for c in picture.content)
                discovery = await call('sem_rpc_read', {'command':'ZemRpc','values':['frozen-discovery']})
                assert discovery['values']['id'] == 'frozen-discovery'
                assert (await call('sem_rpc_call', {'command':'GetNavImgBuf','values':[]}))['values'] == [8,512,512]
                for command, values in [('GetImg',[0]),('GetNavImg',[]),('Snap',[0,1,1,3])]:
                    raw_job = await call('sem_rpc_call', {'command':command,'values':values,'request_id':'raw-'+command})
                    raw_result = await call('sem_wait_job', {'job_id':raw_job['job_id'],'timeout_seconds':10})
                    assert raw_result['status'] == 'completed', raw_result
                    assert raw_result['result']['image']['width'] == (128 if command == 'Snap' else 512)
                unsupported = await session.call_tool('sem_rpc_call', {'command':'SetIe','values':[1]})
                assert unsupported.isError and unsupported.structuredContent['error']['code'] == 'UNSUPPORTED'
                ack_args = {'command':'SetBrightness','values':[40],'request_id':'frozen-ack'}
                ack_job = await call('sem_rpc_call', ack_args)
                ack_result = await call('sem_wait_job', {'job_id':ack_job['job_id'],'timeout_seconds':10})
                assert ack_result['status'] == 'accepted' and not ack_result['result']['rpc']['completion_confirmed']
                assert 'completed_at' not in ack_result['result']
                assert (await call('sem_rpc_call', ack_args))['job_id'] == ack_job['job_id']
                blocked = await session.call_tool('sem_rpc_call', {'command':'SetContrast','values':[60],'request_id':'blocked-new'})
                assert blocked.isError and blocked.structuredContent['error']['code'] == 'DEVICE_BUSY'
                stopped = await call('sem_rpc_call', {'command':'StopScanPoints','values':[],'request_id':'frozen-stop'})
                stop_result = await call('sem_wait_job', {'job_id':stopped['job_id'],'timeout_seconds':10})
                assert stop_result['status'] == 'accepted' and not stop_result['result']['rpc']['completion_confirmed']
                summary = {'passed':True,'tools':len(tools),'catalog':len(catalog['commands']),
                           'mode':caps['mode'],'typed_write_verified':True,'image_returned':True,
                           'raw_images':3,'acknowledgement_semantics':True,'durable_duplicate':True,
                           'unverified_write_blocked':True,'unsupported_stub':True,'explicit_stop':True}
                (OUT/'results.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
                print(json.dumps({**summary, 'evidence':str(OUT)}))
    asyncio.run(check())
    bridge_log.close()
    bridge_errors = (OUT/'bridge-stderr.log').read_text(encoding='utf-8')
    assert 'Traceback' not in bridge_errors and 'ValueError' not in bridge_errors, bridge_errors
    print('frozen_bridge_shutdown=clean')
finally:
    subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],capture_output=True)
    stdout,stderr = process.communicate(timeout=15)
    (OUT/'backend-stderr.log').write_text(stderr,encoding='utf-8')
