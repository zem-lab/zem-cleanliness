"""Read-only preflight. Never moves, captures, changes imaging or switches HV."""
from __future__ import annotations

import math
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .cleanliness_detection import load_cleanliness_config, transform_solution_from_dict
from .optical_particle_analysis import load_optical_export, load_calibration


class ZemPreflightReader:
    READS = {
        'StatusCheck':'parse_status', 'GetHvStatus':'parse_hv_status',
        'GetStgStatus':'parse_stage_status', 'GetStagePos':'parse_stage_pos',
        'GetHV':'parse_hv', 'GetSetHV':'parse_set_hv', 'GetMag':'parse_mag',
        'GetScanInfo':'parse_scan_info', 'IsChangingSamp':'parse_is_changing_samp',
        'AllowOpenGun':'parse_allow_open_gun',
    }

    def __init__(self, host, port):
        self.host, self.port = host, port

    def read(self, command):
        if command not in self.READS:
            raise ValueError('Preflight permits only explicitly listed status queries')
        from .microscope_safety import _exchange_modern_command
        from zemrpc.zem.conn.protocol import RespPacket
        text = _exchange_modern_command(self.host, self.port, command=command, timeout=2)
        if not text.lower().startswith(f'${command}:'.lower()):
            raise RuntimeError(f'{command} 回包不匹配')
        packet = RespPacket(text.encode('utf-8'))
        if packet.is_error():
            raise RuntimeError(f'{command} 返回错误：{text}')
        if command in {'IsChangingSamp','AllowOpenGun'}:
            token = packet._parse_1().strip().lower()
            if command == 'IsChangingSamp':
                if token in {'false','0'}: return 0
                if token in {'true','1'}: return 1
                if token == '-1': return -1
            else:
                if token in {'true','1'}: return True
                if token in {'false','0'}: return False
            raise ValueError(f'{command} 未知状态：{token}')
        value = getattr(packet, self.READS[command])()
        if command == 'GetStagePos': return (float(value.x), float(value.y))
        if command == 'GetHV': return (float(value.acc_v), float(value.beam_i))
        # GetHV reports volts; GetSetHV reports kV on the Zem20F protocol.
        if command == 'GetSetHV': return float(value) * 1000.0
        if command == 'GetScanInfo':
            return None if value is None else {'fieldSizeUm':float(value.scan_um), 'rotationDeg':float(value.rot_degree)}
        return getattr(value, 'value', value)


def _model_load(path):
    import onnxruntime as ort
    session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    if not session.get_inputs() or not session.get_outputs():
        raise ValueError('模型缺少输入或输出')


def run_preflight(config, manifest, operation, *, enabled=True, reader=None, model_paths=(),
                  eds_check=None, should_stop=lambda:False, progress=lambda *args:None,
                  sleep=time.sleep, disk_usage=shutil.disk_usage, model_load=_model_load):
    report = {'enabled':bool(enabled), 'operation':operation, 'checkedAt':datetime.now().astimezone().isoformat(),
              'passed':True, 'items':[], 'deviceActionsPerformed':False}
    if not enabled:
        report['items'].append({'id':'disabled','label':'开工检查','status':'SKIP','detail':'已在设置中关闭；原有运行时保护仍生效'})
        return report
    items = report['items']
    def add(code, label, status, detail):
        items.append({'id':code,'label':label,'status':status,'detail':str(detail)})
        progress(len(items), 18, f'开工检查：{label} · {detail}')
    def check(code, label, fn, *, optional=False):
        if should_stop(): raise InterruptedError('开工检查已取消，未执行设备动作')
        try:
            detail = fn()
            if isinstance(detail, tuple): add(code,label,detail[0],detail[1])
            else: add(code,label,'PASS',detail)
            return True
        except InterruptedError: raise
        except Exception as exc:
            add(code,label,'WARN' if optional else 'FAIL',exc)
            return False
    def require(condition, message):
        if not condition: raise ValueError(message)
    try:
        export = None
        settings = None
        reference = None
        selected = []
        def input_check():
            nonlocal export, selected, settings, reference
            if operation == 'sem_analysis':
                from types import SimpleNamespace
                from .sem_direct import build_plan, fingerprint
                plan = manifest['plan']
                checked = build_plan(plan, manifest['navigation']['boundsUm'])
                require([(t['id'], t['x'], t['y']) for t in checked['tiles']] == [(t['id'], t['x'], t['y']) for t in plan['tiles']], '扫描计划坐标不一致')
                require(fingerprint(Path(config.output_dir)/'navigation.png') == manifest['navigation']['sha256'], '导航原图已变化')
                if manifest['model'].get('backend') == 'sam3':
                    from .sem_segmentation import sam3_config
                    sam3_config(manifest['model'])
                else:
                    require(fingerprint(manifest['model']['path']) == manifest['model']['sha256'], '颗粒模型已变化')
                selected = [t for t in plan['tiles'] if t['status'] != 'COMPLETED']
                require(bool(selected), '没有待执行视野')
                reference = SimpleNamespace(imaging={'fieldRotationDeg': 0})
                return f'导航与模型指纹一致，{len(selected)} 个待扫描视野'
            export = load_optical_export(Path(manifest['input']['path']))
            require(export.sha256 == manifest['input'].get('sha256'), '光镜文件已变化，请重新导入')
            sequences = (manifest.get('selection') or {}).get('sequences') or []
            require(bool(sequences) and len(sequences)==len(set(sequences)), '未选择颗粒或序号重复')
            require(all(s in export.by_sequence for s in sequences), '选择中含不存在的颗粒')
            if operation == 'analysis_resume':
                completed = {p['sequence'] for p in manifest.get('particles',[]) if p.get('status') in {'SUCCEEDED','FAILED'}}
                sequences = [s for s in sequences if s not in completed]
            require(bool(sequences), '没有待执行颗粒')
            selected = [export.by_sequence[s] for s in sequences]
            settings = load_cleanliness_config(config.settings_path)
            reference = load_calibration(Path(config.experience_calibration_path))
            return f'输入指纹一致，{len(selected)} 个待执行颗粒'
        check('input','输入文件与任务范围',input_check)

        def storage_check():
            directory = Path(config.output_dir)
            with tempfile.NamedTemporaryFile(prefix='.preflight-',dir=directory) as f:
                f.write(b'preflight'); f.flush()
            free = disk_usage(directory).free
            require(free >= 256*1024**2, f'可用空间仅 {free/1024**2:.0f} MB，少于 256 MB')
            estimated = 512*1024**2 + len(selected)*6*1024**2
            if free < estimated:
                add('storage_estimate','预计保存空间','WARN',f'当前 {free/1024**3:.2f} GB；粗略预计需 {estimated/1024**3:.2f} GB')
            return f'目录可写，可用 {free/1024**3:.2f} GB'
        check('storage','保存目录与磁盘空间',storage_check)

        def models_check():
            if operation == 'sem_analysis' and manifest['model'].get('backend') == 'sam3':
                from .sem_segmentation import check_sam3
                check_sam3(manifest['model'])
                return '远程 SAM3 服务就绪'
            for path in model_paths:
                require(Path(path).is_file(), f'模型不存在：{path}')
                model_load(Path(path))
            return f'{len(model_paths)} 个所需模型可加载' if model_paths else ('SKIP','本阶段不需要 ONNX 模型')
        check('models','所需模型',models_check)

        def target_check():
            if operation == 'sem_analysis':
                require(reference is not None, '扫描计划校验失败')
                return '扫描中心位于导航图范围内；实际采图后核验 FOV 与样品台回读'
            require(export is not None, '输入校验失败，未计算目标位置')
            if operation not in {'analysis','analysis_resume'}:
                return ('SKIP','标定阶段尚无最终确认的变换；仍由原有逻辑在每次移动前校验安全范围')
            calibration = manifest.get('calibration') or {}
            transform = calibration.get('result') or calibration.get('refinedTransform')
            if transform is None:
                index = calibration.get('selectedGroup', (calibration.get('review') or {}).get('selectedGroup',0))
                groups = calibration.get('groups') or []
                group = next((g for g in groups if g.get('groupIndex')==index), groups[0] if groups else {})
                transform = group.get('result')
            if transform is None:
                raise ValueError('缺少已确认的坐标变换')
            solution = transform_solution_from_dict(transform)
            require(np.isfinite(solution.matrix).all() and abs(np.linalg.det(solution.matrix))>1e-12, '坐标变换无效')
            outside = [p.sequence for p in selected if not reference.target_inside_roi(solution.transform_xy(p.ab_x_um,p.ab_y_um))]
            require(not outside, f'预测目标超出配置 ROI：{outside[:12]}')
            return '预测目标位于配置 ROI 内；不代表已验证机械限位或样品高度'
        check('targets','坐标变换与目标范围',target_check)
        if any(x['status']=='FAIL' for x in items):
            add('device_skipped','设备检查','SKIP','本地检查未通过，未向设备查询')
            report['passed']=False
            return report

        reader = reader or ZemPreflightReader(config.microscope_host,config.microscope_port)
        def idle_check():
            status = reader.read('StatusCheck')
            require(str(status).lower()=='idle', f'电镜不是空闲状态：{status}')
            return '电镜响应 Idle'
        if not check('idle','电镜连接与占用状态',idle_check):
            add('remaining','其余设备检查','SKIP','电镜未确认空闲，未继续查询')
            report['passed']=False
            return report
        gun_open = False
        def gun_check():
            nonlocal gun_open
            a = reader.read('GetHvStatus'); sleep(.2); b = reader.read('GetHvStatus')
            gun_open = a==1 and b==1
            if gun_open:
                return '连续两次状态均为开启；未发送开枪命令'
            return ('WARN', f'电子枪未稳定开启（状态 {a}/{b}；0=关闭，1=开启，2=切换中），不拦截开工；未发送开枪命令')
        check('gun','电子枪状态',gun_check)
        def stage_check():
            require(reader.read('GetStgStatus')==1, '样品台正在移动或状态未知')
            a = np.asarray(reader.read('GetStagePos'),float); sleep(.2); b=np.asarray(reader.read('GetStagePos'),float)
            require(a.shape==(2,) and b.shape==(2,) and np.isfinite([a,b]).all(), '样品台坐标反馈无效')
            require(reader.read('GetStgStatus')==1, '样品台状态已变化，未确认静止')
            delta=float(np.linalg.norm(a-b))
            require(delta<=1., f'样品台反馈不稳定，短时变化 {delta:.2f} μm')
            return f'状态静止，X={b[0]:.2f}、Y={b[1]:.2f} μm，变化 {delta:.2f} μm'
        check('stage','样品台静止与坐标反馈',stage_check)

        # Optional queries never turn unknown/unavailable into a green pass.
        try:
            if should_stop(): raise InterruptedError('开工检查已取消')
            changing = reader.read('IsChangingSamp')
            if changing != 0: add('sample','换样状态','FAIL',f'正在换样或状态未知：{changing}')
            else: add('sample','换样状态','PASS','设备报告未处于换样过程')
        except InterruptedError: raise
        except Exception as exc: add('sample','换样状态','WARN',f'无法读取：{exc}；请在原厂软件确认')
        try:
            allowed = reader.read('AllowOpenGun')
            add('interlock','开枪许可反馈','PASS' if allowed is True else 'WARN',
                '设备报告允许开枪（不等于独立真空/腔门诊断）' if allowed is True else '未确认开枪许可；请核查原厂软件联锁状态')
        except Exception as exc: add('interlock','开枪许可反馈','WARN',f'无法读取：{exc}')
        def voltage_check():
            voltage, current = reader.read('GetHV')
            if not gun_open and not (math.isfinite(voltage) and voltage>0 and math.isfinite(current)):
                return ('WARN', f'关枪时电压/束流 {voltage}/{current}，不拦截开工')
            require(math.isfinite(voltage) and voltage>0 and math.isfinite(current), '加速电压/束流反馈无效')
            try:
                set_voltage = float(reader.read('GetSetHV'))
            except Exception as exc:
                set_voltage = None
                add('voltage_set','设定电压','WARN',f'无法读取设定值：{exc}')
            if set_voltage is not None:
                require(math.isfinite(set_voltage) and set_voltage>0 and abs(voltage-set_voltage)<=max(500,set_voltage*.05), '加速电压未达到设定值或仍在变化')
            expected=(manifest.get('baselineImaging') or {}).get('kv')
            if expected is not None:
                require(abs(voltage/1000-float(expected))<=max(.5,float(expected)*.05), '加速电压与已有采集基线不一致')
            return f'电压 {voltage/1000:.2f} kV' + (f'，设定 {set_voltage/1000:.2f} kV' if set_voltage is not None else '') + f'，束流反馈 {current:g}（设备单位）'
        check('voltage','加速电压反馈',voltage_check)
        def imaging_check():
            mag=float(reader.read('GetMag')); scan=reader.read('GetScanInfo')
            require(math.isfinite(mag) and mag>0 and scan and math.isfinite(scan['fieldSizeUm']) and scan['fieldSizeUm']>0 and math.isfinite(scan['rotationDeg']), '倍率或扫描视场反馈无效')
            expected_rotation = reference.imaging.get('fieldRotationDeg')
            if expected_rotation is not None:
                difference = abs((scan['rotationDeg']-float(expected_rotation)+180) % 360-180)
                add('rotation','扫描旋转一致性','PASS' if difference<=.1 else 'FAIL',
                    f'当前 {scan["rotationDeg"]:.2f}°，标定要求 {float(expected_rotation):.2f}°')
            else:
                add('rotation','扫描旋转一致性','WARN','标定未提供扫描旋转角度，未做一致性核验')
            return f'当前 {mag:g}×，视场 {scan["fieldSizeUm"]:.2f} μm；后续按任务参数配置并校验实际采图'
        check('imaging','倍率与扫描视场',imaging_check,optional=True)
        if operation in {'analysis','analysis_resume','sem_analysis'} and config.eds_enabled:
            def spectrum_check():
                require(eds_check is not None, '未提供能谱就绪检查')
                readiness=eds_check()
                require(readiness.get('status')=='READY', f'能谱未就绪：{readiness}')
                return '能谱服务会话就绪；未开始采谱'
            check('eds','能谱服务',spectrum_check)
        else: add('eds','能谱服务','SKIP','当前阶段不采谱，检测启动时按需检查')
        def final_state_check():
            require(str(reader.read('StatusCheck')).lower()=='idle', '检查期间电镜已变为忙碌')
            reader.read('GetHvStatus')
            require(reader.read('GetStgStatus')==1, '检查期间样品台开始移动或状态未知')
            return '电镜仍空闲、样品台仍静止；电子枪状态不拦截开工'
        check('final_state','启动前再次确认',final_state_check)
        add('limits','检查能力边界','WARN','不试移台、不试拍；未独立测量真空、腔门、样品高度、各轴故障码、探测器信号和图像清晰度。实际移台与采图仍执行原有保护。')
    except InterruptedError as exc:
        report['cancelled']=True
        add('cancelled','开工检查','FAIL',exc)
    report['passed']=not any(x['status']=='FAIL' for x in items)
    return report
