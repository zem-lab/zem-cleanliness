"""All 115 upper-computer RPC contracts, including acknowledgement-only actions."""
from __future__ import annotations

import base64
from copy import deepcopy
import json
import math
import ntpath

from backend.sem_rpc_catalog import READS, WRITES, MAPPED, number, choice, argument_schema


def positional(params=(), *, optional=False):
    schema = {"type": "array", "items": False, "minItems": len(params), "maxItems": len(params)}
    if params:
        schema["prefixItems"] = deepcopy(list(params))
    return {"oneOf": [{"type": "array", "maxItems": 0}, schema]} if optional else schema


TEXT = {"type": "string", "minLength": 1, "maxLength": 1024, "pattern": r"^[^,\x00-\x1f%]+$"}
PATH = {**TEXT, "pattern": r"^[^\x00-\x1f]+$", "description": "上位机 Windows 路径；需在 rpcPathRoots 内，不是本机路径"}
ACK = {
    "SetBrightness": ([number("%", 0, 100)], "imaging"),
    "SetContrast": ([number("%", 0, 100)], "imaging"),
    "SetBeamShift": ([number("µm"), number("µm")], "imaging"),
    "OpenVAC": ([], "vacuum"), "ReleaseVAC": ([], "vacuum"),
    **{f"SetStage{axis}Pos": ([number("上位机原始轴参数；X/Y 内部乘 1000，Z 使用 CustomPos，R/T 使用 RealUnitPos")], "motion")
       for axis in "XYZRT"},
    "SetStgTarget": ([number("上位机参考 X"), number("上位机参考 Y"), number("相对视场 X 偏移"),
                      number("相对视场 Y 偏移"), number("参考倍率；0=使用当前值", 0, 1000000, True)], "motion"),
    "SetAutoSnapTask": ([PATH], "autotask"),
    "SaveImage": ([PATH, {**TEXT, "maxLength": 128, "pattern": r"^[^\\/:*,?\"<>|\x00-\x1f%]+$"},
                   {"type": "string", "enum": ["png", "jpg", "jpeg", "tif", "tiff"]},
                   choice([0, 1], "上位机保存标志"), {"type": "string", "enum": ["BSE", "SE", "ALL"]}], "file"),
    "StartScanPoints": ([], "scan"), "StopScanPoints": ([], "stop"), "SnapCancel": ([], "stop"),
}
SPECS = {cmd: {"kind": "read", "schema": positional(), "completion": "reply", "category": "read"} for cmd in READS}
SPECS.update({cmd: {"kind": "write", "schema": argument_schema(cmd), "completion": "readback", "category": spec['category']}
              for cmd, spec in WRITES.items()})
SPECS['SetAvLenth'].update(kind='request', completion='acknowledgement_only',
                         reason='设置 SnapAvPointGear；GetAvLenth 读的是 VideoAvPointGear，不能作为完成回读')
SPECS.update({cmd: {"kind": "request", "schema": positional(params), "completion": "acknowledgement_only", "category": category}
              for cmd, (params, category) in ACK.items()})
# The upstream accepts flat integer x,y pairs. An even-length schema rejects
# the server's silent truncation of a dangling final coordinate.
SPECS['StartScanPoints']['schema'] = {"type": "array", "items": number("扫描像素坐标", 0, 65535, True),
    "minItems": 2, "maxItems": 256, "allOf": [{"not": {"minItems": n, "maxItems": n}} for n in range(3, 256, 2)]}
SPECS.update({
    "SetIe": {"kind": "unavailable", "schema": positional([number("上位机原始发射电流参数")]),
              "completion": "unsupported", "category": "unsupported", "reason": "上位机实现返回 InvalidCmd"},
    "ZemRpc": {"kind": "read", "schema": positional([{"type": ["string", "integer"], "description": "JSON-RPC 请求 id",
                                                          "maxLength": 128}]),
               "completion": "reply", "category": "read"},
    "SetStagePos": {"kind": "mapped", "schema": positional([number("µm"), number("µm")]),
                    "completion": "position_readback", "category": "motion"},
    "GetNavImgBuf": {"kind": "mapped", "schema": positional(), "completion": "reply", "category": "read"},
    "Snap": {"kind": "mapped", "schema": positional([choice([0, 1]), choice(list(range(1, 8))),
              number("平均帧数", 1, 256, True), choice([0, 1, 2, 3])]), "completion": "image", "category": "capture"},
    "GetImg": {"kind": "mapped", "schema": positional([number("上位机通道参数", 0, 65535, True)], optional=True),
               "completion": "image", "category": "image_read"},
    "GetNavImg": {"kind": "mapped", "schema": positional(), "completion": "image", "category": "image_read"},
})
WRITE_COMMANDS = frozenset(cmd for cmd, spec in SPECS.items() if spec['category'] not in {'read', 'image_read', 'unsupported'})
STOPS = frozenset({'StopScanPoints', 'SnapCancel'})
IMAGES = frozenset({'Snap', 'GetImg', 'GetNavImg'})
QUERY_COMMANDS = frozenset(cmd for cmd, spec in SPECS.items() if spec['category'] in {'read', 'image_read'})


def full_catalog(allowed=()):
    return [{"command": cmd, "kind": spec['kind'], "tool": 'sem_rpc_read' if cmd in QUERY_COMMANDS else 'sem_rpc_call', "argument_schema": deepcopy(spec['schema']),
             "completion": spec['completion'], "category": spec['category'],
             "enabled": spec['category'] in {'read', 'image_read'} or cmd in allowed,
             **({"reason": spec['reason']} if 'reason' in spec else {}),
             **({"alternative_tool": MAPPED[cmd]} if cmd in MAPPED else {}),
             **({"description": READS[cmd][0], "unit": READS[cmd][1]} if cmd in READS else {}),
             **({"note": "仅确认命令受理；不确认生效/停止。将保留待核验记录，不会自动重发"}
                if spec['completion'] == 'acknowledgement_only' else {})}
            for cmd, spec in SPECS.items()]


def validate(command, values):
    from jsonschema import Draft202012Validator
    from backend.sem_control import SemError
    if command not in SPECS:
        raise SemError('UNSUPPORTED', '未注册的 RPC 命令')
    errors = list(Draft202012Validator(SPECS[command]['schema']).iter_errors(values))
    if errors or any(isinstance(v, (int, float)) and (isinstance(v, bool) or not math.isfinite(v)) for v in values):
        raise SemError('INVALID_ARGUMENT', errors[0].message if errors else '参数必须为有限数字')
    if command == 'SaveImage' and (values[1] in {'.', '..'} or values[1].endswith((' ', '.'))):
        raise SemError('INVALID_ARGUMENT', '文件名不能是相对目录或以空格、点结尾')


def wire_values(command, values):
    validate(command, values)
    result = list(values)
    if command in {'SaveImage', 'SetAutoSnapTask'}:
        result[0] = base64.b64encode(values[0].encode('utf-8')).decode('ascii')
    if command == 'ZemRpc':
        result = [base64.b64encode(json.dumps({'jsonrpc': '2.0', 'id': values[0]}, ensure_ascii=False).encode('utf-8')).decode('ascii')]
    return result


def validate_paths(command, values, roots):
    from backend.sem_control import SemError
    if command not in {'SaveImage', 'SetAutoSnapTask'}:
        return
    path = values[0]
    drive, _ = ntpath.splitdrive(path)
    components = path.replace('/', '\\')[3:].split('\\')
    if (len(drive) != 2 or not drive[0].isascii() or not drive[0].isalpha() or drive[1] != ':'
            or path.startswith(('\\\\', '//')) or not ntpath.isabs(path)
            or any(part in {'.', '..'} or part.endswith((' ', '.')) for part in components if part)
            or any(char in path[2:] for char in ':*?"<>|')):
        raise SemError('INVALID_ARGUMENT', '必须是上位机本地绝对路径，不能包含路径穿越或设备路径')
    normalized = ntpath.normcase(ntpath.normpath(path))
    for root in roots if isinstance(roots, (tuple, list)) else ():
        if not isinstance(root, str) or not ntpath.isabs(root):
            continue
        try:
            if ntpath.commonpath([ntpath.normcase(ntpath.normpath(root)), normalized]) == ntpath.normcase(ntpath.normpath(root)):
                return
        except ValueError:
            pass
    raise SemError('OPERATION_DISABLED', '上位机路径不在本机配置的 rpcPathRoots 内')
