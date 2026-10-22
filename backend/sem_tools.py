"""MCP tool contract. No SDK import or hardware initialization on discovery."""
from __future__ import annotations

from copy import deepcopy
from backend.sem_rpc_catalog import READS, WRITES, argument_schema
from backend.sem_rpc_full import SPECS, QUERY_COMMANDS


def _schema(properties=None, required=()):
    return {"type": "object", "properties": properties or {},
            "required": list(required), "additionalProperties": False}


REQUEST = {"type": "string", "minLength": 1, "maxLength": 128,
           "pattern": r"^[A-Za-z0-9_.:-]+$",
           "description": "调用方生成的唯一请求标识；重试必须使用原标识，避免重复操作"}
ID = {"type": "string", "pattern": r"^[0-9a-f]{32}$"}
CAPTURE = {"size": {"type": "integer", "enum": [512, 1024, 2048], "default": 512},
           "pixel_avg": {"type": "integer", "minimum": 1, "maximum": 7, "default": 7}}
XY = {"x_um": {"type": "number", "description": "电镜台坐标 X，单位 µm"},
      "y_um": {"type": "number", "description": "电镜台坐标 Y，单位 µm"}}


def _tool(name, description, schema, *, readonly=False):
    return {"name": name, "description": description, "inputSchema": schema,
            "annotations": {"readOnlyHint": readonly, "destructiveHint": not readonly,
                            "idempotentHint": readonly, "openWorldHint": False}}


TOOLS = [
    _tool("sem_get_capabilities", "查询支持的电镜操作及限制；不连接电镜。先调用此工具。", _schema(), readonly=True),
    _tool("sem_get_status", "只读查询忙闲、枪及样品台状态；不移动、不改倍率、不开枪。", _schema(), readonly=True),
    _tool("sem_get_position", "只读查询电镜 XY 台坐标，单位 µm。", _schema(), readonly=True),
    _tool("sem_get_navigation", "用户要导航图/导航概览时的默认入口：读取设备已有导航图，返回作业标识；"
          "不自动创建导航图，不采 SEM 当前视野。完成后 sem_wait_job（timeout≤20）再 sem_get_image。"
          "禁止用 GetImg 或 sem_capture 冒充导航图。",
          _schema({"request_id": REQUEST}, ["request_id"]), readonly=True),
    _tool("sem_capture", "仅在用户明确要求「新拍一张」时使用（pyzemrpc JSON-RPC snap）。"
          "返回作业标识；不会自动开枪或修改倍率。完成后用 sem_get_image 获取图片。"
          "看图/拿当前画面改用 GetImg；要导航图改用 sem_get_navigation。",
          _schema({"request_id": REQUEST, **CAPTURE}, ["request_id"])),
    _tool("sem_move_xy", "移动到明确的 XY 绝对台坐标（µm）；必须确认用户操作意图。回读确认到位才算完成。",
          _schema({"request_id": REQUEST, **XY}, ["request_id", "x_um", "y_um"])),
    _tool("sem_move_and_capture", "独占设备执行 XY 移动、到位确认、采新图（同 sem_capture，JSON-RPC snap），防止操作被插入。返回作业标识。",
          _schema({"request_id": REQUEST, **XY, **CAPTURE}, ["request_id", "x_um", "y_um"])),
    _tool("sem_get_job", "查询本调用方的作业进度、错误和图像标识；结果未知时禁止换请求标识重试。",
          _schema({"job_id": ID}, ["job_id"]), readonly=True),
    _tool("sem_cancel_job", "请求取消本调用方作业。正在执行的硬件指令需等待结束；不代表设备已物理停止。",
          _schema({"job_id": ID}, ["job_id"])),
    _tool("sem_get_image", "按图像标识读取已保存图像，返回图片和带来源的元数据。",
          _schema({"image_id": ID, "variant": {"type": "string", "enum": ["preview", "original"], "default": "preview"}}, ["image_id"]), readonly=True),
]
TOOLS.extend([
    _tool("sem_get_snapshot", "一次读取状态和 XY，避免模型并行查询冲突；含状态中文解释及计时。", _schema(), readonly=True),
    _tool("sem_list_rpc", "查询完整 RPC 命令清单、单位、参数及未开放原因；不连接设备。",
          _schema({"kind": {"type": "string", "enum": ["read", "write", "request", "mapped", "unavailable"]}}), readonly=True),
    _tool("sem_rpc_read", "只读查询原始参数、ZemRpc 服务地址和已有图像。参数见 sem_list_rpc。"
          "看图/当前画面：command=GetImg，values=[0]，带 request_id → wait → sem_get_image。"
          "导航图优先 sem_get_navigation；若走本工具则 command=GetNavImg（不要用 GetImg 冒充）。"
          "仅当用户明确要求新拍时才用 sem_capture。不会下发设置。",
          {**_schema({"command": {"type": "string", "enum": sorted(QUERY_COMMANDS)},
                      "values": {"type": "array"}, "request_id": REQUEST}, ['command']),
           "allOf": [{"if": {"properties": {"command": {"const": cmd}}},
                      "then": {"properties": {"values": SPECS[cmd]['schema']},
                               **({'required': ['values']} if cmd == 'ZemRpc' else {}),
                               **({'required': ['request_id']} if cmd in {'GetImg', 'GetNavImg'} else {})}}
                     for cmd in sorted(QUERY_COMMANDS)]}, readonly=True),
    _tool("sem_rpc_execute", "执行类型化电镜设置/对焦/开关枪/换样等 RPC；必须有用户明确意图且本机已开放该命令。"
          "返回作业；以实际回读确认完成，SetAvLenth 因读写不同字段仅确认受理，不隐式重试。先用 sem_list_rpc 查参数和权限。",
          {**_schema({"request_id": REQUEST, "command": {"type": "string", "enum": sorted(WRITES)},
                      "values": {"type": "array", "items": {"type": "number"}, "maxItems": 8}},
                    ["request_id", "command", "values"]),
           "allOf": [{"if": {"properties": {"command": {"const": command}}},
                      "then": {"properties": {"values": argument_schema(command)}}} for command in sorted(WRITES)]}),
    _tool("sem_wait_job", "等待本会话已有作业至结束或等待上限，减少模型反复轮询；等待超时不取消、不重发硬件命令。"
          "timeout_seconds 最大值 20（默认 15）；未结束继续等原 job_id，禁止传更大值或换 request_id。",
          _schema({"job_id": ID, "timeout_seconds": {"type": "number", "minimum": 0, "maximum": 20, "default": 15}},
                  ["job_id"]), readonly=True),
])
TOOLS.append(_tool("sem_rpc_call", "按原始 RPC 名称调用全部 115 项接口。先查 sem_list_rpc 获取逐项参数。"
    "查询直接返回；动作需 request_id、本机命令权限及参数范围。acknowledgement_only 仅为受理，"
    "不代表动作生效或停止；需现场核验，不自动重发。SetIe 明确返回上位机不支持。ZemRpc 只查询服务地址。"
    "当前图 GetImg；导航图 GetNavImg 或 sem_get_navigation；新拍用 sem_capture。"
    "不要用本工具的 UDP Snap 当默认识图入口，也不要用 GetImg 冒充导航图。",
    {**_schema({"command": {"type": "string", "enum": sorted(SPECS)},
                "values": {"type": "array", "maxItems": 256}, "request_id": REQUEST}, ['command', 'values']),
     "allOf": [{"if": {"properties": {"command": {"const": cmd}}},
                "then": {"properties": {"values": spec['schema']},
                         **({"required": ['request_id']} if spec['category'] not in {'read', 'unsupported'} else {})}}
               for cmd, spec in SPECS.items()]}))
TOOL_MAP = {tool["name"]: tool for tool in TOOLS}


def tools_for_model():
    return [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                               "parameters": deepcopy(t["inputSchema"])}} for t in TOOLS]
