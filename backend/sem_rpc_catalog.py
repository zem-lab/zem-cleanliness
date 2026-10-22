"""Versioned RPC contract for tribf_dev_hisi, not an arbitrary command tunnel.

Wire names/units come from ExternModel.cs (cc785a3). SDK spellings sometimes
differ (SetEDChanel/SetEDChannel, SetAST/SetAst); use the server's names.
"""
from __future__ import annotations

from copy import deepcopy


def number(unit, minimum=None, maximum=None, integer=False):
    value = {"type": "integer" if integer else "number", "description": unit}
    if minimum is not None: value["minimum"] = minimum
    if maximum is not None: value["maximum"] = maximum
    return value


def choice(values, description="协议枚举值"):
    return {"type": "integer", "enum": values, "description": description}


# No numeric value without a documented unit is described as a physical unit.
READS = {
    "ServerName": ("RPC 服务名称", "文本"),
    "StatusCheck": ("设备忙闲", "状态文本"), "GetStagePos": ("XY 台坐标", "µm"),
    "GetHV": ("实测加速电压与发射电流", "V；电流单位由设备定义"),
    "GetIe": ("发射电流", "设备原始单位"), "GetBeamSz": ("束斑档位", "0..3"),
    "GetMag": ("倍率", "倍"), "GetVAC": ("枪真空", "设备原始真空单位"),
    "GetAvMode": ("平均模式", "0=逐点；1=逐帧"), "GetAvLenth": ("视频逐点平均档位；不是 SetAvLenth 修改的拍照档位", "1..7"),
    "GetAvFrames": ("平均帧数", "帧"), "GetSEGain": ("SE 增益", "%"),
    "GetSECV": ("SE 收集电压", "设备原始电压单位"), "GetOB": ("物镜值", "原始 DAC 值，0..65535"),
    "GetRotation": ("扫描旋转", "度"), "GetEDChannel": ("探测器通道", "0=BSE；1=SE"),
    "GetWD": ("工作距离", "mm"), "GetAst": ("像散双轴", "设备原始 DAC 值"),
    "GetImgSz": ("配置图片档位", "0=512；1=2048；2=1024；3=128"),
    "GetCurImgSz": ("当前图像宽度", "像素"), "IsVideoAlive": ("视频状态", "0=停止；1=运行"),
    "GetNavPara": ("导航坐标映射", "上位机导航坐标约定，未经轴向标定不能驱动移动"),
    "GetStgStatus": ("台状态", "0=运行；1=静止；其他=未知"),
    "GetScanInfo": ("视场宽度与旋转", "µm；度"), "GetModel": ("设备型号", "文本"),
    "GetWindowPos": ("扫描小窗位置", "像素"),
    "CheckRequestOpenVac": ("开始换样进度", "1=进行中；0=完成；-1=失败"),
    "CheckRequestCloseVac": ("结束换样进度", "1=进行中；0=完成；-1=失败"),
    "CheckSnapNav": ("导航拍摄进度", "1=进行中；0=完成；-1=失败"),
    "GetStigX": ("像散 X", "%"), "GetStigY": ("像散 Y", "%"),
    "GetCompressA": ("压缩 A", "%"), "GetCompressB": ("压缩 B", "%"),
    "GetGunVac": ("枪真空", "设备原始真空单位"), "GetSamVac": ("样品仓真空", "设备原始真空单位"),
    "CheckAutoFocus": ("自动对焦状态", "0=结束；1=运行；失败由协议错误返回"),
    **{f"GetStage{axis}Status": (f"{axis} 轴位置及停止位", "设备原始轴单位；第二项 1=停止") for axis in "XYZRT"},
    "GetHvStatus": ("电子枪状态", "0=关闭；1=开启；2=切换中"),
    "GetBseCfg": ("BSE 四象限配置", "四组 状态位/偏压原始值"),
    "AllowOpenGun": ("设备是否允许开枪", "bool"), "IsOpenGun": ("是否开枪", "bool"),
    "IsChangingSamp": ("换样状态", "0=未换样；1=换样中；-1=未知"),
    "GetSetHV": ("设定加速电压", "kV"), "GetBseState": ("BSE 启用状态", "bool"),
    "GetEDSEnable": ("EDS 附件启用状态，不等于采谱", "0/1"),
    **{f"Get{side}Center{axis}": (f"{side} 束中心 {axis}", "设备原始单位") for side in ("Up", "Down") for axis in "XY"},
    "GetVacuum": ("样品仓及枪真空", "设备原始真空单位"),
    "GetPumpState": ("机械泵与分子泵状态", "0/1；-1=未知，不能视为正常"),
    "GetValveState": ("六个阀门状态", "0/1，顺序按上位机协议"),
    "GetCustomRotation": ("自定义扫描旋转", "度"),
}


def write(params, getter=None, *, expected=None, category="imaging", note=""):
    return {"parameters": params, "verify": getter, "expected": expected,
            "category": category, "note": note}


PERCENT = number("%", 0, 100)
SIGNED_PERCENT = number("%", -100, 100)
WRITES = {
    "SetMag": write([number("倍", 1, 1000000, True)], "GetMag"),
    "SetHV": write([number("kV", .1, 30)], "GetSetHV", category="high_voltage"),
    "SetBeamSz": write([choice([0, 1, 2, 3])], "GetBeamSz"),
    "SetAvMode": write([choice([0, 1])], "GetAvMode"),
    # Preserve the legacy typed action, but do not verify SnapAvPointGear by
    # querying VideoAvPointGear. They are separate upstream properties.
    "SetAvLenth": {**write([choice(list(range(1, 8)))], None),
                   "note": "修改拍照档位；GetAvLenth 读取视频档位，不能用于完成确认"},
    "SetAvFrames": write([number("帧", 1, 256, True)], "GetAvFrames"),
    "SetSEGain": write([PERCENT], "GetSEGain"),
    "SetSECV": write([number("设备原始电压单位", 0, 10000, True)], "GetSECV"),
    "SetOB": write([PERCENT], "GetOB", note="输入百分比；回读原始 DAC，按 65535 换算"),
    "SetOBRaw": write([number("原始 DAC", 0, 65535, True)], "GetOB"),
    "SetRotation": write([number("度", -360, 360)], "GetRotation"),
    "SetEDChannel": write([choice([0, 1])], "GetEDChannel"),
    "SetWD": write([number("mm", .1, 100)], "GetWD"),
    "SetAst": write([number("原始 DAC", -65535, 65535, True)] * 2, "GetAst"),
    "SetImgSz": write([choice([0, 1, 2, 3])], "GetImgSz"),
    "VideoPlay": write([choice([0, 1])], "IsVideoAlive", category="video"),
    "SetWindowPos": write([number("像素", 0, 65535, True)] * 2, "GetWindowPos"),
    "SetHV_ON": write([], "GetHvStatus", expected=[1], category="gun_on"),
    "SetHV_OFF": write([], "GetHvStatus", expected=[0], category="gun_off"),
    "RequestOpenVac": write([], "CheckRequestOpenVac", expected=[0], category="sample_begin"),
    "RequestCloseVac": write([], "CheckRequestCloseVac", expected=[0], category="sample_end"),
    "SnapNav": write([], "CheckSnapNav", expected=[0], category="navigation_capture"),
    "AutoFocus": {**write([number("初始物镜 DAC", 0, 65535, True),
                           number("搜索总宽度 DAC", 1, 65535, True)],
                          "CheckAutoFocus", expected=[0], category="autofocus",
                          note="空参数全范围对焦；或提供初始物镜值、搜索总宽度两个参数"), "allow_empty": True},
    **{f"SetStig{axis}": write([SIGNED_PERCENT], f"GetStig{axis}") for axis in "XY"},
    **{f"SetCompress{axis}": write([PERCENT], f"GetCompress{axis}") for axis in "AB"},
    "SetBseCfg": write([choice([0, 1, 2, 3]), number("偏压原始值", -65535, 65535, True)] * 4, "GetBseCfg"),
    "SetBseState": write([choice([0, 1])], "GetBseState"),
    "SetEDSEnable": write([choice([0, 1])], "GetEDSEnable", category="eds_accessory",
                          note="仅切换上位机附件开关，不采集 EDS"),
    **{f"Set{side}Center{axis}": write([number("设备原始单位", -65535, 65535)], f"Get{side}Center{axis}")
       for side in ("Up", "Down") for axis in "XY"},
    "SetCustomRotation": write([number("度", -360, 360)], "GetCustomRotation"),
}

# Only an actual upstream stub is unavailable. Acknowledgement-only commands
# are exposed by sem_rpc_call without claiming physical completion.
UNAVAILABLE = {
    "SetIe": "上位机实现直接返回 InvalidCmd，无可调用功能",
}
MAPPED = {"SetStagePos": "sem_move_xy", "GetNavImgBuf": "sem_get_navigation",
          "Snap": "sem_capture", "GetImg": "sem_get_image", "GetNavImg": "sem_get_navigation"}


def argument_schema(command):
    spec = WRITES[command]
    # Positional wire arguments are typed individually, not free-form strings.
    params = spec["parameters"]
    schema = {"type": "array", **({"prefixItems": deepcopy(params)} if params else {}), "items": False,
              "minItems": len(params), "maxItems": len(params)}
    return {"oneOf": [{"type": "array", "maxItems": 0}, schema]} if spec.get("allow_empty") else schema


def catalog(allowed=()):
    from backend.sem_rpc_full import full_catalog
    return full_catalog(allowed)
