"""OpenAPI 3.0 specification for the third-party report data API.

Served at GET /openapi.json; the interactive Swagger UI page at GET /docs
renders this spec. Keep the schema descriptions in Chinese: the audience is
the third-party integrator who renders their own PDF.
"""
from __future__ import annotations

from typing import Any

API_TITLE = "ZEM Cleanliness 报告数据接口"
API_VERSION = "1.0.0"


def openapi_spec() -> dict[str, Any]:
    return {
        "openapi": "3.0.3",
        "info": {
            "title": API_TITLE,
            "version": API_VERSION,
            "description": (
                "光镜关联分析（光电联用）任务报告原始数据导出接口。"
                "第三方在软件生成报告后调用，拿到结构化数据（颗粒清单、EDS 元素成分、"
                "统计汇总、图片与报告文件下载地址、完整任务 manifest），"
                "自行排版生成所需样式的 PDF。现有软件内 PDF 导出流程不受影响。"
            ),
        },
        "servers": [
            {"url": "http://127.0.0.1:8765", "description": "本机后端默认地址"},
        ],
        "paths": {
            "/tasks/{taskId}/report-data": {
                "get": {
                    "summary": "获取报告原始数据",
                    "description": (
                        "返回报告生成之前的结构化数据。仅当任务是光镜关联分析类型"
                        "（CLEANLINESS_DETECTION）且报告已生成后可用；否则返回 400。"
                        "响应中的图片与报告文件均为可下载 URL，主机与调用地址一致。"
                    ),
                    "tags": ["报告数据"],
                    "parameters": [
                        {
                            "name": "taskId",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string"},
                            "description": "任务编号（任务列表接口或软件界面中的 taskId）",
                            "example": "0d328e34-771d-4fbc-b100-9fa25a3ede6d",
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "报告数据",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ReportData"},
                                }
                            },
                        },
                        "400": {
                            "description": (
                                "任务不存在 / 任务不是光镜关联分析类型 / 报告尚未生成。"
                                "返回 {\"error\": \"原因\"}。"
                            ),
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/Error"},
                                    "example": {"error": "报告尚未生成，请先在软件中导出报告后再调用该接口"},
                                }
                            },
                        },
                    },
                }
            },
            "/tasks/{taskId}/files/{path}": {
                "get": {
                    "summary": "下载任务文件",
                    "description": (
                        "下载报告数据中引用的图片（光镜图、SEM 原图/标注图/颗粒图）"
                        "与报告文件（PDF/HTML/CSV/JSON）。path 为响应里的 URL 最后一段"
                        "之后的部分，URL 本身可直接使用。"
                    ),
                    "tags": ["报告数据"],
                    "parameters": [
                        {
                            "name": "taskId",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string"},
                        },
                        {
                            "name": "path",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string"},
                            "description": "任务目录内的相对路径，例如 optical/0006.png",
                            "example": "optical/0006.png",
                        },
                    ],
                    "responses": {
                        "200": {"description": "文件内容（图片/PDF/CSV 等对应 MIME）"},
                        "404": {
                            "description": "文件不存在",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/Error"},
                                }
                            },
                        },
                    },
                }
            },
        },
        "components": {
            "schemas": {
                "Error": {
                    "type": "object",
                    "properties": {
                        "error": {"type": "string", "description": "错误原因（中文）"},
                    },
                },
                "ReportData": {
                    "type": "object",
                    "description": "报告原始数据。顶层字段为稳定契约；manifest 为内部结构原文，仅供需要全部字段的一方使用。",
                    "properties": {
                        "apiVersion": {"type": "string", "description": "数据契约版本", "example": "1"},
                        "taskId": {"type": "string", "description": "任务编号"},
                        "taskType": {"type": "string", "example": "CLEANLINESS_DETECTION"},
                        "taskStatus": {"type": "string", "description": "任务状态", "example": "COMPLETED"},
                        "exportedAt": {"type": "string", "format": "date-time", "description": "本次导出时间（ISO 8601 含时区）"},
                        "reportedAt": {"type": "string", "format": "date-time", "nullable": True, "description": "软件内报告生成时间"},
                        "workpiece": {"$ref": "#/components/schemas/Workpiece"},
                        "selection": {"$ref": "#/components/schemas/Selection"},
                        "calibration": {"$ref": "#/components/schemas/Calibration"},
                        "summary": {"$ref": "#/components/schemas/Summary"},
                        "particles": {
                            "type": "array",
                            "description": "颗粒明细，按执行顺序排列",
                            "items": {"$ref": "#/components/schemas/Particle"},
                        },
                        "reports": {
                            "type": "object",
                            "description": "软件已生成报告文件的下载 URL",
                            "properties": {
                                "pdf": {"type": "string", "nullable": True},
                                "html": {"type": "string", "nullable": True},
                                "csv": {"type": "string", "nullable": True},
                                "json": {"type": "string", "nullable": True},
                            },
                        },
                        "manifest": {
                            "type": "object",
                            "description": "完整任务清单（task.json）原文。结构跟随软件版本演进，不做向后兼容承诺；稳定字段请使用顶层清洗字段。",
                        },
                    },
                },
                "Workpiece": {
                    "type": "object",
                    "properties": {
                        "projectId": {"type": "string", "description": "项目编号"},
                        "workpieceId": {"type": "string", "description": "工件编号"},
                        "workpieceName": {"type": "string", "description": "样品名称"},
                        "opticalParticleCount": {"type": "integer", "description": "光镜导出文件中的颗粒总数"},
                        "registrationMode": {"type": "string", "description": "配准方式，例如 twoPointAB", "nullable": True},
                    },
                },
                "Selection": {
                    "type": "object",
                    "properties": {
                        "filters": {"type": "object", "description": "进入任务的筛选条件（minEcdUm 等）"},
                        "selectedCount": {"type": "integer", "description": "进入任务的颗粒数量"},
                    },
                },
                "Calibration": {
                    "type": "object",
                    "properties": {
                        "strategy": {"type": "string", "description": "标定策略", "example": "multi_particle"},
                        "accepted": {"type": "boolean", "description": "标定结果是否已确认"},
                    },
                },
                "Summary": {
                    "type": "object",
                    "description": "与软件报告一致的统计口径",
                    "properties": {
                        "total": {"type": "integer", "description": "任务颗粒总数"},
                        "succeeded": {"type": "integer", "description": "成功颗粒数（status=SUCCEEDED）"},
                        "failed": {"type": "integer", "description": "失败颗粒数"},
                        "matchedHigh": {"type": "integer", "description": "高置信匹配数"},
                        "matchedMedium": {"type": "integer", "description": "中置信匹配数"},
                        "successRate": {"type": "number", "description": "成功率 0–1"},
                        "centerErrorUm": {
                            "type": "object",
                            "description": "定位中心误差（µm）",
                            "properties": {
                                "mean": {"type": "number", "nullable": True},
                                "minimum": {"type": "number", "nullable": True},
                                "maximum": {"type": "number", "nullable": True},
                                "rms": {"type": "number", "nullable": True},
                            },
                        },
                    },
                },
                "Particle": {
                    "type": "object",
                    "properties": {
                        "sequence": {"type": "integer", "description": "颗粒序号"},
                        "particleId": {"type": "string", "description": "光镜导出文件中的颗粒 ID"},
                        "label": {"type": "string", "description": "颗粒标签（metal/fiber 等）", "nullable": True},
                        "rank": {"type": "integer", "description": "执行顺序（1 起）", "nullable": True},
                        "status": {"type": "string", "description": "颗粒结果状态", "example": "SUCCEEDED"},
                        "dimensions": {"$ref": "#/components/schemas/Dimensions"},
                        "opticalPositionUm": {
                            "type": "object",
                            "description": "光镜 AB 坐标（µm）",
                            "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
                        },
                        "targetStageUm": {
                            "type": "object",
                            "description": "预测样品台坐标（µm）",
                            "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
                        },
                        "actualStageUm": {
                            "type": "object",
                            "description": "实测样品台坐标（µm）",
                            "nullable": True,
                            "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
                        },
                        "stagePositionErrorUm": {"type": "number", "nullable": True, "description": "台位误差（µm）"},
                        "match": {
                            "type": "object",
                            "nullable": True,
                            "description": "匹配结论",
                            "properties": {
                                "status": {"type": "string", "example": "MATCHED_HIGH"},
                                "confidence": {"type": "string", "example": "high"},
                            },
                        },
                        "eds": {"$ref": "#/components/schemas/Eds"},
                        "images": {"$ref": "#/components/schemas/Images"},
                        "timingSeconds": {
                            "type": "object",
                            "description": "各阶段耗时（秒）",
                            "properties": {
                                "stageMove": {"type": "number", "nullable": True},
                                "capture": {"type": "number", "nullable": True},
                                "match": {"type": "number", "nullable": True},
                                "eds": {"type": "number", "nullable": True},
                                "total": {"type": "number", "nullable": True},
                            },
                        },
                    },
                },
                "Dimensions": {
                    "type": "object",
                    "description": "颗粒尺寸（µm / µm²）",
                    "properties": {
                        "lengthUm": {"type": "number", "nullable": True},
                        "widthUm": {"type": "number", "nullable": True},
                        "areaUm2": {"type": "number", "nullable": True},
                        "equivalentDiameterUm": {"type": "number", "nullable": True, "description": "等效圆直径"},
                        "aspectRatio": {"type": "number", "nullable": True, "description": "长宽比"},
                        "roundness": {"type": "number", "nullable": True},
                    },
                },
                "Eds": {
                    "type": "object",
                    "description": "能谱分析结果",
                    "properties": {
                        "acquired": {"type": "boolean", "description": "是否成功采谱"},
                        "acquiredAt": {"type": "string", "nullable": True, "description": "采谱时间"},
                        "quantMethod": {"type": "string", "nullable": True, "description": "定量方法"},
                        "unit": {"type": "string", "description": "元素含量单位", "example": "percent"},
                        "liveTimeSeconds": {"type": "number", "nullable": True},
                        "realTimeSeconds": {"type": "number", "nullable": True},
                        "materialClassification": {
                            "type": "object",
                            "nullable": True,
                            "description": "材料分类（按硬度体系）",
                            "properties": {
                                "materialClass": {"type": "string", "nullable": True},
                                "parentClass": {"type": "string", "nullable": True},
                                "hardnessLabel": {"type": "string", "nullable": True},
                            },
                        },
                        "vdaClassification": {
                            "type": "object",
                            "nullable": True,
                            "description": "VDA 分类",
                            "properties": {
                                "materialClass": {"type": "string", "nullable": True},
                                "hardnessLabel": {"type": "string", "nullable": True},
                            },
                        },
                        "elements": {
                            "type": "array",
                            "description": "元素成分（重量百分比，保留 4 位小数，按含量降序）",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "symbol": {"type": "string", "example": "Fe"},
                                    "weightPercent": {"type": "number", "example": 3.77},
                                },
                            },
                        },
                    },
                },
                "Images": {
                    "type": "object",
                    "description": "图片下载 URL（缺失为 null）",
                    "properties": {
                        "opticalImage": {"type": "string", "nullable": True, "description": "光镜颗粒图"},
                        "semRawImage": {"type": "string", "nullable": True, "description": "SEM 原始图"},
                        "semDisplayImage": {"type": "string", "nullable": True, "description": "SEM 显示图"},
                        "semAnnotatedImage": {"type": "string", "nullable": True, "description": "SEM 标注图（含编号）"},
                        "semParticleImage": {"type": "string", "nullable": True, "description": "SEM 颗粒裁剪图"},
                    },
                },
            }
        },
    }
