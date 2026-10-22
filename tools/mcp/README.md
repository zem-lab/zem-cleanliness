# ZEPTOOLS 清洁度软件 MCP 服务器

让任意 Agent（MCP 客户端）驱动**真实的清洁度检测软件**：Agent 调用 MCP 工具 → 服务器内部用 Playwright（与 `e2e/*.cjs` 同栈）操作 Electron 界面 → 用户看到软件像被人点击一样变化。

本仓库另有一套**电镜 RPC MCP**（`backend/sem_mcp.py`，工具名多为 `sem_rpc_*` / `sem_capture` 等），直接调电镜接口、不点 UI。二者可并存：本目录是「控软件」，后者是「控电镜」。配置 Agent 时请分清服务器入口，避免混用。

软件由 `app_launch` 拉起时会注入 `ZEM_AGENT_CONTROL=1`，界面顶部显示「当前软件正在被 Agent 控制」徽标，整个窗口带淡蓝色呼吸描边，用户一眼可辨。

## 范围约定

- **不包含光电联用（光镜关联分析）**：光镜数据导入、坐标标定（含 AB 系列）、检测运行页均不在 MCP 范围内；`get_tasks` 可列出光镜任务但 `open_task` 会拒绝打开。
- **ROI 圈选只能由人完成**：在导航图上拖动圈选扫描区域不提供工具；`sem_preview_grid`、`sem_start_analysis` 在人圈选前不可用并明确提示。
- **工程师面板不进 MCP**：密码保护的高风险算法参数区，Agent 不应绕过人工授权。
- **观测类工具**（任务/设置/设备/报告数据读取）通过应用自身 IPC 通道读后端，界面无变化；**操作类工具一律真实 UI 点击**，用户全程可见。

## 硬件安全门

`app_launch` 默认 `allowHardware=false`：`sem_get_navigation`（连接电镜读导航图）与 `sem_start_analysis`（移动样品台、拍照、采谱、可能自动关枪）会被直接拒绝。确认设备与样品安全后，显式传 `allowHardware: true` 启动才会放行。

## 启动

```bash
cd tools/mcp
pnpm install          # 首次
node server.cjs       # stdio 传输，供 MCP 客户端拉起
```

自测（会真实启动一次软件，不需要电镜硬件）：

```bash
node smoke.cjs
```

依赖：仓库根目录需已 `pnpm install` 且 `pnpm build`（Electron、`dist/` 前端构建产物、后端 Python 环境）。

## Agent 配置示例

```json
{
  "mcpServers": {
    "zem-cleanliness": {
      "command": "node",
      "args": ["<workspace>/tools/mcp/server.cjs"]
    }
  }
}
```

## 工具一览（24 个）

**生命周期**：`app_launch`（pythonPath / taskRoot / allowHardware）· `app_close` · `app_status`

**导航与原子操作**：`navigate`（home/reports/settings/new-task/sem-direct）· `ui_read` · `ui_click` · `ui_fill` · `ui_select`（兜底，语义工具未覆盖的新界面可自行组合）

**观测**：`screenshot`（返回图片）· `console_errors` · `get_errors` / `clear_errors`（错误中心）

**任务管理**：`get_tasks`（全量任务列表）· `open_task`（仅直接 SEM 分析任务）· `get_report_data`（完整报告数据）

**设备**：`get_devices` · `refresh_devices`

**系统设置**：`get_settings` · `set_settings`（设备地址/端口、EDS 开关、恢复台位、自动关枪、开工检查、拍照档位）

**直接 SEM 分析**：`sem_get_navigation`[硬件] · `set_scan_settings`（样品名/形状/FOV/分辨率/重叠/最小颗粒/EDS）· `set_segmentation_model`（yolo / sam3 + 服务地址/提示词/阈值）· `sem_preview_grid` · `sem_start_analysis`[硬件] · `sem_pause_analysis` · `sem_generate_report` · `sem_get_status` · `sem_get_particles`（全量颗粒明细）· `sem_open`（打开 PDF/网页报告/CSV/时间报告/任务目录）

典型协作流程（人机配合）：
1. Agent：`app_launch({ allowHardware: true })` → `sem_get_navigation` 获取导航图
2. 人：在导航图上圈选扫描区域（ROI）
3. Agent：`set_scan_settings` + `set_segmentation_model(sam3, url, score)` → `sem_preview_grid` 确认视野数 → `sem_start_analysis`
4. Agent：轮询 `sem_get_status` / `sem_get_particles`，完成后 `sem_generate_report` → `sem_open('report_pdf')` 给用户看报告

## 会话数据

默认隔离在 `.tmp/mcp-sessions/<会话>/tasks`，不影响真实历史任务；要复用历史任务数据需显式传 `taskRoot`（指向软件任务目录）。
