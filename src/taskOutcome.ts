import type { TaskManifest } from './api'

export function gunShutdownSummary(task: TaskManifest): string {
  const cleanup = task.cleanup as { gunOffRequested?: boolean; gunOffVerified?: boolean; gunOffError?: string } | undefined
  if (!cleanup?.gunOffRequested) return ''
  if (cleanup.gunOffVerified) return '电子枪已关闭（设备反馈确认）。'
  return `自动关枪未确认，请人工检查电子枪状态。${cleanup.gunOffError || ''}`
}

export function completedTaskSummary(task: TaskManifest): string {
  const records = task.particles || []
  const succeeded = records.filter(record => record.status === 'SUCCEEDED').length
  const failed = records.filter(record => record.status === 'FAILED')
  const edsFailed = failed.filter(record => Array.isArray(record.edsAttempts) && record.edsAttempts.length > 0).length
  const total = Number(task.selection?.count || records.length)
  const pending = Math.max(0, total - succeeded - failed.length)
  const reports = task.reports?.pdf ? '报告已生成。' : '报告尚未生成。'
  return `检测已完成：成功 ${succeeded} 个，定位/采图未通过 ${failed.length - edsFailed} 个，能谱失败 ${edsFailed} 个。${pending ? `另有 ${pending} 个未处理。` : ''}${reports}${gunShutdownSummary(task)}`
}
