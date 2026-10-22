import { Component } from 'react'
import type { ErrorInfo, ReactNode } from 'react'
import { Copy, RotateCcw, TriangleAlert } from 'lucide-react'
import { logClientEvent } from './api'
import { createErrorRecord, loadErrorRecords, mergeErrorRecord, saveErrorRecords } from './errors'

type ErrorBoundaryProps = {
  children: ReactNode
}

type ErrorBoundaryState = {
  error: Error | null
  details: string
}

export default class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = { error: null, details: '' }

  static getDerivedStateFromError(error: Error): Partial<ErrorBoundaryState> {
    return { error, details: error.stack || error.message }
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    const details = [error.stack || error.message, info.componentStack].filter(Boolean).join('\n')
    logClientEvent('error', 'react_render_error', { message: error.message, stack: details })
    const record = createErrorRecord(details, '前端页面渲染')
    saveErrorRecords(mergeErrorRecord(loadErrorRecords(), record))
    this.setState({ details })
  }

  render() {
    if (!this.state.error) return this.props.children
    return (
      <main className="fatalErrorPage" role="alert">
        <div className="fatalErrorIcon"><TriangleAlert size={28} /></div>
        <h1>页面运行出错</h1>
        <p>前端组件发生异常，错误已经保存到本机错误记录。重新加载后可在右上角“错误记录”中继续查看。</p>
        <div className="fatalErrorActions">
          <button type="button" onClick={() => window.location.reload()}><RotateCcw size={15} />重新加载应用</button>
          <button type="button" onClick={() => copyDetails(this.state.details)}><Copy size={15} />复制技术详情</button>
        </div>
        <details><summary>查看技术详情</summary><pre>{this.state.details}</pre></details>
      </main>
    )
  }
}

function copyDetails(value: string): void {
  if (!navigator.clipboard) return
  void navigator.clipboard.writeText(value).catch(() => undefined)
}
