const { spawn } = require('node:child_process')
const path = require('node:path')

const root = path.join(__dirname, '..')
const viteEntry = path.join(root, 'node_modules', 'vite', 'bin', 'vite.js')
const electronExecutable = process.platform === 'win32'
  ? path.join(root, 'node_modules', 'electron', 'dist', 'electron.exe')
  : path.join(root, 'node_modules', '.bin', 'electron')

const vite = spawn(process.execPath, [viteEntry, '--host', '127.0.0.1'], {
  cwd: root,
  stdio: 'inherit',
  shell: false
})

async function waitForVite() {
  for (let attempt = 0; attempt < 80; attempt += 1) {
    try {
      const response = await fetch('http://127.0.0.1:5173')
      if (response.ok) return
    } catch {}
    await new Promise((resolve) => setTimeout(resolve, 250))
  }
  throw new Error('Vite 启动超时')
}

waitForVite().then(() => {
  const electron = spawn(electronExecutable, ['.'], {
    cwd: root,
    stdio: 'inherit',
    env: { ...process.env, VITE_DEV_SERVER_URL: 'http://127.0.0.1:5173' }
  })
  electron.on('exit', (code) => {
    vite.kill()
    process.exit(code || 0)
  })
}).catch((error) => {
  console.error(error)
  vite.kill()
  process.exit(1)
})
