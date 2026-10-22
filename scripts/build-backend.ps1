$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
$python = $env:ZEM_PYTHON
if (-not $python) {
  if (-not (Test-Path $venvPython)) {
    & (Join-Path $PSScriptRoot 'setup-backend.ps1')
    if ($LASTEXITCODE -ne 0) { throw "Failed to prepare backend environment" }
  }
  $python = $venvPython
}
$distDir = Join-Path $projectRoot 'backend-dist'
$workDir = Join-Path $projectRoot 'backend-build'
$specDir = Join-Path $projectRoot 'backend-spec'

& $python -m pip install --disable-pip-version-check -r (Join-Path $projectRoot 'backend\requirements.txt')
if ($LASTEXITCODE -ne 0) { throw "Failed to install backend requirements" }
& $python -m pip install --disable-pip-version-check --index-url https://pypi.tuna.tsinghua.edu.cn/simple pyinstaller==6.16.0
if ($LASTEXITCODE -ne 0) { throw "Failed to install PyInstaller" }

& $python -m PyInstaller `
  --noconfirm `
  --clean `
  --onefile `
  --console `
  --name zem-cleanliness-backend `
  --paths $projectRoot `
  --distpath $distDir `
  --workpath $workDir `
  --specpath $specDir `
  --collect-all onnxruntime `
  --collect-all cv2 `
  --collect-all reportlab `
  --collect-submodules mcp.server `
  --collect-submodules mcp.client `
  --collect-submodules mcp.shared `
  --copy-metadata mcp `
  --collect-submodules scipy._external `
  --collect-submodules backend.cleanliness_core.eds.generated.v2 `
  --hidden-import backend.cleanliness_core.cleanliness_task `
  --hidden-import backend.cleanliness_core.cleanliness_detection `
  --hidden-import backend.cleanliness_core.cleanliness_report `
  --hidden-import backend.cleanliness_core.ab_point_calibration `
  --hidden-import backend.cleanliness_core.hole_center_locator `
  --hidden-import backend.cleanliness_core.energy_spectrum `
  --hidden-import zemrpc.zem.zem_api `
  (Join-Path $projectRoot 'backend\server.py')
if ($LASTEXITCODE -ne 0) { throw "Backend EXE build failed" }

if (-not (Test-Path (Join-Path $distDir 'zem-cleanliness-backend.exe'))) {
  throw '后端 EXE 构建失败'
}

# A live-driver lazy import must work in the frozen EXE, not only in Python.
& (Join-Path $distDir 'zem-cleanliness-backend.exe') --check-runtime
if ($LASTEXITCODE -ne 0) { throw 'Packaged microscope runtime dependency check failed' }
