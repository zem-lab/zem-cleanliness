$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $PSScriptRoot
$venvRoot = Join-Path $projectRoot '.venv'
$python = if ($env:ZEM_BOOTSTRAP_PYTHON) { $env:ZEM_BOOTSTRAP_PYTHON } else { 'python' }

if (-not (Test-Path (Join-Path $venvRoot 'Scripts\python.exe'))) {
  & $python -m venv $venvRoot
  if ($LASTEXITCODE -ne 0) { throw "Failed to create backend virtual environment" }
}

$venvPython = Join-Path $venvRoot 'Scripts\python.exe'
& $venvPython -m pip install --disable-pip-version-check --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "Failed to upgrade pip in backend virtual environment" }
& $venvPython -m pip install --disable-pip-version-check -r (Join-Path $projectRoot 'backend\requirements.txt')
if ($LASTEXITCODE -ne 0) { throw "Failed to install backend requirements" }

Write-Output "Backend environment ready: $venvPython"
