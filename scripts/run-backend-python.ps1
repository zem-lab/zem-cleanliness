param(
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$PythonArgs
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
$python = if ($env:ZEM_PYTHON) {
  $env:ZEM_PYTHON
} elseif (Test-Path $venvPython) {
  $venvPython
} else {
  'python'
}

& $python @PythonArgs
exit $LASTEXITCODE
