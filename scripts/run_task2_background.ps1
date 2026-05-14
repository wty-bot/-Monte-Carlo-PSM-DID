param(
    [int]$NSims = 500,
    [switch]$Smoke
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
$stdout = Join-Path $repoRoot "outputs\logs\task2_run_stdout.log"
$stderr = Join-Path $repoRoot "outputs\logs\task2_run_stderr.log"

if (-not (Test-Path $python)) {
    throw "Python executable not found: $python"
}

if ($Smoke) {
    & $python "scripts/run_task2_estimate.py" "--smoke" 1>> $stdout 2>> $stderr
} else {
    & $python "scripts/run_task2_estimate.py" "--n-sims" "$NSims" 1>> $stdout 2>> $stderr
}
