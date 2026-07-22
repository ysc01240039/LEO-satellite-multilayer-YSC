# P1: Beta scan (beta=0.1 and beta=2.0)
# Purpose: Verify P3 — n_cores is independent of beta (topological invariance)
param(
    [double]$beta = 0.1,
    [double]$duration = 0.5
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$output_name = "multilayer_results_beta_$beta.json"
$output_path = Join-Path $ScriptDir $output_name

if (Test-Path $output_path) {
    Write-Host "[SKIP] beta=$beta — output already exists: $output_name"
    exit 0
}

Write-Host "================================================"
Write-Host "P1 BETA: beta=$beta (gamma=6.0), duration=$duration h"
Write-Host "Output: $output_name"
Write-Host "Start: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "================================================"

$env:SIM_GAMMA = "6.0"
$env:SIM_BETA = "$beta"
$env:SIM_N_SATS = "1000"
$env:SIM_GRID_RES = "40"
$env:SIM_DURATION = "$duration"
$env:SIM_OUTPUT_PHI = "0"

$proc = Start-Process -FilePath (Join-Path $ScriptDir "multilayer_sim_real.exe") `
    -WorkingDirectory $ScriptDir -Wait -PassThru -NoNewWindow

$default_output = Join-Path $ScriptDir "multilayer_results_real.json"
if (Test-Path $default_output) {
    Move-Item $default_output $output_path -Force
    Write-Host "[OK] beta=$beta done, exit=$($proc.ExitCode)"
    Write-Host "End: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
} else {
    Write-Host "[FAIL] beta=$beta — no output file generated"
    exit 1
}