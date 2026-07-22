# P1: No-source control run (rho=0 everywhere)
# Purpose: Verify that source is necessary for core formation
param(
    [double]$duration = 0.5
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$output_name = "multilayer_results_no_source.json"
$output_path = Join-Path $ScriptDir $output_name

if (Test-Path $output_path) {
    Write-Host "[SKIP] No-source control — output already exists: $output_name"
    exit 0
}

Write-Host "================================================"
Write-Host "P1 NO-SOURCE: gamma=6.0, no satellites (n_sats=0), duration=$duration h"
Write-Host "Output: $output_name"
Write-Host "Start: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "================================================"

$env:SIM_GAMMA = "6.0"
$env:SIM_BETA = "0.6"
$env:SIM_N_SATS = "0"
$env:SIM_GRID_RES = "40"
$env:SIM_DURATION = "$duration"
$env:SIM_OUTPUT_PHI = "0"

$proc = Start-Process -FilePath (Join-Path $ScriptDir "multilayer_sim_real.exe") `
    -WorkingDirectory $ScriptDir -Wait -PassThru -NoNewWindow

$default_output = Join-Path $ScriptDir "multilayer_results_real.json"
if (Test-Path $default_output) {
    Move-Item $default_output $output_path -Force
    Write-Host "[OK] No-source control done, exit=$($proc.ExitCode)"
    Write-Host "End: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
} else {
    Write-Host "[FAIL] No-source control — no output file generated"
    exit 1
}