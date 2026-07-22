# P0: Long gamma runs (2h each) for gamma=0.4 and gamma=1.0
# Purpose: Test whether gamma effects emerge at 18x tau_diff (vs 0.5h = 4.5x tau_diff)
param(
    [double]$gamma = 0.4,
    [double]$duration = 2.0
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$output_name = "multilayer_results_gamma_long_$gamma.json"
$output_path = Join-Path $ScriptDir $output_name

if (Test-Path $output_path) {
    Write-Host "[SKIP] gamma=$gamma (2h) — output already exists: $output_name"
    exit 0
}

$ratio = [math]::Round($gamma / 0.4441, 2)
Write-Host "================================================"
Write-Host "P0 LONG: gamma=$gamma ($ratio x gamma_c), duration=$duration h"
Write-Host "Output: $output_name"
Write-Host "Start: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "================================================"

$env:SIM_GAMMA = "$gamma"
$env:SIM_BETA = "0.6"
$env:SIM_N_SATS = "1000"
$env:SIM_GRID_RES = "40"
$env:SIM_DURATION = "$duration"
$env:SIM_OUTPUT_PHI = "0"

$proc = Start-Process -FilePath (Join-Path $ScriptDir "multilayer_sim_real.exe") `
    -WorkingDirectory $ScriptDir -Wait -PassThru -NoNewWindow

$default_output = Join-Path $ScriptDir "multilayer_results_real.json"
if (Test-Path $default_output) {
    Move-Item $default_output $output_path -Force
    Write-Host "[OK] gamma=$gamma (2h) done, exit=$($proc.ExitCode)"
    Write-Host "End: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
} else {
    Write-Host "[FAIL] gamma=$gamma (2h) — no output file generated"
    exit 1
}