# P0: Uniform source control run
# Purpose: Verify that "source distribution determines core count" by running with uniform rho=2.0
param(
    [double]$duration = 0.5
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$output_name = "multilayer_results_uniform_source.json"
$output_path = Join-Path $ScriptDir $output_name

if (Test-Path $output_path) {
    Write-Host "[SKIP] Uniform source control — output already exists: $output_name"
    exit 0
}

Write-Host "================================================"
Write-Host "P0 UNIFORM SOURCE: gamma=6.0, uniform rho=2.0, duration=$duration h"
Write-Host "Output: $output_name"
Write-Host "Start: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "================================================"

$env:SIM_GAMMA = "6.0"
$env:SIM_BETA = "0.6"
$env:SIM_N_SATS = "1000"
$env:SIM_GRID_RES = "40"
$env:SIM_DURATION = "$duration"
$env:SIM_UNIFORM_SOURCE = "1"
$env:SIM_OUTPUT_PHI = "0"

$proc = Start-Process -FilePath (Join-Path $ScriptDir "multilayer_sim_real.exe") `
    -WorkingDirectory $ScriptDir -Wait -PassThru -NoNewWindow

$default_output = Join-Path $ScriptDir "multilayer_results_real.json"
if (Test-Path $default_output) {
    Move-Item $default_output $output_path -Force
    Write-Host "[OK] Uniform source control done, exit=$($proc.ExitCode)"
    Write-Host "End: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
} else {
    Write-Host "[FAIL] Uniform source control — no output file generated"
    exit 1
}