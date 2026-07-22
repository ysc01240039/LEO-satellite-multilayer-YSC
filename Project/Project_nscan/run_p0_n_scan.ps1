# P0: n_scan (Round 35)
# Validates sub-linear scaling n_cores ∝ N^0.275
# Runs in parallel with gamma_critical scan via Project_nscan directory
$n_values = @(200, 400, 600, 800, 1000, 2000)
$gamma = 6.0
$beta = 0.6
$grid_res = 40
$duration = 0.5

Write-Host "================================================"
Write-Host "P0: n_scan (Parallel)"
Write-Host "gamma = $gamma, beta = $beta"
Write-Host "N values: $n_values"
Write-Host "Duration per run: $duration hours"
Write-Host "Estimated total: $($n_values.Count * $duration) hours"
Write-Host "================================================"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$completed = 0
$failed = 0
$skipped = 0

foreach ($n in $n_values) {
    $output_name = "multilayer_results_nscan_N$n.json"
    $output_path = Join-Path $ScriptDir $output_name
    
    if (Test-Path $output_path) {
        Write-Host "[SKIP] N=$n — output already exists"
        $skipped++
        continue
    }
    
    Write-Host "[RUN] N=$n ..."
    
    $env:SIM_GAMMA = "$gamma"
    $env:SIM_BETA = "$beta"
    $env:SIM_N_SATS = "$n"
    $env:SIM_GRID_RES = "$grid_res"
    $env:SIM_DURATION = "$duration"
    $env:SIM_OUTPUT_PHI = "0"
    
    $proc = Start-Process -FilePath (Join-Path $ScriptDir "multilayer_sim_real.exe") `
        -WorkingDirectory $ScriptDir -Wait -PassThru -NoNewWindow
    
    $default_output = Join-Path $ScriptDir "multilayer_results_real.json"
    if (Test-Path $default_output) {
        Move-Item $default_output $output_path -Force
        Write-Host "  [OK] N=$n done, exit=$($proc.ExitCode)"
        $completed++
    } else {
        Write-Host "  [FAIL] N=$n : no output file generated"
        $failed++
    }
}

Write-Host ""
Write-Host "================================================"
Write-Host "P0 n_scan COMPLETE"
Write-Host "Completed: $completed, Failed: $failed, Skipped: $skipped"
Write-Host "================================================"