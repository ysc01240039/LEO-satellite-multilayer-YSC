# P0: gamma_critical scan (Round 31)
# Validates perturbation theory near gamma_c
$gamma_values = @(0.40, 0.43, 0.445, 0.46, 0.50, 0.60, 0.80, 1.00)
$beta = 0.6
$n_sats = 1000
$grid_res = 40
$duration = 0.5

Write-Host "================================================"
Write-Host "P0: gamma_critical scan"
Write-Host "gamma_c = 0.4441 (beta=0.6)"
Write-Host "Values: $gamma_values"
Write-Host "Duration per run: $duration hours"
Write-Host "Estimated total: $($gamma_values.Count * $duration) hours"
Write-Host "================================================"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$completed = 0
$failed = 0
$skipped = 0

foreach ($gamma in $gamma_values) {
    $output_name = "multilayer_results_gamma_critical_$gamma.json"
    $output_path = Join-Path $ScriptDir $output_name
    
    if (Test-Path $output_path) {
        Write-Host "[SKIP] gamma=$gamma — output already exists"
        $skipped++
        continue
    }
    
    $ratio = [math]::Round($gamma / 0.4441, 2)
    Write-Host "[RUN] gamma=$gamma (ratio=$ratio x gamma_c) ..."
    
    $env:SIM_GAMMA = "$gamma"
    $env:SIM_BETA = "$beta"
    $env:SIM_N_SATS = "$n_sats"
    $env:SIM_GRID_RES = "$grid_res"
    $env:SIM_DURATION = "$duration"
    $env:SIM_OUTPUT_PHI = "0"
    
    $proc = Start-Process -FilePath (Join-Path $ScriptDir "multilayer_sim_real.exe") `
        -WorkingDirectory $ScriptDir -Wait -PassThru -NoNewWindow
    
    $default_output = Join-Path $ScriptDir "multilayer_results_real.json"
    if (Test-Path $default_output) {
        Move-Item $default_output $output_path -Force
        Write-Host "  [OK] gamma=$gamma done, exit=$($proc.ExitCode)"
        $completed++
    } else {
        $msg = "  [FAIL] gamma=" + $gamma + " : no output file generated"
        Write-Host $msg
        $failed++
    }
}

Write-Host ""
Write-Host "================================================"
Write-Host "P0 gamma_critical scan COMPLETE"
Write-Host "Completed: $completed, Failed: $failed, Skipped: $skipped"
Write-Host "================================================"