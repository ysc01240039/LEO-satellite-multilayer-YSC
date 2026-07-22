# =============================================================================
# C++ Parameter Scanning Master Plan (Round 30)
# =============================================================================
# PURPOSE: Systematically scan PDE parameters to validate theoretical predictions
#          and improve project health from 84% to >90% (IF>10 target).
#
# PRIORITY LEGEND:
#   P0 = CRITICAL: Required for C1/C2 validation, directly impacts IF>10 accept
#   P1 = HIGH: Required for generalizability, strengthens core conclusions
#   P2 = MEDIUM: Supplementary validation, improves robustness
#   P3 = LOW: Nice-to-have, minor improvements
#
# RUNTIME ESTIMATES (based on 40^3 grid, N=1000, 2h C++ simulation):
#   - Single gamma run: ~2.0 hours (40^3 grid, 7200 time units)
#   - Single N run: ~2.0 hours (scales with N)
#   - Single beta run: ~2.0 hours
#   - Grid convergence run: ~0.5-8.0 hours (scales with grid^3)
#
# TOTAL ESTIMATED RUNTIME (all P0+P1): ~38 hours
# TOTAL ESTIMATED RUNTIME (all scans): ~70 hours
# =============================================================================

param(
    [ValidateSet("P0","P1","P2","P3","all")]
    [string]$Priority = "P0",
    
    [ValidateSet("gamma_critical","n_scan","beta_scan","gamma_scan","grid_conv","all")]
    [string]$Scan = "all",
    
    [int]$DurationHours = 2,
    [switch]$DryRun,
    [switch]$Resume
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ExePath = Join-Path $ScriptDir "multilayer_sim_real.exe"
$DefaultBeta = 0.6
$DefaultN = 1000
$DefaultGrid = 40
$DefaultDuration = $DurationHours

# Verify executable exists
if (-not $DryRun -and -not (Test-Path $ExePath)) {
    Write-Error "Executable not found: $ExePath"
    Write-Host "Please compile main.cpp first: g++ -O3 main.cpp -o multilayer_sim_real.exe"
    exit 1
}

# =============================================================================
# SCAN DEFINITIONS
# =============================================================================

$Scans = @{
    # P0: CRITICAL — gamma near gamma_c to validate perturbation theory (C1/C2)
    "gamma_critical" = @{
        Priority = "P0"
        Description = "Gamma near critical point (gamma_c=0.444) to validate perturbation theory"
        Impact = "Validates C1 (perturbation theory at epsilon<<1) and C2 (amplitude equation)"
        HealthImprovement = "物理合理性 +10%, 理论自洽性 +5%"
        Parameters = @(
            @{gamma=0.40; beta=0.6; N=1000; grid=40; label="0.40_below_critical"},
            @{gamma=0.43; beta=0.6; N=1000; grid=40; label="0.43_near_critical"},
            @{gamma=0.445; beta=0.6; N=1000; grid=40; label="0.445_at_critical"},
            @{gamma=0.46; beta=0.6; N=1000; grid=40; label="0.46_above_critical"},
            @{gamma=0.50; beta=0.6; N=1000; grid=40; label="0.50_1.13x_critical"},
            @{gamma=0.60; beta=0.6; N=1000; grid=40; label="0.60_1.35x_critical"},
            @{gamma=0.80; beta=0.6; N=1000; grid=40; label="0.80_1.80x_critical"},
            @{gamma=1.00; beta=0.6; N=1000; grid=40; label="1.00_2.25x_critical"}
        )
        EstimatedRuntime = "16 hours (8 runs × 2h each)"
    }
    
    # P0: CRITICAL — N scan to determine true n_cores vs N scaling
    "n_scan" = @{
        Priority = "P0"
        Description = "Satellite count N scan to determine n_cores vs N scaling"
        Impact = "Validates generalizability, determines alpha_N exponent"
        HealthImprovement = "泛用性 +15%, 数据完整性 +5%"
        Parameters = @(
            @{gamma=6.0; beta=0.6; N=200; grid=40; label="N200"},
            @{gamma=6.0; beta=0.6; N=400; grid=40; label="N400"},
            @{gamma=6.0; beta=0.6; N=600; grid=40; label="N600"},
            @{gamma=6.0; beta=0.6; N=800; grid=40; label="N800"},
            @{gamma=6.0; beta=0.6; N=1500; grid=40; label="N1500"},
            @{gamma=6.0; beta=0.6; N=2000; grid=40; label="N2000"}
        )
        EstimatedRuntime = "12 hours (6 runs × 2h each)"
    }
    
    # P1: HIGH — beta scan to verify weak beta dependence
    "beta_scan" = @{
        Priority = "P1"
        Description = "Beta scan to verify weak beta dependence of n_cores"
        Impact = "Validates beta independence, strengthens phase diagram"
        HealthImprovement = "泛用性 +5%, 理论自洽性 +3%"
        Parameters = @(
            @{gamma=6.0; beta=0.1; N=1000; grid=40; label="beta_0.1"},
            @{gamma=6.0; beta=0.6; N=1000; grid=40; label="beta_0.6"},
            @{gamma=6.0; beta=1.0; N=1000; grid=40; label="beta_1.0"},
            @{gamma=6.0; beta=2.0; N=1000; grid=40; label="beta_2.0"}
        )
        EstimatedRuntime = "8 hours (4 runs × 2h each)"
    }
    
    # P1: HIGH — gamma scan for saturation model falsification
    "gamma_scan" = @{
        Priority = "P1"
        Description = "Broad gamma scan for saturation model validation"
        Impact = "Strengthens saturation model falsification with more data points"
        HealthImprovement = "数据完整性 +5%, 物理合理性 +3%"
        Parameters = @(
            @{gamma=0.5; beta=0.6; N=1000; grid=40; label="gamma_0.5"},
            @{gamma=1.0; beta=0.6; N=1000; grid=40; label="gamma_1.0"},
            @{gamma=3.0; beta=0.6; N=1000; grid=40; label="gamma_3.0"},
            @{gamma=6.0; beta=0.6; N=1000; grid=40; label="gamma_6.0"},
            @{gamma=10.0; beta=0.6; N=1000; grid=40; label="gamma_10.0"}
        )
        EstimatedRuntime = "10 hours (5 runs × 2h each)"
    }
    
    # P2: MEDIUM — grid convergence
    "grid_conv" = @{
        Priority = "P2"
        Description = "Grid resolution convergence analysis"
        Impact = "Validates numerical convergence, ensures grid independence"
        HealthImprovement = "数据完整性 +3%"
        Parameters = @(
            @{gamma=6.0; beta=0.6; N=1000; grid=20; label="grid20"},
            @{gamma=6.0; beta=0.6; N=1000; grid=30; label="grid30"},
            @{gamma=6.0; beta=0.6; N=1000; grid=40; label="grid40"},
            @{gamma=6.0; beta=0.6; N=1000; grid=50; label="grid50"}
        )
        EstimatedRuntime = "~10 hours (grid^3 scaling: 20³=0.25h, 30³=0.8h, 40³=2h, 50³=5h)"
    }
}

# =============================================================================
# FILTER SCANS BY PRIORITY
# =============================================================================

$SelectedScans = @()
foreach ($scanName in $Scans.Keys) {
    $scan = $Scans[$scanName]
    if ($Priority -eq "all" -or $scan.Priority -eq $Priority) {
        if ($Scan -eq "all" -or $Scan -eq $scanName) {
            $SelectedScans += @{Name=$scanName; Config=$scan}
        }
    }
}

if ($SelectedScans.Count -eq 0) {
    Write-Host "No scans selected. Priority=$Priority, Scan=$Scan"
    Write-Host "Available scans:"
    foreach ($scanName in $Scans.Keys) {
        $scan = $Scans[$scanName]
        Write-Host "  [$($scan.Priority)] $scanName — $($scan.Description)"
        Write-Host "         Runtime: $($scan.EstimatedRuntime)"
        Write-Host "         Impact: $($scan.Impact)"
    }
    exit 0
}

# =============================================================================
# DISPLAY PLAN
# =============================================================================

Write-Host "=============================================================================="
Write-Host "C++ PARAMETER SCANNING PLAN (Round 30)"
Write-Host "=============================================================================="
Write-Host "Priority filter: $Priority"
Write-Host "Scan filter: $Scan"
Write-Host "Duration per run: $DurationHours hours"
Write-Host ""

$TotalRuns = 0
foreach ($scan in $SelectedScans) {
    $TotalRuns += $scan.Config.Parameters.Count
}

Write-Host "Selected scans: $($SelectedScans.Count) scan groups, $TotalRuns total runs"
Write-Host ""

foreach ($scan in $SelectedScans) {
    $cfg = $scan.Config
    Write-Host "[$($cfg.Priority)] $($scan.Name)"
    Write-Host "  Description: $($cfg.Description)"
    Write-Host "  Impact: $($cfg.Impact)"
    Write-Host "  Health: $($cfg.HealthImprovement)"
    Write-Host "  Runtime: $($cfg.EstimatedRuntime)"
    Write-Host "  Runs: $($cfg.Parameters.Count)"
    foreach ($p in $cfg.Parameters) {
        Write-Host "    gamma=$($p.gamma), beta=$($p.beta), N=$($p.N), grid=$($p.grid) [$($p.label)]"
    }
    Write-Host ""
}

if ($DryRun) {
    Write-Host "DRY RUN — no simulations executed. Remove -DryRun to run."
    exit 0
}

# =============================================================================
# HEALTH IMPACT SUMMARY
# =============================================================================

Write-Host "=============================================================================="
Write-Host "EXPECTED HEALTH IMPACT"
Write-Host "=============================================================================="
Write-Host "Current: 84%"
Write-Host ""
Write-Host "After P0 scans (gamma_critical + n_scan):"
Write-Host "  物理合理性: 63% -> 73% (+10%)"
Write-Host "  泛用性: 38% -> 53% (+15%)"
Write-Host "  数据完整性: 88% -> 93% (+5%)"
Write-Host "  理论自洽性: 92% -> 97% (+5%)"
Write-Host "  Expected total: 84% -> 89%"
Write-Host ""
Write-Host "After P0+P1 scans (all above + beta_scan + gamma_scan):"
Write-Host "  物理合理性: 73% -> 76% (+3%)"
Write-Host "  泛用性: 53% -> 58% (+5%)"
Write-Host "  数据完整性: 93% -> 98% (+5%)"
Write-Host "  理论自洽性: 97% -> 100% (+3%)"
Write-Host "  Expected total: 89% -> 93%"
Write-Host ""

# =============================================================================
# EXECUTE SCANS
# =============================================================================

Write-Host "=============================================================================="
Write-Host "STARTING SCANS"
Write-Host "=============================================================================="

$StartTime = Get-Date
$CompletedRuns = 0
$FailedRuns = 0

foreach ($scan in $SelectedScans) {
    $cfg = $scan.Config
    Write-Host ""
    Write-Host "--- Scan: $($scan.Name) [$($cfg.Priority)] ---"
    Write-Host "--- $($cfg.Description) ---"
    
    $ScanStartTime = Get-Date
    
    foreach ($p in $cfg.Parameters) {
        $label = $p.label
        $gamma = $p.gamma
        $beta = $p.beta
        $N = $p.N
        $grid = $p.grid
        
        $outputName = "multilayer_results_$($scan.Name)_$label.json"
        
        # Skip if output exists and Resume mode
        if ($Resume -and (Test-Path (Join-Path $ScriptDir $outputName))) {
            Write-Host "  [SKIP] $label — output already exists"
            continue
        }
        
        Write-Host "  [RUN] $label (gamma=$gamma, beta=$beta, N=$N, grid=$grid)"
        Write-Host "        Output: $outputName"
        
        # Set environment variables
        $env:SIM_GAMMA = $gamma
        $env:SIM_BETA = $beta
        $env:SIM_N_SATS = $N
        $env:SIM_GRID_RES = $grid
        $env:SIM_DURATION = $DefaultDuration
        $env:SIM_OUTPUT_PHI = 0
        
        # Run simulation
        $RunStart = Get-Date
        try {
            $psi = New-Object System.Diagnostics.ProcessStartInfo
            $psi.FileName = $ExePath
            $psi.WorkingDirectory = $ScriptDir
            $psi.UseShellExecute = $false
            $psi.RedirectStandardOutput = $true
            $psi.RedirectStandardError = $true
            
            $proc = [System.Diagnostics.Process]::Start($psi)
            $stdout = $proc.StandardOutput.ReadToEnd()
            $stderr = $proc.StandardError.ReadToEnd()
            $proc.WaitForExit()
            
            $RunEnd = Get-Date
            $RunDuration = ($RunEnd - $RunStart).TotalMinutes
            
            if ($proc.ExitCode -eq 0) {
                # Rename output
                $defaultOutput = Join-Path $ScriptDir "multilayer_results_real.json"
                if (Test-Path $defaultOutput) {
                    $targetPath = Join-Path $ScriptDir $outputName
                    Move-Item $defaultOutput $targetPath -Force
                    Write-Host "        [OK] Completed in $($RunDuration.ToString('F1')) min"
                    $CompletedRuns++
                } else {
                    Write-Host "        [WARN] Exit code 0 but no output file found"
                    $FailedRuns++
                }
            } else {
                Write-Host "        [FAIL] Exit code $($proc.ExitCode)"
                if ($stderr) { Write-Host "        Error: $stderr" }
                $FailedRuns++
            }
        } catch {
            Write-Host "        [ERROR] $_"
            $FailedRuns++
        }
        
        # Progress
        $TotalDone = $CompletedRuns + $FailedRuns
        $Elapsed = (Get-Date) - $StartTime
        Write-Host "        Progress: $TotalDone/$TotalRuns runs ($CompletedRuns ok, $FailedRuns failed)"
        Write-Host "        Elapsed: $($Elapsed.ToString('hh\:mm\:ss'))"
    }
    
    $ScanEndTime = Get-Date
    $ScanDuration = ($ScanEndTime - $ScanStartTime).TotalMinutes
    Write-Host "  Scan $($scan.Name) complete: $($ScanDuration.ToString('F1')) min"
}

# =============================================================================
# SUMMARY
# =============================================================================

$EndTime = Get-Date
$TotalDuration = ($EndTime - $StartTime).TotalHours

Write-Host ""
Write-Host "=============================================================================="
Write-Host "SCAN PLAN COMPLETE"
Write-Host "=============================================================================="
Write-Host "Total runs: $TotalRuns"
Write-Host "Completed: $CompletedRuns"
Write-Host "Failed: $FailedRuns"
Write-Host "Total duration: $($TotalDuration.ToString('F1')) hours"
Write-Host ""

if ($FailedRuns -gt 0) {
    Write-Host "WARNING: $FailedRuns runs failed. Check output above for details."
    Write-Host "Re-run with -Resume to skip completed runs and retry failed ones."
}

Write-Host "Output files are in: $ScriptDir"
Write-Host "  Pattern: multilayer_results_<scan>_<label>.json"
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Run analysis: python dim_empirical_findings.py"
Write-Host "  2. Update figures: python paper/generate_figures.py"
Write-Host "  3. Update manuscript: paper/YSC_CN.tex, paper/YSC_EN.tex"
Write-Host "  4. Update WORKFLOW.md with new results"