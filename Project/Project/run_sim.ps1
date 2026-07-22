$env:SIM_GAMMA = "6.0"
$env:SIM_BETA = "0.6"
$env:SIM_N_SATS = "1000"
$env:SIM_DURATION = "2.0"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = Join-Path $ScriptDir "multilayer_sim_real.exe"
$psi.WorkingDirectory = $ScriptDir
$psi.UseShellExecute = $false
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError = $true
$proc = [System.Diagnostics.Process]::Start($psi)
$stdout = $proc.StandardOutput.ReadToEnd()
$stderr = $proc.StandardError.ReadToEnd()
$proc.WaitForExit()
Write-Host "Exit code: $($proc.ExitCode)"
if ($stdout) { Write-Host "STDOUT: $stdout" }
if ($stderr) { Write-Host "STDERR: $stderr" }
