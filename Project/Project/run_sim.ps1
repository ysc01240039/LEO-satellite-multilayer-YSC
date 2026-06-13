$env:SIM_GAMMA = "5.5"
$env:SIM_BETA = "0.5"
$env:SIM_N_SATS = "1000"
$env:SIM_DURATION = "0.5"
$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = "E:\pytorchFile\YSC_2\Project\Project\multilayer_sim_real.exe"
$psi.WorkingDirectory = "E:\pytorchFile\YSC_2\Project\Project"
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
