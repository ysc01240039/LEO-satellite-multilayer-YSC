$vsPath = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat"
$compileCmd = "cl /O2 /openmp /std:c++17 main.cpp /Fe:multilayer_sim_real.exe"
cmd /c "`"$vsPath`" x64 > nul && $compileCmd" 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "Compilation SUCCESS"
} else {
    # Try VS Build Tools path
    $vsPath2 = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"
    cmd /c "`"$vsPath2`" x64 > nul && $compileCmd" 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Compilation SUCCESS (BuildTools)"
    } else {
        Write-Host "Compilation FAILED"
    }
}