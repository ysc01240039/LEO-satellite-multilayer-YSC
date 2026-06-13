call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
cl /O2 /openmp /std:c++17 main.cpp /Fe:multilayer_sim_real.exe
