@echo off
set SIM_GAMMA=2.0
set SIM_BETA=0.6
set SIM_N_SATS=1000
set SIM_DT=0.004
set SIM_GS_ITERS=5
set SIM_PHI_MAX=1e10
set SIM_KAPPA=0.01
set SIM_DURATION=2.0
set SIM_SEED=42
set SIM_OUTPUT=e:\pytorchFile\YSC_2\Project\Project\scan_output\long_g2p0.json
"e:\pytorchFile\YSC_2\Project\Project\multilayer_sim_real.exe" >> "e:\pytorchFile\YSC_2\Project\Project\scan_output\long_g2p0.log" 2>&1
