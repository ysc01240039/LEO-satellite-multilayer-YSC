# LEO-satellite-multilayer-YSC

**Task-Aware Distributed Inter-Layer Topology Optimization Method in Resource-Limited LEO-LEO Satellite Networks**

This repository contains the simulation code, calibration pipeline, and raw data for a continuum-PDE-based, task-aware distributed topology control method for resource-limited LEO-LEO satellite networks. The core idea is to model satellite load distribution as a reaction–diffusion system (a Keller–Segel-type PDE) on the sphere, drive core-formation via a nonlocal diffusion–concentration balance, and route traffic through a self-organizing SNC (satellite network core) hierarchy with intra-core load spreading.

> The final paper text (LaTeX) is not distributed here; this repository hosts the **executable simulation code, the C++ calibration data, and the reproducible result reports** used to generate the paper's figures and tables.

---

## Overview

- **Continuum model.** The normalized load field satisfies a dimensionless reaction–diffusion equation with nonlocal diffusion, gradient-driven drift (chemotaxis), nonlinear damping, and a saturation ceiling. Buckingham Pi analysis reduces the parameter space to five dimensionless groups, of which the balance between the nonlocal diffusion rate `gamma` and the gradient gain `beta` controls pattern formation.
- **Core detection (SNC).** Cores are identified by a nonlocal-operator spectral analysis rather than by local intensity thresholds, which removes the circular dependency between core detection and the routing hierarchy.
- **CBDP (Core-Based Distributed Protocol).** A three-tier routing protocol: SNC core election, portal relay across the satellite network, and intra-core load spreading. Two variants are benchmarked (CBDP v2 = `benchmark_cbdp`, CBDP v3 = grid-searched alpha/k_cores).
- **SOTA comparison.** CBDP is compared against Dijkstra (centralized oracle), Nearest-3, PFNSAR, and LPIH baselines under 5 constellations (Iridium 66, Globalstar 48, Medium 500, Large 1000, Gen1 4408).

---

## Repository layout

```
Project/Project/main.cpp              C++ PDE simulator (IMEX time-stepping, OMP)
Project/Project/scan_output/          C++ scan data: gamma scan, kappa scan,
                                      N scan, long runs, multi-seed runs (JSON)
Project/Project_nscan/                N-scan (constellation size) results
common_utils.py                       Shared constants, network generator, all
                                      benchmark algorithms (dijkstra/greedy/
                                      nearest3/roundrobin/cbdp/pfnsar/lpih)
algorithm_v2.py                       CBDP design v2 with phase-diagram-optimized
                                      parameters (legacy entry point)
rerun_e2e_g1p0.py                     Multi-seed end-to-end benchmark at gamma=1.0
rerun_benchmark_g1p0.py               Access-layer benchmark at gamma=1.0
rerun_benchmark_v2.py                 Multi-seed re-run of algorithm_v2 Part E
fig5_ablation_v2_experiment.py        Ablation study (PDE SNC / dynamic reconfig /
                                      SNC mesh / intra-core spreading) -> Fig. 5
fig5_ablation_experiment.py           DEPRECATED v1 ablation (audit only)
dim5_phase_diagram.py                 Phase-diagram analysis (dimensionless group)
paper/ns3/                            ns-3 packet-level validation & comparison:
  leo_cbdp_eval.cc                    CBDP ns-3 implementation (routes, portals)
  leo_route_compare.cc                Comparison vs OLSR/AODV/Dijkstra
  parse_and_plot_eval.py              Parses logs, writes Fig. 6/8/9 + overview
  run_compare.sh / run_full_batch.sh  Batch evaluation drivers
  eval_results_summary.json           Summarized ns-3 results
paper/generate_fig4_real.py           Regenerates Fig. 4 (algorithm benchmark)
paper/generate_fig5_real.py           Regenerates Fig. 5 (ablation)
paper/figures/                        Final publication-quality figure PDFs
tools/                                Auxiliary tools (LaTeX checks, PDF annotation
                                      extraction, Lyapunov verification, ns-3 fixes)
*.json (root)                         Result reports used by the paper's tables
                                      and figures
```

> Note: the large binary `orbit_bin/` constellation data (~6 GB per copy, two
> copies on disk) exceeds GitHub's per-repo limits and is therefore not
> distributed here. The orbit generator and all downstream consumers read it
> deterministically; contact the authors if you need the raw orbit files.

---

## Dependencies

- Python 3.8+, `numpy`, `scipy`
- C++17 compiler with OpenMP (e.g. `g++ -O3 -fopenmp -std=c++17`)
- Matplotlib (only if you regenerate figures from the report JSONs)

Install Python dependencies:

```bash
pip install numpy scipy
```

---

## Reproduce the benchmark reports

All report JSONs are committed and can be reproduced from scratch as follows.

### 1. End-to-end scalability benchmark at gamma = 1.0 (paper benchmark / Fig. 4)

```bash
python rerun_e2e_g1p0.py
```

Output: `algorithm_v2_e2e_g1p0_report.json` — per-seed rows for each constellation and the mean/std summary across all methods.

### 2. Access-layer benchmark re-run

```bash
python rerun_benchmark_g1p0.py
python rerun_benchmark_v2.py
```

Output: `algorithm_v2_rerun_g1p0_report.json`, `algorithm_v2_rerun_report.json`.

### 3. Ablation study (paper Fig. 5)

```bash
python fig5_ablation_v2_experiment.py
```

Output: `fig5_ablation_v2_results.json` and `fig5_ablation_report.txt`.

### 4. C++ PDE simulator (calibration data)

```bash
cd Project/Project
g++ -O3 -fopenmp -std=c++17 main.cpp -o multilayer_sim_real
./multilayer_sim_real
```

The simulator writes scan outputs into `scan_output/` (gamma scan, kappa scan, N scan, long runs, and seed sweeps). The committed JSON files under `scan_output/` are the real calibration data used for model calibration and for validating the perturbation-theory prediction of the critical `gamma`.

### 5. ns-3 packet-level validation (paper Fig. 6/8/9 and comparison)

The ns-3 evaluation lives under `paper/ns3/`. Build and run the two simulations against the required ns-3 build, then regenerate the figure PDFs:

```bash
cd paper/ns3
# compile leo_cbdp_eval.cc / leo_route_compare.cc against your ns-3 tree
# run the batch driver (produces logs under logs_eval/)
bash run_full_batch.sh
# parse logs and write Fig. 6 / 8 / 9 / overview PDFs
python parse_and_plot_eval.py
```

### 6. Regenerate the algorithm benchmark and ablation figures

```bash
python paper/generate_fig4_real.py   # Fig. 4 from algorithm_v2_e2e_g1p0_report.json
python paper/generate_fig5_real.py   # Fig. 5 from fig5_ablation_v2_results.json
```

---

## Data provenance

- **C++ calibration data** (`Project/Project/scan_output/*.json`): real simulation outputs from `main.cpp`, run to steady state (beyond 1.25x the characteristic diffusion time) to ensure convergence. Used for model calibration, replacing all hardcoded baseline values.
- **Report JSONs (root)**: produced by the scripts above with the exact committed code, so every number in the paper's tables and figures is reproducible.
- All simulation data and code are available at
  <https://github.com/ysc01240039/LEO-satellite-multilayer-YSC>,
  to be permanently archived on Zenodo upon acceptance.

---

## Method summary

The normalized load field `phi` evolves under a dimensionless reaction–diffusion equation with nonlocal diffusion, gradient-driven drift, nonlinear damping, and a saturation ceiling. The four dimensionless groups (`Pi1..Pi4`) come from a Buckingham Pi analysis; after setting `gamma = gamma_tilde / D_tilde`, `beta = beta_tilde L^2 / D_tilde`, and `D = 1`, pattern formation is governed by the nonlocal-diffusion vs. gradient-balance parameter `gamma` and the cross-layer coupling `beta`. Cores are detected with an independent nonlocal-operator spectral method (removing circular dependencies), and the SNC hierarchy then drives portal-based relay and intra-core load spreading. See the paper for the full derivation.

---

## License

Research code for reproducibility. Please cite the paper upon use.
