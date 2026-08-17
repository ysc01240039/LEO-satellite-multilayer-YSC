// main.cpp
// Compile: g++ -O3 -fopenmp -std=c++17 main.cpp -o multilayer_sim_real
// Run: ./multilayer_sim_real

#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <omp.h>
#include <algorithm>
#include <cstring>

#include <string>
#include <array>
#include <numeric>
#include <tuple>
#include <cstdlib>

// ==================== Parameters ====================
struct Params {
    double L_ref = 1000.0;
    double T_ref = 1.0;
    double sigma_km = 1000.0;
    double r_rep_km = 800.0;
    double gamma = 6.0;
    double beta = 0.6;
    double mu_prime = 10.0;
    double nu_prime = 1.0;
    double R_max_km = 5000.0;
    double dt = 0.004;  // Reduced to satisfy H-theorem condition dt < 2/L ≈ 0.0048 (L ≈ 419)
    double duration_hours = 2.0;
    double D = 1.0;       // diffusion coefficient (D=1 after non-dimensionalization t → Dt/L²)
    int grid_res = 40;
    int n_sats = 1000;
    int n_gs_iters = 5;    // number of Jacobi iterations for implicit diffusion solve (IMEX scheme)
    double phi_max = 1e10; // saturation ceiling (set very high; nonlinear damping via kappa is primary stabilization)
    double kappa = 0.01;   // nonlinear damping coefficient (κφ² term prevents blow-up at high φ while preserving dynamics at low φ)
    int rho_update_interval = 20;
    int payload_update_interval = 10;
    int save_interval = 1800;
    std::string orbit_dir = "orbit_bin";
};

// ==================== Environment variable parsing ====================
double getEnvDouble(const char* name, double default_val) {
    const char* val = std::getenv(name);
    if (val) {
        return std::atof(val);
    }
    return default_val;
}

int getEnvInt(const char* name, int default_val) {
    const char* val = std::getenv(name);
    if (val) {
        return std::atoi(val);
    }
    return default_val;
}

void loadParamsFromEnv(Params& p) {
    p.gamma = getEnvDouble("SIM_GAMMA", p.gamma);
    p.beta = getEnvDouble("SIM_BETA", p.beta);
    p.duration_hours = getEnvDouble("SIM_DURATION", p.duration_hours);
    p.dt = getEnvDouble("SIM_DT", p.dt);
    p.n_gs_iters = getEnvInt("SIM_GS_ITERS", p.n_gs_iters);
    p.phi_max = getEnvDouble("SIM_PHI_MAX", p.phi_max);
    p.kappa = getEnvDouble("SIM_KAPPA", p.kappa);
    p.n_sats = getEnvInt("SIM_N_SATS", p.n_sats);
    p.grid_res = getEnvInt("SIM_GRID_RES", p.grid_res);
}

int getEnvBool(const char* name, int default_val) {
    const char* val = std::getenv(name);
    if (val) return std::atoi(val);
    return default_val;
}

// ==================== Data structures ====================
struct OrbitData {
    std::vector<std::array<double, 3>> positions;
};

struct Satellite {
    double pos[3];
    double beam[3];
    double queue;
    double cap;
    bool task;
    int core_id;
};

// ==================== Helper functions ====================
bool load_orbit(int sat_id, const std::string& orbit_dir, OrbitData& data) {
    std::string filename = orbit_dir + "/sat_" + std::to_string(sat_id) + ".bin";
    std::ifstream f(filename, std::ios::binary);
    if (!f) {
        std::cerr << "Failed to open " << filename << std::endl;
        return false;
    }
    int n_points;
    f.read(reinterpret_cast<char*>(&n_points), sizeof(int));
    data.positions.resize(n_points);
    for (int i = 0; i < n_points; ++i) {
        f.read(reinterpret_cast<char*>(data.positions[i].data()), 3 * sizeof(double));
    }
    return true;
}

double gaussian(double r, double sigma) {
    return std::exp(-r*r / (2*sigma*sigma));
}

void update_payloads(std::vector<Satellite>& sats, double mu_prime, double nu_prime, std::mt19937& rng) {
    std::uniform_real_distribution<double> dist01(0.0, 1.0);
    for (auto& sat : sats) {
        double mu = mu_prime * (1.0 - sat.cap);
        double nu = nu_prime * sat.cap;
        double d_cap = mu * dist01(rng) - nu * dist01(rng);
        sat.cap = std::max(0.0, std::min(1.0, sat.cap + d_cap));
        sat.task = dist01(rng) > 0.5;
    }
}

// ==================== Main function ====================
int main() {
    Params p;
    loadParamsFromEnv(p);
    
    int uniform_source = getEnvBool("SIM_UNIFORM_SOURCE", 0);
    int output_phi = getEnvBool("SIM_OUTPUT_PHI", 0);

    std::cout << "Parameters: gamma=" << p.gamma << ", beta=" << p.beta
              << ", n_sats=" << p.n_sats << ", duration=" << p.duration_hours << "h"
              << ", dt=" << p.dt << ", gs_iters=" << p.n_gs_iters
              << ", phi_max=" << p.phi_max << ", kappa=" << p.kappa
              << ", uniform=" << uniform_source << std::endl;

    std::random_device rd;
    // Use SIM_SEED env var for reproducible/different seeds; 
    // on MinGW/Windows, std::random_device is deterministic, so SIM_SEED is essential
    int seed = getEnvInt("SIM_SEED", -1);
    std::mt19937 rng(seed >= 0 ? static_cast<unsigned>(seed) : rd());
    std::uniform_real_distribution<double> dist01(0.0, 1.0);

    double sigma = p.sigma_km / p.L_ref;
    double r_rep = p.r_rep_km / p.L_ref;
    double R_max = p.R_max_km / p.L_ref;

    int res = p.grid_res;
    double grid_size = 10.0;
    double dx = 2 * grid_size / res;

    std::vector<double> grid_x(res), grid_y(res), grid_z(res);
    for (int i = 0; i < res; ++i) {
        grid_x[i] = -grid_size + i * dx;
        grid_y[i] = -grid_size + i * dx;
        grid_z[i] = -grid_size + i * dx;
    }

    // Load orbit data
    std::vector<OrbitData> orbits(p.n_sats);
    for (int i = 0; i < p.n_sats; ++i) {
        int sat_id = i + 1;
        if (!load_orbit(sat_id, p.orbit_dir, orbits[i])) {
            std::cerr << "Failed to load orbit for sat " << sat_id << std::endl;
            return 1;
        }
    }

    // Initialize satellites
    std::vector<Satellite> sats(p.n_sats);
    for (int i = 0; i < p.n_sats; ++i) {
        sats[i].pos[0] = orbits[i].positions[0][0] / p.L_ref;
        sats[i].pos[1] = orbits[i].positions[0][1] / p.L_ref;
        sats[i].pos[2] = orbits[i].positions[0][2] / p.L_ref;
        double norm = std::sqrt(sats[i].pos[0] * sats[i].pos[0] + sats[i].pos[1] * sats[i].pos[1] + sats[i].pos[2] * sats[i].pos[2]);
        if (norm > 0) {
            sats[i].beam[0] = sats[i].pos[0] / norm;
            sats[i].beam[1] = sats[i].pos[1] / norm;
            sats[i].beam[2] = sats[i].pos[2] / norm;
        } else {
            sats[i].beam[0] = 1.0; sats[i].beam[1] = 0.0; sats[i].beam[2] = 0.0;
        }
        sats[i].queue = dist01(rng) * 100.0;
        sats[i].cap = dist01(rng);
        sats[i].task = dist01(rng) > 0.5;
        sats[i].core_id = -1;
    }

    update_payloads(sats, p.mu_prime, p.nu_prime, rng);

    // Main simulation loop
    double t = 0.0;
    int step = 0;
    double duration = p.duration_hours * 3600.0 / p.T_ref;

    std::vector<double> times;
    std::vector<int> n_cores_hist, n_links_hist, isolated_hist, order_hist;
    std::vector<std::vector<double>> final_cores_x, final_cores_y, final_cores_z, final_cores_peak;
    std::vector<double> phi_final;  // preserved for post-loop output

    // === phi field MUST persist across timesteps for PDE evolution ===
    // Previously declared inside the while loop (line 221), causing phi to be
    // re-initialized to zero every iteration. This meant the KS dynamics was
    // never solved — phi was always just dt*rho. Fixed 2026-07-25.
    std::vector<double> phi(res*res*res, 0.0);

    // Initialize phi perturbation for uniform source (once, before loop)
    if (uniform_source) {
        #pragma omp parallel for
        for (int idx = 0; idx < res*res*res; ++idx) {
            int x = idx / (res * res);
            int y = (idx / res) % res;
            int z = idx % res;
            phi[idx] = 0.01 * std::sin(0.5 * (x + 2*y + 3*z));
        }
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    while (t < duration) {
        int orbit_index = static_cast<int>(t * 1000);
        for (int i = 0; i < p.n_sats; ++i) {
            int orbit_size = static_cast<int>(orbits[i].positions.size());
            int idx = orbit_index % orbit_size;
            sats[i].pos[0] = orbits[i].positions[idx][0] / p.L_ref;
            sats[i].pos[1] = orbits[i].positions[idx][1] / p.L_ref;
            sats[i].pos[2] = orbits[i].positions[idx][2] / p.L_ref;
            double norm = std::sqrt(sats[i].pos[0] * sats[i].pos[0] + sats[i].pos[1] * sats[i].pos[1] + sats[i].pos[2] * sats[i].pos[2]);
            if (norm > 0) {
                sats[i].beam[0] = sats[i].pos[0] / norm;
                sats[i].beam[1] = sats[i].pos[1] / norm;
                sats[i].beam[2] = sats[i].pos[2] / norm;
            }
        }

        if (step % p.payload_update_interval == 0) {
            update_payloads(sats, p.mu_prime, p.nu_prime, rng);
        }

        std::vector<double> rho(res*res*res, 0.0);

        if (uniform_source) {
            // Uniform source — phi perturbation initialized once before loop
            double uniform_rho = 2.0;
            #pragma omp parallel for
            for (int idx = 0; idx < res*res*res; ++idx) {
                rho[idx] = uniform_rho;
            }
        } else {
            for (const auto& sat : sats) {
                int x = static_cast<int>((sat.pos[0] + grid_size) / dx);
                int y = static_cast<int>((sat.pos[1] + grid_size) / dx);
                int z = static_cast<int>((sat.pos[2] + grid_size) / dx);
                if (x >= 0 && x < res && y >= 0 && y < res && z >= 0 && z < res) {
                    int idx = x * res * res + y * res + z;
                    rho[idx] += 1.0 + sat.cap + (sat.task ? 1.0 : 0.0);  // +1.0 = base contribution per satellite (background load)
                }
            }
        }

        // ===== IMEX Scheme: Explicit chemotaxis + Implicit diffusion (Jacobi) =====
        // Step 1: Compute chemotaxis term for all grid points (explicit, using current phi)
        std::vector<double> chem_arr(res*res*res, 0.0);
        #pragma omp parallel for
        for (int idx = 0; idx < res*res*res; ++idx) {
            int x = idx / (res * res);
            int y = (idx / res) % res;
            int z = idx % res;

            double chem = 0.0;
            for (int xx = -1; xx <= 1; ++xx) {
                for (int yy = -1; yy <= 1; ++yy) {
                    for (int zz = -1; zz <= 1; ++zz) {
                        if (xx == 0 && yy == 0 && zz == 0) continue;
                        int nx = x + xx;
                        int ny = y + yy;
                        int nz = z + zz;
                        if (nx < 0) nx = -nx; else if (nx >= res) nx = 2*res - 2 - nx;
                        if (ny < 0) ny = -ny; else if (ny >= res) ny = 2*res - 2 - ny;
                        if (nz < 0) nz = -nz; else if (nz >= res) nz = 2*res - 2 - nz;
                        nx = std::max(0, std::min(res-1, nx));
                        ny = std::max(0, std::min(res-1, ny));
                        nz = std::max(0, std::min(res-1, nz));
                        int nidx = nx * res * res + ny * res + nz;
                        double dr = std::sqrt(static_cast<double>(xx*xx + yy*yy + zz*zz)) * dx;
                        chem += (phi[nidx] - phi[idx]) * gaussian(dr, sigma) / dr;
                    }
                }
            }
            chem_arr[idx] = chem;
        }

        // Step 2: Explicit update (chemotaxis + nonlinear damping + source)
        std::vector<double> phi_explicit(res*res*res);
        #pragma omp parallel for
        for (int idx = 0; idx < res*res*res; ++idx) {
            double phi_star = phi[idx] + p.dt * (
                - p.gamma * chem_arr[idx] * std::max(0.0, 1.0 - phi[idx] / p.phi_max)
                - p.kappa * phi[idx] * phi[idx]
                + rho[idx]
            );
            phi_explicit[idx] = std::max(0.0, phi_star);
        }

        // Step 3: Implicit diffusion + linear decay (Jacobi iteration)
        // Solve: (I - dt*D*∇² + dt*β) φ_new = φ_explicit
        double diag = 1.0 + p.dt * p.beta + 6.0 * p.dt * p.D / (dx * dx);
        double off_diag = p.dt * p.D / (dx * dx);

        std::vector<double> phi_old = phi_explicit;
        std::vector<double> phi_new_jacobi(res*res*res);

        for (int iter = 0; iter < p.n_gs_iters; ++iter) {
            #pragma omp parallel for
            for (int idx = 0; idx < res*res*res; ++idx) {
                int x = idx / (res * res);
                int y = (idx / res) % res;
                int z = idx % res;

                // Neumann (zero-flux) boundary conditions
                int xp = (x + 1 < res) ? x + 1 : (x - 1);
                int xm = (x - 1 >= 0) ? x - 1 : (x + 1);
                int yp = (y + 1 < res) ? y + 1 : (y - 1);
                int ym = (y - 1 >= 0) ? y - 1 : (y + 1);
                int zp = (z + 1 < res) ? z + 1 : (z - 1);
                int zm = (z - 1 >= 0) ? z - 1 : (z + 1);

                double neighbor_sum = phi_old[xp * res * res + y * res + z]
                                    + phi_old[xm * res * res + y * res + z]
                                    + phi_old[x * res * res + yp * res + z]
                                    + phi_old[x * res * res + ym * res + z]
                                    + phi_old[x * res * res + y * res + zp]
                                    + phi_old[x * res * res + y * res + zm];

                phi_new_jacobi[idx] = (phi_explicit[idx] + off_diag * neighbor_sum) / diag;
            }
            phi_old.swap(phi_new_jacobi);
        }
        phi.swap(phi_old);  // final result after Jacobi iterations

        // Enforce positivity after Jacobi solve (not inside iterations, to avoid
        // nonlinear modification of the linear solver)
        #pragma omp parallel for
        for (int idx = 0; idx < res*res*res; ++idx) {
            if (phi[idx] < 0.0) phi[idx] = 0.0;
        }

        // ===== Analysis: only compute when saving (was ~99.9% wasted computation) =====
        if (step % p.save_interval == 0) {
        std::vector<int> assignments(res*res*res, -1);
        int core_id = 0;
        double max_phi = *std::max_element(phi.begin(), phi.end());
        double threshold = 0.1 * max_phi;

        std::vector<double> core_phi;
        std::vector<int> core_count;
        std::vector<double> core_x_sum, core_y_sum, core_z_sum;
        std::vector<double> core_phi_sum;

        for (int x = 0; x < res; ++x) {
            for (int y = 0; y < res; ++y) {
                for (int z = 0; z < res; ++z) {
                    int idx = x * res * res + y * res + z;
                    if (phi[idx] > threshold && assignments[idx] == -1) {
                        std::vector<std::tuple<int, int, int>> stack;
                        stack.emplace_back(x, y, z);
                        assignments[idx] = core_id;
                        core_phi.push_back(phi[idx]);
                        core_count.push_back(1);
                        core_x_sum.push_back((x * dx - grid_size) * phi[idx]);
                        core_y_sum.push_back((y * dx - grid_size) * phi[idx]);
                        core_z_sum.push_back((z * dx - grid_size) * phi[idx]);
                        core_phi_sum.push_back(phi[idx]);
                        while (!stack.empty()) {
                            auto top = stack.back();
                            stack.pop_back();
                            int cx = std::get<0>(top);
                            int cy = std::get<1>(top);
                            int cz = std::get<2>(top);
                            for (int ddx = -1; ddx <= 1; ++ddx) {
                                for (int ddy = -1; ddy <= 1; ++ddy) {
                                    for (int ddz = -1; ddz <= 1; ++ddz) {
                                        int nx = cx + ddx, ny = cy + ddy, nz = cz + ddz;
                                        if (nx >= 0 && nx < res && ny >= 0 && ny < res && nz >= 0 && nz < res) {
                                            int nidx = nx * res * res + ny * res + nz;
                                            if (phi[nidx] > threshold && assignments[nidx] == -1) {
                                                assignments[nidx] = core_id;
                                                stack.emplace_back(nx, ny, nz);
                                                core_phi[core_id] = std::max(core_phi[core_id], phi[nidx]);
                                                core_x_sum[core_id] += (nx * dx - grid_size) * phi[nidx];
                                                core_y_sum[core_id] += (ny * dx - grid_size) * phi[nidx];
                                                core_z_sum[core_id] += (nz * dx - grid_size) * phi[nidx];
                                                core_phi_sum[core_id] += phi[nidx];
                                                core_count[core_id]++;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        core_id++;
                    }
                }
            }
        }

        std::vector<double> core_pos_x, core_pos_y, core_pos_z, core_peak;
        for (int c = 0; c < core_id; ++c) {
            double cx = core_phi_sum[c] > 0 ? core_x_sum[c] / core_phi_sum[c] : 0.0;
            double cy = core_phi_sum[c] > 0 ? core_y_sum[c] / core_phi_sum[c] : 0.0;
            double cz = core_phi_sum[c] > 0 ? core_z_sum[c] / core_phi_sum[c] : 0.0;
            core_pos_x.push_back(cx);
            core_pos_y.push_back(cy);
            core_pos_z.push_back(cz);
            core_peak.push_back(core_phi[c]);
        }

        for (auto& sat : sats) {
            int x = static_cast<int>((sat.pos[0] + grid_size) / dx);
            int y = static_cast<int>((sat.pos[1] + grid_size) / dx);
            int z = static_cast<int>((sat.pos[2] + grid_size) / dx);
            if (x >= 0 && x < res && y >= 0 && y < res && z >= 0 && z < res) {
                int idx = x * res * res + y * res + z;
                sat.core_id = assignments[idx];
            } else {
                sat.core_id = -1;
            }
        }

        std::vector<std::vector<char>> adj(p.n_sats, std::vector<char>(p.n_sats, 0));
        for (int i = 0; i < p.n_sats; ++i) {
            for (int j = i+1; j < p.n_sats; ++j) {
                if (sats[i].core_id != -1 && sats[i].core_id == sats[j].core_id) {
                    double dx = sats[i].pos[0] - sats[j].pos[0];
                    double dy = sats[i].pos[1] - sats[j].pos[1];
                    double dz = sats[i].pos[2] - sats[j].pos[2];
                    double d2 = dx*dx + dy*dy + dz*dz;
                    if (d2 < R_max * R_max) {
                        adj[i][j] = adj[j][i] = 1;
                    }
                }
            }
        }

        double order = 0.0;
        int n_links = 0;
        int n_isolated = 0;
        for (int i = 0; i < p.n_sats; ++i) {
            int degree = 0;
            for (int j = i + 1; j < p.n_sats; ++j) {
                if (adj[i][j]) {
                    degree++;
                    n_links++;
                }
            }
            for (int j = 0; j < i; ++j) {
                if (adj[i][j]) {
                    degree++;
                }
            }
            if (degree == 0) {
                n_isolated++;
            }
            order += degree * degree;
        }
        order /= (p.n_sats * p.n_sats);

        times.push_back(t);
        n_cores_hist.push_back(core_id);
        n_links_hist.push_back(n_links);
        isolated_hist.push_back(n_isolated);
        order_hist.push_back(static_cast<int>(order * 1000));
        std::vector<double> step_core_x, step_core_y, step_core_z, step_core_peak;
        for (int c = 0; c < core_id; ++c) {
            double cx = core_phi_sum[c] > 0 ? core_x_sum[c] / core_phi_sum[c] : 0.0;
            double cy = core_phi_sum[c] > 0 ? core_y_sum[c] / core_phi_sum[c] : 0.0;
            double cz = core_phi_sum[c] > 0 ? core_z_sum[c] / core_phi_sum[c] : 0.0;
            step_core_x.push_back(cx);
            step_core_y.push_back(cy);
            step_core_z.push_back(cz);
            step_core_peak.push_back(core_phi[c]);
        }
        final_cores_x.push_back(step_core_x);
        final_cores_y.push_back(step_core_y);
        final_cores_z.push_back(step_core_z);
        final_cores_peak.push_back(step_core_peak);
        std::cout << "t=" << t << ", cores=" << core_id << ", links=" << n_links << ", isolated=" << n_isolated << ", order=" << order << std::endl;
        }  // end if (step % p.save_interval == 0)

        t += p.dt;
        step++;
    }

    phi_final = phi;  // preserve for post-loop Fourier output (only once, not every step)

    auto end_time = std::chrono::high_resolution_clock::now();
    double total_elapsed = std::chrono::duration<double>(end_time - start_time).count();
    std::cout << "Simulation finished in " << total_elapsed / 60 << " minutes." << std::endl;

    // Output phi field as binary for Fourier analysis
    if (output_phi && !phi_final.empty()) {
        std::ofstream phi_out("phi_field.bin", std::ios::binary);
        int n_cells = res * res * res;
        phi_out.write(reinterpret_cast<char*>(&res), sizeof(int));
        phi_out.write(reinterpret_cast<char*>(phi_final.data()), n_cells * sizeof(double));
        phi_out.close();
        std::cout << "Phi field saved: phi_field.bin (" << n_cells << " cells)" << std::endl;
    }

    const char* out_filename = std::getenv("SIM_OUTPUT");
    std::string out_name = out_filename ? out_filename : "multilayer_results_real.json";
    std::ofstream out(out_name);
    out << "{\n";
    out << "  \"gamma\": " << p.gamma << ",\n";
    out << "  \"beta\": " << p.beta << ",\n";
    out << "  \"n_sats\": " << p.n_sats << ",\n";
    out << "  \"seed\": " << seed << ",\n";
    out << "  \"kappa\": " << p.kappa << ",\n";
    out << "  \"avg_cores\": " << std::accumulate(n_cores_hist.begin(), n_cores_hist.end(), 0.0) / n_cores_hist.size() << ",\n";
    out << "  \"final_cores\": {\n";
    out << "    \"x\": [";
    for (size_t i = 0; i < final_cores_x.size(); ++i) {
        out << (i ? ",[" : "[");
        for (size_t j = 0; j < final_cores_x[i].size(); ++j) {
            out << (j ? "," : "") << final_cores_x[i][j];
        }
        out << "]";
    }
    out << "],\n    \"y\": [";
    for (size_t i = 0; i < final_cores_y.size(); ++i) {
        out << (i ? ",[" : "[");
        for (size_t j = 0; j < final_cores_y[i].size(); ++j) {
            out << (j ? "," : "") << final_cores_y[i][j];
        }
        out << "]";
    }
    out << "],\n    \"z\": [";
    for (size_t i = 0; i < final_cores_z.size(); ++i) {
        out << (i ? ",[" : "[");
        for (size_t j = 0; j < final_cores_z[i].size(); ++j) {
            out << (j ? "," : "") << final_cores_z[i][j];
        }
        out << "]";
    }
    out << "],\n    \"peak\": [";
    for (size_t i = 0; i < final_cores_peak.size(); ++i) {
        out << (i ? ",[" : "[");
        for (size_t j = 0; j < final_cores_peak[i].size(); ++j) {
            out << (j ? "," : "") << final_cores_peak[i][j];
        }
        out << "]";
    }
    out << "]\n  },\n";
    out << "  \"time_series\": {\n";
    out << "    \"t\": [";
    for (size_t i = 0; i < times.size(); ++i) out << (i ? "," : "") << times[i];
    out << "],\n    \"n_cores\": [";
    for (size_t i = 0; i < n_cores_hist.size(); ++i) out << (i ? "," : "") << n_cores_hist[i];
    out << "],\n    \"n_links\": [";
    for (size_t i = 0; i < n_links_hist.size(); ++i) out << (i ? "," : "") << n_links_hist[i];
    out << "],\n    \"isolated\": [";
    for (size_t i = 0; i < isolated_hist.size(); ++i) out << (i ? "," : "") << isolated_hist[i];
    out << "],\n    \"order\": [";
    for (size_t i = 0; i < order_hist.size(); ++i) out << (i ? "," : "") << order_hist[i];
    out << "]\n  }\n}\n";
    out.close();
    std::cout << "Results saved to " << out_name << std::endl;

    return 0;
}
