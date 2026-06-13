"""
Fourier spectrum analysis of C++ phi field.
Detects Turing instability modes at k_c.
Usage: python analyze_fourier.py [phi_file.bin] [--compare]

With --compare: runs uniform source sim at multiple gamma to find phase transition.
"""

import struct, json, io, os, sys, subprocess, time
import numpy as np
from scipy.ndimage import uniform_filter

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

WORK_DIR = r"E:\pytorchFile\YSC_2"
EXE_PATH = r"E:\pytorchFile\YSC_2\Project\Project\multilayer_sim_real_uniform.exe"
RESULTS_DIR = os.path.join(WORK_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_phi_binary(filepath):
    """Load phi field from C++ binary dump."""
    with open(filepath, 'rb') as f:
        res = struct.unpack('i', f.read(4))[0]
        n_cells = res * res * res
        phi_data = struct.unpack(f'{n_cells}d', f.read(n_cells * 8))
    phi = np.array(phi_data, dtype=np.float64).reshape(res, res, res)
    return phi, res


def compute_radial_power_spectrum(phi, dx=0.5):
    """Compute 3D FFT and radial power spectrum P(k)."""
    res = phi.shape[0]
    # Subtract mean to remove DC component
    phi_centered = phi - phi.mean()
    
    # 3D FFT
    phi_hat = np.fft.fftn(phi_centered)
    power = np.abs(phi_hat)**2 / (res**3)
    
    # Frequency grid
    freqs = np.fft.fftfreq(res, d=dx)
    kx, ky, kz = np.meshgrid(freqs, freqs, freqs, indexing='ij')
    k_mag = np.sqrt(kx**2 + ky**2 + kz**2)
    
    # Radial binning
    k_max = np.pi / dx  # Nyquist
    n_bins = 15
    k_edges = np.linspace(0, k_max, n_bins + 1)
    k_centers = 0.5 * (k_edges[:-1] + k_edges[1:])
    
    P_k = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    
    for i in range(n_bins):
        mask = (k_mag >= k_edges[i]) & (k_mag < k_edges[i+1])
        if mask.sum() > 0:
            P_k[i] = power[mask].mean()
        else:
            P_k[i] = 0.0
        counts[i] = mask.sum()
    
    return k_centers, P_k, counts


def detect_turing_peak(k_centers, P_k, counts, k_c_theory=3.28):
    """Detect Turing peak in P(k) and compare with theoretical k_c."""
    # Find peak in P(k), skip bins with zero counts
    valid_mask = (counts > 0) & (np.arange(len(P_k)) > 0)
    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) < 2:
        return {'k_peak': 0, 'k_c_theory': k_c_theory, 'k_ratio': 0,
                'P_peak': 0, 'bg_mean': 0, 'significance': 0, 'peak_idx': 0}
    
    peak_idx = valid_idx[np.argmax(P_k[valid_mask])]
    k_peak = k_centers[peak_idx]
    P_peak = P_k[peak_idx]
    
    # Background from first few valid bins
    bg_bins = [i for i in valid_idx if i != peak_idx and i <= min(peak_idx + 3, len(P_k)-1)]
    bg_mean = np.mean([P_k[i] for i in bg_bins[:5]]) if bg_bins else P_peak
    significance = P_peak / bg_mean if bg_mean > 0 else 1.0
    
    return {
        'k_peak': k_peak,
        'k_c_theory': k_c_theory,
        'k_ratio': k_peak / k_c_theory if k_c_theory > 0 else 0,
        'P_peak': P_peak,
        'bg_mean': bg_mean,
        'significance': significance,
        'peak_idx': peak_idx,
    }


def analyze_single(filepath, label=""):
    """Analyze a single phi field."""
    phi, res = load_phi_binary(filepath)
    
    print(f"Phi field: {res}³ grid, range=[{phi.min():.4f}, {phi.max():.4f}], "
          f"mean={phi.mean():.4f}, std={phi.std():.4f}")
    
    # Smooth power spectrum
    k_centers, P_k, counts = compute_radial_power_spectrum(phi)
    
    # Detect Turing peak
    k_c_theory = 3.2774  # from analyze_cpp_mismatch.py
    peak_info = detect_turing_peak(k_centers, P_k, k_c_theory)
    
    print(f"\n  {'k':>8s}  {'P(k)':>12s}  {'counts':>8s}")
    print(f"  {'-'*8}  {'-'*12}  {'-'*8}")
    for i in range(min(len(k_centers), 15)):
        marker = " <-- PEAK" if i == peak_info['peak_idx'] else ""
        print(f"  {k_centers[i]:8.4f}  {P_k[i]:12.4e}  {counts[i]:8.0f}{marker}")
    
    print(f"\n{'='*60}")
    print(f"Turing Mode Detection: {label}")
    print(f"  k_peak = {peak_info['k_peak']:.4f}")
    print(f"  k_c_theory = {peak_info['k_c_theory']:.4f}")
    print(f"  ratio = {peak_info['k_ratio']:.3f}")
    print(f"  significance = {peak_info['significance']:.1f}x above background")
    
    if peak_info['significance'] > 2.0 and 0.7 < peak_info['k_ratio'] < 1.3:
        print(f"  → TURING MODE DETECTED at k={peak_info['k_peak']:.2f} ≈ k_c")
    elif peak_info['significance'] > 2.0:
        print(f"  → Pattern detected but at unexpected k={peak_info['k_peak']:.2f}")
    else:
        print(f"  → No significant Turing peak detected")
    
    return {
        'res': res, 'phi_mean': float(phi.mean()), 'phi_std': float(phi.std()),
        'k_centers': k_centers.tolist(), 'P_k': P_k.tolist(),
        'peak_info': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                      for k, v in peak_info.items()},
        'label': label,
    }


def compare_gamma_values(gammas, beta=0.6, duration=0.01):
    """Run uniform source sim at multiple gamma and compare Fourier spectra."""
    results = {}
    for gamma in gammas:
        print(f"\n{'='*60}")
        print(f"Running: gamma={gamma}, beta={beta} (uniform source)")

        # Remove old phi field
        phi_path = os.path.join(WORK_DIR, "phi_field.bin")
        if os.path.exists(phi_path):
            os.remove(phi_path)

        env = os.environ.copy()
        env["SIM_GAMMA"] = str(gamma)
        env["SIM_BETA"] = str(beta)
        env["SIM_N_SATS"] = "400"
        env["SIM_DURATION"] = str(duration)
        env["SIM_UNIFORM_SOURCE"] = "1"
        env["SIM_OUTPUT_PHI"] = "1"

        start = time.time()
        result = subprocess.run([EXE_PATH], cwd=WORK_DIR, env=env,
            capture_output=True, text=True, timeout=3600)
        elapsed = time.time() - start
        print(f"  Elapsed: {elapsed:.1f}s")

        if os.path.exists(phi_path):
            analysis = analyze_single(phi_path, f"gamma={gamma}")
            results[gamma] = analysis
            
            # Read core count from JSON
            json_path = os.path.join(WORK_DIR, "multilayer_results_real.json")
            if os.path.exists(json_path):
                with open(json_path) as f:
                    data = json.load(f)
                results[gamma]['avg_cores'] = data['avg_cores']
                print(f"  cores(avg) = {data['avg_cores']:.1f}")
        else:
            print(f"  ERROR: phi_field.bin not found!")
            results[gamma] = {'error': 'phi_field.bin not found'}

    # Summary table
    print(f"\n{'='*60}")
    print(f"GAMMA COMPARISON (uniform source, beta={beta})")
    print(f"  {'γ':>8s}  {'cores':>8s}  {'k_peak':>8s}  {'k/k_c':>8s}  {'signif':>8s}")
    print(f"  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
    for gamma in gammas:
        r = results[gamma]
        if 'error' in r:
            print(f"  {gamma:8.1f}  ERROR")
        else:
            pi = r['peak_info']
            print(f"  {gamma:8.1f}  {r.get('avg_cores', 'N/A'):>8}  "
                  f"{pi['k_peak']:8.3f}  {pi['k_ratio']:8.3f}  {pi['significance']:8.1f}")

    # Save results
    out_path = os.path.join(RESULTS_DIR, "fourier_gamma_comparison.json")
    # Convert numpy arrays to lists for JSON
    json_results = {}
    for gamma, r in results.items():
        json_results[str(gamma)] = {
            'res': r.get('res'), 
            'phi_mean': r.get('phi_mean'), 
            'phi_std': r.get('phi_std'),
            'avg_cores': r.get('avg_cores'),
            'k_centers': r.get('k_centers'),
            'P_k': r.get('P_k'),
            'peak_info': r.get('peak_info'),
        }
    with open(out_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved: {out_path}")
    
    return results


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--compare":
        # Compare gamma values for uniform source
        print("Fourier Spectrum Gamma Comparison (uniform source)")
        print("="*60)
        gammas = [0.1, 0.3, 0.5, 0.7, 1.0, 2.0, 4.0, 8.0]
        results = compare_gamma_values(gammas, beta=0.6, duration=0.01)
        
    elif len(sys.argv) > 1:
        # Analyze single file
        filepath = sys.argv[1]
        results = analyze_single(filepath, os.path.basename(filepath))
        out_path = os.path.join(RESULTS_DIR, "fourier_analysis.json")
        json_results = {
            'res': results['res'], 'phi_mean': results['phi_mean'],
            'phi_std': results['phi_std'], 'k_centers': results['k_centers'],
            'P_k': results['P_k'], 'peak_info': results['peak_info'],
            'label': results['label'],
        }
        with open(out_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        print(f"\nResults saved: {out_path}")
        
    else:
        # Default: analyze phi_field.bin
        filepath = os.path.join(WORK_DIR, "phi_field.bin")
        if os.path.exists(filepath):
            results = analyze_single(filepath, "current phi_field.bin")
        else:
            print("No phi_field.bin found. Use --compare to run simulations.")