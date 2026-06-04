"""
GA-SU(3) Banks-Casher Eigenvalue Spectrum
==========================================
Measures the spectral density rho(lambda) of the Cl(6) Wilson-Dirac operator
and extracts the physical chiral condensate via the Banks-Casher relation.

PHYSICS
-------
The Banks-Casher relation (1980) states:
    Sigma_phys = pi * rho(0)
where rho(0) is the density of Dirac eigenvalues at zero and Sigma_phys
is the chiral condensate. This is exact, model-independent, and requires
NO renormalization -- the eigenvalue density is a physical observable.

The Leutwyler-Smilga spectral sum rules give EXACT parameter-free predictions:
    <sum_k 1/lambda_k^2> = Sigma * V / (4 * N_f)          [LS1]
    <sum_k 1/lambda_k^4> = (Sigma * V)^2 / (32 * N_f)     [LS2]

For standard QCD (Cl(1,3), 4-component spinor): N_f = 1
For Cl(6) (8-component spinor): IF the doubled zero mode structure holds,
the effective N_f may be 2, giving different numerical predictions.
This comparison is the UNIQUE test that distinguishes Cl(6) from standard QCD.

The algorithm:
  1. Thermalize gauge field at beta=4.5 and beta=6.0 (both phases)
  2. For each config: compute k_eigen lowest eigenvalues of D†D
  3. Physical Dirac eigenvalues: lambda = sqrt(eigenvalue of D†D)
  4. Pool eigenvalues across configs, bin into histogram
  5. Fit rho(lambda) near lambda=0 to extract rho(0) = Sigma/pi
  6. Compute Leutwyler-Smilga sums and compare to QCD vs Cl(6) predictions

USAGE
-----
    python3 ga_su3_eigenvalue.py

REQUIRES
--------
    ga_su3_lattice_v2.py and ga_su3_dirac_v3.py in same directory.

OUTPUT
------
    eigenvalue_results/
        eigenvalues_b{beta}_N{N}.npy    -- raw eigenvalue arrays
        eigenvalue_report.txt           -- full analysis report
        eigenvalue_summary.csv          -- machine-readable summary
"""

import os, sys, csv, time, warnings
import numpy as np
from scipy.sparse.linalg import eigsh, LinearOperator
from scipy import stats, optimize
import multiprocessing as mp
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'eigenvalue_results')

# Physics parameters
# N=6 recommended: 4x faster than N=8, still sufficient volume
# Run both phases to directly compare rho(0)
RUNS = [
    dict(beta=4.5, N=6, label='confined'),
    # beta=6.0 excluded: nu is noise-dominated at this kappa (narrow band)
]

M0          = 0.05    # same as run 1 -- extending that dataset to 100 configs
N_THERM     = 300     # thermalization sweeps
N_CONFIGS   = 100     # 100 configs -> ~70 spacings below s<1 -> nu error ~0.10
N_DECORR    = 10      # Metropolis sweeps between configs
K_EIGEN     = 50      # number of lowest eigenvalues per config
EIGEN_TOL   = 1e-10   # eigsh tolerance
N_WORKERS   = 1       # single run -- no parallelism needed

# ============================================================
# FRAMEWORK LOADER
# ============================================================

def load_framework():
    g = {}
    for fname in ['ga_su3_lattice_v2.py', 'ga_su3_dirac_v3.py']:
        path = os.path.join(BASE_DIR, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f'{fname} not found in {BASE_DIR}.\n'
                f'Place ga_su3_lattice_v2.py and ga_su3_dirac_v3.py '
                f'in the same directory as this script.'
            )
        src = open(path, encoding='utf-8').read().replace(
            "if __name__ == '__main__':", "if False:"
        )
        exec(compile(src, path, 'exec'), g)
    return g

# ============================================================
# DIRAC EIGENVALUE COMPUTATION (matrix-free)
# ============================================================

def compute_eigenvalues(D, lat, k=50, tol=1e-10):
    """
    Compute the k smallest eigenvalues of D†D using ARPACK (matrix-free).

    D†D is Hermitian positive semidefinite, so eigsh is appropriate.
    Physical Dirac eigenvalues: lambda_k = sqrt(eigenvalue_k of D†D)

    Matrix-free: only the action of D†D on a vector is computed,
    never the full matrix. Memory: O(n_dof) not O(n_dof^2).

    Parameters
    ----------
    D   : DiracOperator instance
    lat : Lattice instance
    k   : number of smallest eigenvalues to compute
    tol : convergence tolerance for eigsh

    Returns
    -------
    lambda_k : array of k Dirac eigenvalues (positive real)
    """
    n = lat.N**lat.D * D.n_dof

    # Build matrix-free LinearOperator for D†D
    def DdagD_matvec(v):
        """Apply D†D to vector v."""
        Dv    = D.apply(v)
        DdagDv = D._apply_Ddag(Dv)
        return DdagDv

    op = LinearOperator((n, n), matvec=DdagD_matvec, dtype=complex)

    # Compute k smallest eigenvalues of D†D
    # which='SM' = smallest magnitude
    # maxiter: ARPACK iteration limit; increase if not converging
    try:
        evals = eigsh(
            op, k=k, which='SM',
            tol=tol, maxiter=10*n,
            return_eigenvectors=False
        )
        # D†D eigenvalues are real and >= 0; take sqrt for Dirac eigenvalues
        evals = np.sort(np.abs(evals.real))
        lambda_k = np.sqrt(np.maximum(evals, 0.0))
        return lambda_k

    except Exception as e:
        print(f'    eigsh failed: {e}')
        return np.array([])

# ============================================================
# SINGLE RUN: thermalize and collect eigenvalues
# ============================================================

def worker_init():
    for v in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS',
              'MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
        os.environ[v] = '1'


def run_eigenvalue_measurement(run_spec):
    """
    Thermalize gauge field, then compute Dirac eigenvalues on N_CONFIGS
    independent configurations.

    Returns dict with eigenvalue arrays and metadata.
    """
    for v in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS',
              'MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
        os.environ[v] = '1'

    beta  = run_spec['beta']
    N     = run_spec['N']
    label = run_spec['label']
    seed  = run_spec.get('seed', 42)

    outpath_npy = os.path.join(OUTPUT_DIR,
                               f'eigenvalues_b{beta}_N{N}.npy')
    outpath_txt = os.path.join(OUTPUT_DIR,
                               f'eigenvalue_run_b{beta}_N{N}.txt')

    # Checkpoint: skip only if we already have enough configs
    if os.path.exists(outpath_npy):
        arr = np.load(outpath_npy)
        if arr.size > 0 and arr.shape[0] >= N_CONFIGS:
            print(f'  [{label}] Skipping -- {outpath_npy} exists '
                  f'({arr.shape[0]} configs >= target {N_CONFIGS})')
            return {
                'beta': beta, 'N': N, 'label': label,
                'eigenvalues': arr,
                'n_configs': arr.shape[0],
                'status': 'skipped',
            }

    try:
        g = load_framework()
    except Exception as e:
        return {'status': 'error', 'error': str(e),
                'beta': beta, 'N': N, 'label': label}

    np.random.seed(seed)
    t0 = time.time()

    cl  = g['CliffordCl6']()
    su3 = g['SU3Algebra'](eps=0.3, g=1.0)
    lat = g['Lattice'](N=N, D=4)

    all_eigenvalues = []   # shape will be (N_CONFIGS, K_EIGEN)

    with open(outpath_txt, 'w', encoding='utf-8', buffering=1) as f:
        def log(s=''):
            print(s); f.write(str(s)+'\n'); f.flush()

        log('='*60)
        log(f'EIGENVALUE RUN: beta={beta}  N={N}  ({label})')
        log(f'  m0={M0}  k_eigen={K_EIGEN}  n_configs={N_CONFIGS}')
        log(f'  n_therm={N_THERM}  n_decorr={N_DECORR}')
        log(f'  Started: {time.strftime("%Y-%m-%d %H:%M:%S")}')
        log('='*60)

        # Thermalization
        log(f'\nThermalization: {N_THERM} sweeps at beta={beta}...')
        fields = g['Fields'](lat, cl, su3, mode='hot')
        metro  = g['Metropolis'](fields, beta=beta, eps=0.25)

        log(f'  {"Sweep":>6}  {"<P>":>10}  {"accept":>8}')
        for i in range(1, N_THERM+1):
            acc = metro.sweep()
            if i % max(1, N_THERM//8) == 0 or i == 1:
                p = g['Observables'](fields).avg_plaquette()
                log(f'  {i:>6}  {p:>10.6f}  {acc:>8.4f}')

        p_therm = g['Observables'](fields).avg_plaquette()
        log(f'\n  Post-therm <P>={p_therm:.6f}  acc={metro.acceptance_rate:.4f}')

        # Measurement loop
        log(f'\nComputing {K_EIGEN} eigenvalues on '
            f'{N_CONFIGS} configurations...')
        log(f'  {"Cfg":>4}  {"<P>":>8}  {"lambda_1":>10}  '
            f'{"lambda_5":>10}  {"lambda_10":>11}  {"time":>7}')

        for cfg in range(1, N_CONFIGS+1):
            for _ in range(N_DECORR): metro.sweep()
            plaq = g['Observables'](fields).avg_plaquette()

            # Build Dirac operator
            D = g['DiracOperator'](fields, m0=M0, r=1.0)

            # Compute eigenvalues
            tc = time.time()
            evals = compute_eigenvalues(D, lat, k=K_EIGEN, tol=EIGEN_TOL)
            dt    = time.time() - tc

            if len(evals) == K_EIGEN:
                all_eigenvalues.append(evals)
                log(f'  {cfg:>4}  {plaq:>8.5f}  {evals[0]:>10.6f}  '
                    f'{evals[4]:>10.6f}  {evals[9]:>11.6f}  {dt:>6.1f}s')
            else:
                log(f'  {cfg:>4}  {plaq:>8.5f}  eigsh FAILED  {dt:>6.1f}s')

        # Save raw eigenvalues
        if all_eigenvalues:
            arr = np.array(all_eigenvalues)
            np.save(outpath_npy, arr)
            log(f'\nSaved eigenvalues: {outpath_npy}')
            log(f'Shape: {arr.shape}  ({arr.shape[0]} configs x {arr.shape[1]} eigenvalues)')
        else:
            arr = np.array([])
            log('\nWARNING: No eigenvalues collected')

        elapsed = (time.time()-t0)/60
        log(f'\nElapsed: {elapsed:.1f} minutes')
        log(f'Completed: {time.strftime("%Y-%m-%d %H:%M:%S")}')

    return {
        'beta':        beta,
        'N':           N,
        'label':       label,
        'eigenvalues': arr,
        'n_configs':   len(all_eigenvalues),
        'status':      'completed',
        'elapsed_min': elapsed,
    }

# ============================================================
# SPECTRAL ANALYSIS: Banks-Casher + Leutwyler-Smilga
# ============================================================

def analyze_spectrum(results, log=print):
    """
    Given eigenvalue measurements from both phases, compute:
      1. Spectral density rho(lambda) for each phase
      2. Banks-Casher condensate: Sigma = pi * rho(0)
      3. Leutwyler-Smilga sum rules and comparison to QCD vs Cl(6)
      4. Phase ratio Sigma(confined) / Sigma(deconfined)
    """
    log('\n' + '='*65)
    log('SPECTRAL ANALYSIS')
    log('='*65)

    phase_data = {}
    for r in results:
        if r.get('status') in ('completed', 'skipped') and r['eigenvalues'].size > 0:
            phase_data[r['label']] = r

    if len(phase_data) < 1:
        log('No completed results to analyze.')
        return {}
    if len(phase_data) < 2:
        log(f'Single-phase analysis mode: {list(phase_data.keys())}')
        log('Phase comparison skipped -- running spectral analysis on available phase.')

    condensates = {}

    for label, r in phase_data.items():
        beta = r['beta']
        N    = r['N']
        evals_all = r['eigenvalues']   # shape (n_configs, K_EIGEN)
        n_cfg     = evals_all.shape[0]
        V         = N**4

        log(f'\n{"─"*50}')
        log(f'{label.upper()} PHASE  (beta={beta}, N={N})')
        log(f'{"─"*50}')
        log(f'  {n_cfg} configurations x {evals_all.shape[1]} eigenvalues')
        log(f'  Total eigenvalues: {evals_all.size}')

        # Pool all eigenvalues
        evals_flat = evals_all.flatten()
        log(f'  lambda range: [{evals_flat.min():.5f}, {evals_flat.max():.5f}]')
        log(f'  Mean smallest eigenvalue: {evals_all[:,0].mean():.5f} '
            f'± {evals_all[:,0].std():.5f}')

        # ---- BANKS-CASHER: fit rho(lambda) near lambda=0 ----
        log(f'\n  BANKS-CASHER: rho(lambda) near lambda=0')

        # Histogram
        n_bins  = 20
        lam_max = np.percentile(evals_flat, 60)   # use lower 60% for density
        bins    = np.linspace(0, lam_max, n_bins+1)
        counts, edges = np.histogram(evals_flat, bins=bins)
        bin_centers   = (edges[:-1] + edges[1:]) / 2
        bin_width     = edges[1] - edges[0]

        # Normalize: rho(lambda) = counts / (n_configs * V * bin_width)
        # The factor V comes from the density being per unit volume
        rho = counts / (n_cfg * V * bin_width)

        log(f'  Spectral density rho(lambda):')
        for i in range(min(8, n_bins)):
            log(f'    lambda={bin_centers[i]:.4f}: rho={rho[i]:.4f}')

        # Fit rho(lambda) = rho_0 + rho_1 * lambda^2 near lambda=0
        # (chRMT prediction: rho is flat near zero to leading order)
        # Use only the lowest bins for the fit
        n_fit = min(6, n_bins//3)
        x_fit = bin_centers[:n_fit]
        y_fit = rho[:n_fit]
        ye_fit = np.sqrt(counts[:n_fit]+1) / (n_cfg * V * bin_width)

        # Linear fit: rho = a + b*lambda^2
        def rho_model(lam, a, b):
            return a + b * lam**2

        try:
            from scipy.optimize import curve_fit
            popt, pcov = curve_fit(rho_model, x_fit, y_fit,
                                   p0=[y_fit[0], 0.0], sigma=ye_fit,
                                   maxfev=1000)
            perr = np.sqrt(np.diag(pcov))
            rho_0     = popt[0]
            rho_0_err = perr[0]

            # Banks-Casher: Sigma = pi * rho(0)
            Sigma     = np.pi * rho_0
            Sigma_err = np.pi * rho_0_err

            log(f'\n  Fit rho(lambda) = rho_0 + rho_1*lambda^2:')
            log(f'    rho_0 = {rho_0:.5f} ± {rho_0_err:.5f}')
            log(f'    rho_1 = {popt[1]:.5f} ± {perr[1]:.5f}')
            log(f'\n  Banks-Casher condensate:')
            log(f'    Sigma = pi * rho(0) = {Sigma:.5f} ± {Sigma_err:.5f}')

        except Exception as e:
            log(f'  Density fit failed: {e}')
            # Fall back: use mean of lowest eigenvalues
            rho_0     = rho[:3].mean()
            rho_0_err = rho[:3].std() / np.sqrt(3)
            Sigma     = np.pi * rho_0
            Sigma_err = np.pi * rho_0_err
            log(f'  Fallback: rho_0 (mean of lowest bins) = {rho_0:.5f}')
            log(f'  Sigma = {Sigma:.5f} ± {Sigma_err:.5f}')

        # ---- LEUTWYLER-SMILGA SUM RULES ----
        log(f'\n  LEUTWYLER-SMILGA SUM RULES')
        log(f'  (per configuration, averaged over {n_cfg} configs)')

        # Compute per-config sums
        ls1_per_cfg = np.array([np.sum(1.0/evals_all[i]**2)
                                 for i in range(n_cfg)])
        ls2_per_cfg = np.array([np.sum(1.0/evals_all[i]**4)
                                 for i in range(n_cfg)])

        ls1_mean = ls1_per_cfg.mean()
        ls2_mean = ls2_per_cfg.mean()
        ls1_err  = ls1_per_cfg.std() / np.sqrt(n_cfg)
        ls2_err  = ls2_per_cfg.std() / np.sqrt(n_cfg)

        log(f'  Measured:')
        log(f'    LS1 = sum(1/lambda^2) = {ls1_mean:.4f} ± {ls1_err:.4f}')
        log(f'    LS2 = sum(1/lambda^4) = {ls2_mean:.4f} ± {ls2_err:.4f}')

        # Standard QCD prediction (N_f=1, using Sigma from Banks-Casher)
        # LS1_pred = Sigma * V / 4
        # LS2_pred = (Sigma * V)^2 / 32
        log(f'\n  Predictions from Banks-Casher Sigma={Sigma:.5f}:')
        log(f'  {"Model":>20}  {"N_f":>4}  {"LS1_pred":>12}  {"LS2_pred":>14}')
        log(f'  {"─"*20}  {"─"*4}  {"─"*12}  {"─"*14}')

        for model, Nf in [('Standard QCD (Cl13)', 1),
                           ('Cl(6) doubled', 2),
                           ('Cl(6) quadrupled', 4)]:
            ls1_pred = Sigma * V / (4 * Nf)
            ls2_pred = (Sigma * V)**2 / (32 * Nf)
            ls1_chi  = (ls1_mean - ls1_pred) / ls1_err if ls1_err > 0 else 0
            ls2_chi  = (ls2_mean - ls2_pred) / ls2_err if ls2_err > 0 else 0
            log(f'  {model:>20}  {Nf:>4}  '
                f'{ls1_pred:>12.4f}  {ls2_pred:>14.4f}  '
                f'[LS1: {ls1_chi:+.1f}s, LS2: {ls2_chi:+.1f}s]')

        # Which model fits best?
        best_Nf = None
        best_chi2 = float('inf')
        for Nf in [1, 2, 4]:
            ls1_pred = Sigma * V / (4 * Nf)
            ls2_pred = (Sigma * V)**2 / (32 * Nf)
            chi2 = ((ls1_mean-ls1_pred)/ls1_err)**2 + ((ls2_mean-ls2_pred)/ls2_err)**2
            if chi2 < best_chi2:
                best_chi2 = chi2
                best_Nf   = Nf

        log(f'\n  BEST FIT: N_f={best_Nf}  chi2={best_chi2:.3f}')
        if best_Nf == 1:
            log(f'  => Standard QCD universality class (Cl(1,3))')
        elif best_Nf == 2:
            log(f'  => Cl(6) DOUBLED universality class -- differs from QCD')
            log(f'     This would be a NOVEL result specific to Cl(6) spinors')
        elif best_Nf == 4:
            log(f'  => Cl(6) QUADRUPLED class -- strong deviation from QCD')

        # RMT eigenvalue distribution test
        log(f'\n  RANDOM MATRIX THEORY: eigenvalue spacing distribution')
        log(f'  For chGUE (QCD universality): P(s) = pi*s/2 * exp(-pi*s^2/4)')
        log(f'  For GUE (different class):   P(s) = 32*s^2/pi^2 * exp(-4*s^2/pi)')

        # Compute unfolded spacing distribution from k=1 eigenvalues
        lambda1 = evals_all[:, 0]   # smallest eigenvalue per config
        lambda2 = evals_all[:, 1]
        spacings = lambda2 - lambda1
        mean_spacing = spacings.mean()
        s = spacings / mean_spacing   # normalized spacing

        # Compare to Wigner surmise: chGUE gives p(s) ~ s * exp(-s^2)
        # GUE gives p(s) ~ s^2 * exp(-s^2)
        # The POWER of s at small s distinguishes them

        # Fit P(s) ~ s^nu for small s
        s_small = s[s < 1.0]
        if len(s_small) > 5:
            try:
                log_s   = np.log(s_small + 1e-10)
                log_N   = np.log(np.arange(1, len(s_small)+1)
                                 / len(s_small) + 1e-10)
                nu, _, r_nu, _, _ = stats.linregress(log_s, log_N)
                log(f'  Fitted level repulsion exponent nu = {nu:.3f}')
                log(f'    chGUE (QCD) predicts nu ~ 1')
                log(f'    GUE predicts nu ~ 2')
                if abs(nu - 1) < abs(nu - 2):
                    log(f'  => Consistent with chGUE (standard QCD universality)')
                else:
                    log(f'  => Closer to GUE -- possible Cl(6) modification')
            except Exception:
                pass

        condensates[label] = {
            'Sigma':     Sigma,
            'Sigma_err': Sigma_err,
            'rho_0':     rho_0,
            'LS1':       ls1_mean,
            'LS2':       ls2_mean,
            'LS1_err':   ls1_err,
            'LS2_err':   ls2_err,
            'best_Nf':   best_Nf,
            'V':         V,
        }

    # ---- PHASE COMPARISON ----
    if 'confined' in condensates and 'deconfined' in condensates:
        c  = condensates['confined']
        d  = condensates['deconfined']

        log(f'\n{"="*65}')
        log(f'PHASE COMPARISON: Banks-Casher Condensate Ratio')
        log(f'{"="*65}')

        ratio     = c['Sigma'] / d['Sigma'] if d['Sigma'] > 0 else float('nan')
        ratio_err = ratio * np.sqrt((c['Sigma_err']/c['Sigma'])**2
                                  + (d['Sigma_err']/d['Sigma'])**2)
        sig       = (ratio-1) / ratio_err if ratio_err > 0 else 0

        log(f'\n  Sigma(confined)   = {c["Sigma"]:.5f} ± {c["Sigma_err"]:.5f}')
        log(f'  Sigma(deconfined) = {d["Sigma"]:.5f} ± {d["Sigma_err"]:.5f}')
        log(f'  Ratio = {ratio:.5f} ± {ratio_err:.5f}')
        log(f'  Significance from unity: {sig:.2f} sigma')

        log(f'\n  COMPARISON TO GOR RESULT:')
        log(f'  GOR intercept ratio (from condensate analysis): 1.027-1.033')
        log(f'  Banks-Casher ratio (from eigenvalue spectrum):  {ratio:.4f}')

        if abs(ratio - 1.030) < 3 * ratio_err:
            log(f'  STATUS: CONSISTENT -- both methods agree to within 3 sigma')
            log(f'  This is strong evidence the chiral condensate measurement is real')
        else:
            log(f'  STATUS: TENSION -- methods disagree by '
                f'{abs(ratio-1.030)/ratio_err:.1f} sigma')
            log(f'  Investigate: finite volume? plateau artifact? renormalization?')

        log(f'\n  UNIVERSALITY CLASS:')
        if c['best_Nf'] == d['best_Nf'] == 1:
            log(f'  Both phases fit N_f=1 (standard QCD Cl(1,3) class)')
            log(f'  The Cl(6) spinor does NOT change the universality class')
            log(f'  The chiral dynamics are equivalent to standard QCD')
        elif c['best_Nf'] == 2 or d['best_Nf'] == 2:
            log(f'  At least one phase fits N_f=2 (Cl(6) doubled class)')
            log(f'  This is a NOVEL finding: Cl(6) modifies the topology coupling')
            log(f'  The effective number of fermionic degrees of freedom near zero')
            log(f'  is doubled compared to standard QCD Wilson fermions')
        else:
            log(f'  Ambiguous: more data needed to determine universality class')

    return condensates


# ============================================================
# MAIN
# ============================================================

def main():
    print('='*62)
    print('GA-SU(3) BANKS-CASHER EIGENVALUE SPECTRUM')
    print(f'  N={RUNS[0]["N"]}  m0={M0}  k_eigen={K_EIGEN}')
    print(f'  Betas: {[r["beta"] for r in RUNS]}')
    print(f'  Workers: {N_WORKERS}')
    print('='*62)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Assign seeds
    for i, r in enumerate(RUNS):
        r['seed'] = 1000 + i * 7919

    print(f'\nEstimated time per config (N={RUNS[0]["N"]}): ~2-15 minutes')
    print(f'Total estimated wall time: ~{N_CONFIGS * 5 / N_WORKERS:.0f} minutes')
    print(f'(varies widely with eigsh convergence speed)')
    print()

    # Run both phases in parallel
    results = []
    with mp.Pool(processes=N_WORKERS, initializer=worker_init,
                 maxtasksperchild=1) as pool:
        for result in pool.imap_unordered(run_eigenvalue_measurement,
                                           RUNS, chunksize=1):
            status = result.get('status', '?')
            label  = result.get('label', '?')
            n_cfg  = result.get('n_configs', 0)
            print(f'  {status:>10}  {label}  ({n_cfg} configs collected)')
            results.append(result)

    # Spectral analysis
    report_path = os.path.join(OUTPUT_DIR, 'eigenvalue_report.txt')
    lines = []

    def log(s=''):
        print(s)
        lines.append(str(s))

    condensates = analyze_spectrum(results, log=log)

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\nReport: {report_path}')

    # CSV summary
    csv_path = os.path.join(OUTPUT_DIR, 'eigenvalue_summary.csv')
    fields   = ['label','beta','N','Sigma','Sigma_err','rho_0',
                'LS1','LS2','LS1_err','LS2_err','best_Nf']
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for label, c in condensates.items():
            row = {'label': label}
            row.update(c)
            w.writerow(row)
    print(f'CSV:    {csv_path}')

    print('\n' + '='*62)
    print('COMPLETE')
    print('='*62)
    print(f'\nRaw eigenvalues saved as .npy files in {OUTPUT_DIR}/')
    print('These can be reanalyzed at any time with different bin sizes.')
    print()
    print('Key outputs to bring back:')
    print('  1. eigenvalue_report.txt -- full spectral analysis')
    print('  2. eigenvalue_summary.csv -- condensate ratio and sum rules')
    print('  3. eigenvalues_b4.5_N6.npy -- raw confined eigenvalues')
    print('  4. eigenvalues_b6.0_N6.npy -- raw deconfined eigenvalues')


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()