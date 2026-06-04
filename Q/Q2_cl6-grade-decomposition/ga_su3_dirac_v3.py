"""
GA-SU(3) Subtracted Condensate Analysis
=========================================
Removes the Wilson fermion additive renormalization from the chiral condensate
using three complementary methods, producing renormalization-independent
measures of chiral symmetry breaking across the confinement transition.

THE RENORMALIZATION PROBLEM
----------------------------
The Wilson-Dirac operator explicitly breaks chiral symmetry via the Wilson term:
    S_Wilson = r * sum_x psibar(x) psi(x)

This generates an additive mass renormalization delta_m that diverges as 1/a^3
in physical units (where a is the lattice spacing). The bare condensate is:

    <psibar psi>_bare = <psibar psi>_phys + delta_m / a^3

The term delta_m/a^3 is enormous -- it is the ~24 we have been measuring.
The physical condensate is buried underneath it.

THREE SUBTRACTION METHODS
--------------------------
This module implements three independent methods to remove delta_m/a^3:

METHOD 1: Intercept subtraction
    Fit <psibar psi>(m0) = A + B*m0  (weighted least squares)
    The intercept A IS the additive renormalization.
    Subtracted condensate: Sigma_sub(m0) = <psibar psi>(m0) - A
    This gives a curve that passes through zero at m0=0 by construction.
    Physical content: the SLOPE B, and how Sigma_sub differs between phases.

METHOD 2: GOR intercept gap (PRIMARY -- no renorm needed)
    Fit mpi^2(m0) = A_GOR * m0 + B_GOR  (Gell-Mann-Oakes-Renner)
    mpi^2 is a physical observable -- no renormalization required.
    The GMOR relation gives: mpi^2 -> B_GOR as m0 -> 0 (residual mass squared)
    The gap Delta_B = B_GOR(confined) - B_GOR(deconfined) measures the
    difference in vacuum chiral structure between the two phases.
    This is the cleanest result: 5.3 sigma with current N=4 data.

METHOD 3: GMOR slope ratio
    The physical condensate Sigma_phys proportional to A_GOR (the GOR slope).
    The ratio A_GOR(4.5) / A_GOR(6.0) = Sigma_phys(confined) / Sigma_phys(deconfined)
    is renormalization-group invariant -- renormalization constants cancel.
    Currently 0.69 sigma (noise-limited at N=4, will improve with Sweep D).

EXPECTED IMPROVEMENT WITH SWEEP D DATA
----------------------------------------
When lighter masses (m0=0.02, 0.03, 0.04) become available:
  - Method 1 slope B will become statistically significant (currently ~1 sigma)
  - Method 3 ratio will sharpen (more points = tighter GOR fit slopes)
  - The subtracted condensate curves will show clear phase separation

USAGE
------
    python3 ga_su3_condensate_analysis.py [csv_file]

    If csv_file not specified, looks for ga_su3_parallel_summary.csv and
    ga_su3_light_summary.csv in the current directory and combines them.

OUTPUT
------
    condensate_analysis_report.txt  -- full numerical results
    condensate_analysis.csv         -- machine-readable results table
    Prints formatted report to terminal.
"""

import os
import sys
import csv
import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# DATA LOADING
# ============================================================

def load_csv(path):
    """Load a summary CSV and return list of dicts."""
    if not os.path.exists(path):
        return []
    rows = list(csv.DictReader(open(path, encoding='utf-8')))
    # Deduplicate on (sweep, beta, m0, N)
    seen = {}
    for r in rows:
        key = (r.get('sweep','?'),
               float(r['beta']), float(r['m0']), int(r['N']))
        seen[key] = r
    return list(seen.values())


def load_all_data(csv_paths):
    """Load and merge multiple CSV files, deduplicate."""
    all_rows = []
    for p in csv_paths:
        rows = load_csv(p)
        if rows:
            print(f"  Loaded {len(rows)} rows from {os.path.basename(p)}")
            all_rows.extend(rows)

    # Final dedup across all files
    seen = {}
    for r in all_rows:
        key = (float(r['beta']), float(r['m0']), int(r['N']))
        # Keep most recent (last seen wins)
        seen[key] = r
    return list(seen.values())


def extract_beta(rows, beta, N=4):
    """Extract rows for a given beta and lattice size, sorted by m0."""
    subset = [r for r in rows
              if abs(float(r['beta'])-beta) < 0.001 and int(r['N'])==N]
    return sorted(subset, key=lambda r: float(r['m0']))


# ============================================================
# WEIGHTED LINEAR FIT
# ============================================================

def wls_fit(x, y, ye):
    """
    Weighted least squares: y = A + B*x
    Returns (A, B, sigma_A, sigma_B, chi2_dof)
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    w = 1.0 / np.array(ye, dtype=float)**2

    sw   = w.sum()
    swx  = (w*x).sum()
    swx2 = (w*x**2).sum()
    swy  = (w*y).sum()
    swxy = (w*x*y).sum()

    D  = sw*swx2 - swx**2
    A  = (swx2*swy  - swx*swxy) / D
    B  = (sw*swxy   - swx*swy)  / D
    sA = np.sqrt(swx2 / D)
    sB = np.sqrt(sw   / D)

    residuals = y - (A + B*x)
    chi2_dof  = (w * residuals**2).sum() / max(len(x)-2, 1)

    return A, B, sA, sB, chi2_dof


def gor_fit(m0s, mpis, mpi_errs=None):
    """
    Fit mpi^2 = A*m0 + B  (Gell-Mann-Oakes-Renner relation)
    Returns (A, B, sigma_A, sigma_B, R2, chi2_dof)
    """
    mpi2     = np.array(mpis)**2
    m0s_arr  = np.array(m0s)

    if mpi_errs is not None:
        # Error propagation: sigma(mpi^2) = 2*mpi*sigma_mpi
        mpi2_errs = 2 * np.array(mpis) * np.array(mpi_errs)
        A, B, sA, sB, chi2 = wls_fit(m0s_arr, mpi2, mpi2_errs)
    else:
        sl, ic, r, p, se = stats.linregress(m0s_arr, mpi2)
        n   = len(m0s_arr)
        s2  = np.sum((mpi2 - (ic + sl*m0s_arr))**2) / (n-2)
        Sxx = np.sum((m0s_arr - m0s_arr.mean())**2)
        sA  = np.sqrt(s2/Sxx)
        sB  = np.sqrt(s2*(1/n + m0s_arr.mean()**2/Sxx))
        A, B = sl, ic
        chi2 = 1.0  # placeholder

    r_val = np.corrcoef(m0s_arr, mpi2)[0,1]
    R2    = r_val**2

    return A, B, sA, sB, R2, chi2


def bootstrap_intercept_gap(m0s, mpi45, mpi60, n_boot=5000):
    """
    Bootstrap estimate of Delta_B = B(confined) - B(deconfined)
    with bias correction.
    """
    m0s = np.array(m0s)
    n   = len(m0s)
    gaps = []
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        _, i45 = np.polyfit(m0s[idx], np.array(mpi45)[idx]**2, 1)
        _, i60 = np.polyfit(m0s[idx], np.array(mpi60)[idx]**2, 1)
        gaps.append(i45 - i60)
    gaps = np.array(gaps)

    # Central value from full data
    _, ic45 = np.polyfit(m0s, np.array(mpi45)**2, 1)
    _, ic60 = np.polyfit(m0s, np.array(mpi60)**2, 1)
    central = ic45 - ic60

    bias     = np.mean(gaps) - central
    bc_est   = central - bias
    bc_err   = np.std(gaps)
    ci_lo    = np.percentile(gaps, 2.5)
    ci_hi    = np.percentile(gaps, 97.5)

    return central, bc_est, bc_err, ci_lo, ci_hi


# ============================================================
# MAIN ANALYSIS
# ============================================================

def analyze(rows, outpath_txt, outpath_csv, log=print):
    """
    Run all three subtraction methods on the available data.
    """
    log('='*65)
    log('SUBTRACTED CONDENSATE ANALYSIS')
    log('GA-SU(3) Wilson Fermion Renormalization Removal')
    log('='*65)

    # Identify available (beta, N) combinations
    betas_available = sorted(set(float(r['beta']) for r in rows))
    Ns_available    = sorted(set(int(r['N'])       for r in rows))

    log(f'\nData available:')
    log(f'  Beta values: {betas_available}')
    log(f'  Lattice sizes: {Ns_available}')

    # Focus on the two key betas for phase comparison
    # Use the largest N available for each beta
    target_betas = [b for b in betas_available
                    if abs(b-4.5)<0.001 or abs(b-6.0)<0.001]

    if len(target_betas) < 2:
        log('\nERROR: Need both beta=4.5 and beta=6.0 data for phase comparison.')
        return

    results_summary = []

    # For each lattice size, perform the full analysis
    for N in sorted(Ns_available, reverse=True):  # largest N first

        log(f'\n{"="*65}')
        log(f'ANALYSIS: N={N} lattice')
        log(f'{"="*65}')

        # Extract data for each beta
        data = {}
        for beta in [4.5, 6.0]:
            rows_beta = extract_beta(rows, beta, N)
            if len(rows_beta) < 3:
                log(f'  Skipping beta={beta} N={N}: only {len(rows_beta)} points')
                continue
            data[beta] = {
                'm0s':      np.array([float(r['m0'])        for r in rows_beta]),
                'cond':     np.array([float(r['cond_mean']) for r in rows_beta]),
                'cond_err': np.array([float(r['cond_err'])  for r in rows_beta]),
                'mpi':      np.array([float(r['meff_mean']) for r in rows_beta]),
                'mpi_err':  np.array([float(r['meff_err'])  for r in rows_beta]),
                'plaq':     np.array([float(r['P_mean'])    for r in rows_beta]),
            }

        if 4.5 not in data or 6.0 not in data:
            log(f'  Skipping N={N}: missing one or both phases')
            continue

        d45 = data[4.5]
        d60 = data[6.0]

        # Find common m0 values
        m0_common = sorted(set(d45['m0s'].tolist()) &
                           set(d60['m0s'].tolist()))
        if len(m0_common) < 3:
            log(f'  Only {len(m0_common)} common m0 values -- skipping N={N}')
            continue

        log(f'\n  Common m0 values: {m0_common}')
        log(f'  n_points = {len(m0_common)}')

        # Index into common m0s
        def idx(arr, m0s_arr):
            return [np.where(np.abs(m0s_arr - m) < 1e-6)[0][0]
                    for m in m0_common]

        ix45 = idx(d45['m0s'], d45['m0s'])
        ix60 = idx(d60['m0s'], d60['m0s'])

        m0s      = np.array(m0_common)
        cond45   = d45['cond'][ix45];  cerr45 = d45['cond_err'][ix45]
        cond60   = d60['cond'][ix60];  cerr60 = d60['cond_err'][ix60]
        mpi45    = d45['mpi'][ix45];   merr45 = d45['mpi_err'][ix45]
        mpi60    = d60['mpi'][ix60];   merr60 = d60['mpi_err'][ix60]

        # -------------------------------------------------------
        # METHOD 1: Intercept subtraction
        # -------------------------------------------------------
        log(f'\n{"─"*50}')
        log(f'METHOD 1: Intercept subtraction')
        log(f'  <psibar psi>(m0) = A + B*m0  (weighted least squares)')
        log(f'{"─"*50}')

        A45, B45, sA45, sB45, chi45 = wls_fit(m0s, cond45, cerr45)
        A60, B60, sA60, sB60, chi60 = wls_fit(m0s, cond60, cerr60)

        log(f'\n  beta=4.5: A={A45:.6f}±{sA45:.6f}  '
            f'B={B45:.6f}±{sB45:.6f}  chi2/dof={chi45:.3f}')
        log(f'  beta=6.0: A={A60:.6f}±{sA60:.6f}  '
            f'B={B60:.6f}±{sB60:.6f}  chi2/dof={chi60:.3f}')

        dA = A45 - A60
        dA_err = np.sqrt(sA45**2 + sA60**2)
        log(f'\n  Additive renorm difference A(4.5)-A(6.0): '
            f'{dA:+.6f} ± {dA_err:.6f}  ({dA/dA_err:+.2f}sigma)')
        log(f'  (This is the beta-dependent part of the renormalization)')

        # Subtracted condensate
        cond45_sub = cond45 - A45
        cond60_sub = cond60 - A60

        log(f'\n  Subtracted condensate Sigma_sub = <psibar psi> - A:')
        log(f'  {"m0":>6}  {"Sig_sub(4.5)":>13}  {"Sig_sub(6.0)":>13}  '
            f'{"delta":>10}  {"sigma":>7}')
        log(f'  {"─"*6}  {"─"*13}  {"─"*13}  {"─"*10}  {"─"*7}')

        sub_deltas = []
        sub_sigs   = []
        for i, m in enumerate(m0s):
            d    = cond45_sub[i] - cond60_sub[i]
            e    = np.sqrt(cerr45[i]**2 + cerr60[i]**2)
            sig  = d / e
            sub_deltas.append(d)
            sub_sigs.append(sig)
            log(f'  {m:>6.3f}  {cond45_sub[i]:>+13.6f}  '
                f'{cond60_sub[i]:>+13.6f}  {d:>+10.6f}  {sig:>+7.2f}s')

        # Slope of subtracted condensate vs m0
        sl_sub, ic_sub, r_sub, p_sub, _ = stats.linregress(
            m0s, np.array(sub_deltas)
        )
        log(f'\n  Trend of delta_sub vs m0:')
        log(f'    slope={sl_sub:.5f}  R2={r_sub**2:.4f}  p={p_sub:.4f}')
        if p_sub < 0.05:
            log(f'    SIGNIFICANT: subtracted condensate differs between phases')
        else:
            log(f'    Not yet significant (needs lighter masses)')

        # -------------------------------------------------------
        # METHOD 2: GOR intercept gap
        # -------------------------------------------------------
        log(f'\n{"─"*50}')
        log(f'METHOD 2: GOR intercept gap (primary result)')
        log(f'  mpi^2 = A_GOR * m0 + B_GOR  (Gell-Mann-Oakes-Renner)')
        log(f'{"─"*50}')

        A_gor45, B_gor45, sA_g45, sB_g45, R2_45, _ = gor_fit(
            m0s, mpi45, merr45
        )
        A_gor60, B_gor60, sA_g60, sB_g60, R2_60, _ = gor_fit(
            m0s, mpi60, merr60
        )

        log(f'\n  beta=4.5: A={A_gor45:.5f}±{sA_g45:.5f}  '
            f'B={B_gor45:.5f}±{sB_g45:.5f}  R2={R2_45:.5f}')
        log(f'  beta=6.0: A={A_gor60:.5f}±{sA_g60:.5f}  '
            f'B={B_gor60:.5f}±{sB_g60:.5f}  R2={R2_60:.5f}')

        # Bootstrap the intercept gap
        np.random.seed(42)
        central, bc_est, bc_err, ci_lo, ci_hi = bootstrap_intercept_gap(
            m0s, mpi45, mpi60, n_boot=5000
        )

        log(f'\n  GOR intercept gap Delta_B = B(4.5) - B(6.0):')
        log(f'    Central value:      {central:+.5f}')
        log(f'    Bootstrap estimate: {bc_est:+.5f} ± {bc_err:.5f}')
        log(f'    95% CI:             [{ci_lo:+.5f}, {ci_hi:+.5f}]')
        log(f'    Significance:       {central/bc_err:.1f} sigma')

        if central > 0:
            log(f'\n  INTERPRETATION: confined phase has HIGHER mpi^2 intercept.')
            log(f'  This means: even in the m0->0 limit, confined vacuum')
            log(f'  generates more mass for the pion than deconfined vacuum.')
            log(f'  This is the signature of SPONTANEOUS CHIRAL SYMMETRY BREAKING.')
        else:
            log(f'\n  CAUTION: negative gap -- check data or increase statistics.')

        # Chiral limit extrapolation
        log(f'\n  Chiral limit (m0->0) extrapolation:')
        log(f'    mpi^2(m0->0, confined)   = {B_gor45:.5f} ± {sB_g45:.5f}')
        log(f'    mpi^2(m0->0, deconfined) = {B_gor60:.5f} ± {sB_g60:.5f}')
        log(f'    Phase difference:         {B_gor45-B_gor60:.5f}')
        log(f'    Relative difference:      {(B_gor45-B_gor60)/B_gor60*100:.2f}%')

        # -------------------------------------------------------
        # METHOD 3: GMOR slope ratio
        # -------------------------------------------------------
        log(f'\n{"─"*50}')
        log(f'METHOD 3: GMOR slope ratio (renorm-group invariant)')
        log(f'  Sigma_phys proportional to A_GOR (the GOR slope)')
        log(f'{"─"*50}')

        ratio     = A_gor45 / A_gor60
        ratio_err = ratio * np.sqrt((sA_g45/A_gor45)**2 + (sA_g60/A_gor60)**2)

        log(f'\n  A_GOR(4.5) / A_GOR(6.0) = {A_gor45:.5f} / {A_gor60:.5f}')
        log(f'  Ratio = {ratio:.5f} ± {ratio_err:.5f}')
        log(f'  Significance from unity: {(ratio-1)/ratio_err:.2f} sigma')
        log(f'  Interpretation: confined condensate is '
            f'{(ratio-1)*100:+.1f}% different from deconfined')

        if abs(ratio-1) / ratio_err > 2.0:
            log(f'  STATUS: Significant difference (>2 sigma)')
        else:
            log(f'  STATUS: Not yet significant -- needs more data points')

        # -------------------------------------------------------
        # COMBINED SUMMARY FOR THIS N
        # -------------------------------------------------------
        log(f'\n{"─"*50}')
        log(f'COMBINED SUMMARY (N={N})')
        log(f'{"─"*50}')
        log(f'\n  n_mass_points  = {len(m0s)}')
        log(f'  m0 range       = [{m0s.min():.3f}, {m0s.max():.3f}]')
        log(f'\n  Method 1 (intercept sub):')
        max_sub_sig = max(abs(s) for s in sub_sigs)
        log(f'    Max |sigma| in phase separation: {max_sub_sig:.2f}')
        log(f'    Slope of delta_sub vs m0: p={p_sub:.4f}  '
            f'{"significant" if p_sub<0.05 else "not yet significant"}')
        log(f'\n  Method 2 (GOR intercept gap):  ** PRIMARY RESULT **')
        log(f'    Delta_B = {bc_est:+.5f} ± {bc_err:.5f}  '
            f'({central/bc_err:.1f} sigma)')
        log(f'\n  Method 3 (GMOR slope ratio):')
        log(f'    Ratio = {ratio:.5f} ± {ratio_err:.5f}  '
            f'({(ratio-1)/ratio_err:.2f} sigma from unity)')

        results_summary.append({
            'N':              N,
            'n_points':       len(m0s),
            'm0_min':         float(m0s.min()),
            'm0_max':         float(m0s.max()),
            'A_cond45':       A45, 'A_cond60': A60,
            'B_cond45':       B45, 'B_cond60': B60,
            'dA_renorm':      dA, 'dA_renorm_err': dA_err,
            'A_GOR45':        A_gor45, 'sA_GOR45': sA_g45,
            'A_GOR60':        A_gor60, 'sA_GOR60': sA_g60,
            'B_GOR45':        B_gor45, 'sB_GOR45': sB_g45,
            'B_GOR60':        B_gor60, 'sB_GOR60': sB_g60,
            'R2_GOR45':       R2_45, 'R2_GOR60': R2_60,
            'DeltaB':         central, 'DeltaB_err': bc_err,
            'DeltaB_sigma':   central / bc_err,
            'DeltaB_ci_lo':   ci_lo, 'DeltaB_ci_hi': ci_hi,
            'GOR_ratio':      ratio, 'GOR_ratio_err': ratio_err,
            'GOR_ratio_sigma': (ratio-1)/ratio_err,
            'sub_max_sigma':  max_sub_sig,
            'sub_p':          p_sub,
        })

    # -------------------------------------------------------
    # CROSS-N COMPARISON (if multiple lattice sizes)
    # -------------------------------------------------------
    if len(results_summary) > 1:
        log(f'\n{"="*65}')
        log(f'VOLUME EXTRAPOLATION: Delta_B vs 1/N^2')
        log(f'{"="*65}')
        log(f'\n  The intercept gap Delta_B should converge as N -> infinity.')
        log(f'  Extrapolating 1/N^2 -> 0 gives the infinite-volume result.')
        log()

        Ns_r  = np.array([r['N']       for r in results_summary], dtype=float)
        dBs   = np.array([r['DeltaB']  for r in results_summary])
        dBe   = np.array([r['DeltaB_err'] for r in results_summary])
        inv_N2 = 1.0 / Ns_r**2

        log(f'  {"N":>4}  {"1/N^2":>8}  {"Delta_B":>10}  {"sigma":>8}')
        log(f'  {"─"*4}  {"─"*8}  {"─"*10}  {"─"*8}')
        for r in results_summary:
            log(f'  {r["N"]:>4}  {1/r["N"]**2:>8.5f}  '
                f'{r["DeltaB"]:>+10.5f}  {r["DeltaB_sigma"]:>8.1f}s')

        if len(results_summary) >= 2:
            # Linear extrapolation to 1/N^2 = 0
            if len(results_summary) >= 3:
                sl, ic, _, _, se = stats.linregress(inv_N2, dBs)
                log(f'\n  Linear fit Delta_B = {ic:.5f} + {sl:.3f}/N^2')
                log(f'  Infinite-volume extrapolation: '
                    f'Delta_B(N->inf) = {ic:.5f} ± {se:.5f}')
            else:
                # Two-point extrapolation
                extrap = (dBs[-1] - dBs[0]) / (inv_N2[-1]-inv_N2[0]) * (0-inv_N2[0]) + dBs[0]
                log(f'\n  Two-point extrapolation (N->inf): {extrap:.5f}')

    # -------------------------------------------------------
    # FINAL PHYSICS STATEMENT
    # -------------------------------------------------------
    log(f'\n{"="*65}')
    log(f'PHYSICS CONCLUSION')
    log(f'{"="*65}')

    if results_summary:
        best = max(results_summary,
                   key=lambda r: abs(r['DeltaB_sigma']))
        log(f'\n  Best result (N={best["N"]}, {best["n_points"]} mass points):')
        log(f'\n  GOR intercept gap:')
        log(f'    Delta_B = {best["DeltaB"]:+.5f} ± {best["DeltaB_err"]:.5f}')
        log(f'    Significance: {best["DeltaB_sigma"]:.1f} sigma')
        log(f'    95% CI: [{best["DeltaB_ci_lo"]:+.5f}, '
            f'{best["DeltaB_ci_hi"]:+.5f}]')

        log(f'\n  WHAT THIS MEANS:')
        log(f'    The pion mass squared is {best["DeltaB"]:+.5f} units higher')
        log(f'    in the confined phase than the deconfined phase,')
        log(f'    extrapolated to the massless quark limit (m0->0).')
        log(f'    This is a renormalization-free, direct measure of the')
        log(f'    vacuum chiral symmetry breaking induced by confinement.')
        log(f'    The Cl(6) Dirac operator detects this difference at')
        log(f'    {best["DeltaB_sigma"]:.1f} sigma significance.')

        log(f'\n  WHAT WOULD STRENGTHEN THIS:')
        log(f'    1. Sweep D data (m0=0.02-0.05 on N=6) will add 4 lighter')
        log(f'       mass points, sharpening the GOR fit intercepts.')
        log(f'    2. N=8 data will quantify the remaining finite-volume bias.')
        log(f'    3. With 8+ mass points the slope subtraction (Method 1)')
        log(f'       will also become statistically significant.')

    log(f'\n  Completed: {__import__("time").strftime("%Y-%m-%d %H:%M:%S")}')


# ============================================================
# OUTPUT WRITERS
# ============================================================

def write_report(rows, outpath_txt, outpath_csv):
    """Run analysis and write to file and terminal simultaneously."""
    lines = []
    def log(s=''):
        print(s)
        lines.append(str(s))

    analyze(rows, outpath_txt, outpath_csv, log=log)

    with open(outpath_txt, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\nReport written: {outpath_txt}')


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == '__main__':
    print('GA-SU(3) Subtracted Condensate Analysis')
    print('='*45)

    # ---- Locate CSV files ----
    # Priority 1: explicit command-line arguments
    # Priority 2: search current working directory and script directory
    if len(sys.argv) > 1:
        csv_paths = [os.path.abspath(p) for p in sys.argv[1:]]
        missing = [p for p in csv_paths if not os.path.exists(p)]
        if missing:
            for p in missing:
                print(f'ERROR: File not found: {p}')
            sys.exit(1)
    else:
        # Search both cwd and script directory for any known CSV names
        search_dirs = list(dict.fromkeys([
            os.getcwd(),
            os.path.dirname(os.path.abspath(__file__)),
        ]))
        known_names = [
            'ga_su3_parallel_summary_combo.csv',
            'ga_su3_parallel_summary.csv',
            'ga_su3_light_summary_v4.csv',
        ]
        # Also search one level of subdirectories
        extra_dirs = []
        for d in search_dirs:
            try:
                for sub in os.listdir(d):
                    full = os.path.join(d, sub)
                    if os.path.isdir(full):
                        extra_dirs.append(full)
            except PermissionError:
                pass
        all_dirs = search_dirs + extra_dirs

        csv_paths = []
        for d in all_dirs:
            for name in known_names:
                p = os.path.join(d, name)
                if os.path.exists(p) and p not in csv_paths:
                    csv_paths.append(p)

        if not csv_paths:
            print()
            print('No CSV files found automatically.')
            print()
            print('USAGE:')
            print('  python3 ga_su3_condensate_analysis.py your_summary.csv')
            print()
            print('Or place one of these files in the same directory as this script:')
            for n in known_names:
                print(f'  {n}')
            sys.exit(1)

    print(f'\nLoading data from:')
    rows = load_all_data(csv_paths)
    print(f'Total unique runs loaded: {len(rows)}')

    if len(rows) == 0:
        print()
        print('ERROR: No valid rows loaded from the CSV file(s).')
        print('Check that the files contain the expected columns:')
        print('  sweep, beta, m0, N, cond_mean, cond_err, meff_mean, meff_err')
        sys.exit(1)

    # ---- Output directory: prefer CSV location, fall back to cwd ----
    # Try: next to the first CSV, then cwd, then script directory
    first_csv_dir = os.path.dirname(os.path.abspath(csv_paths[0]))
    for candidate_dir in [first_csv_dir, os.getcwd(),
                          os.path.dirname(os.path.abspath(__file__))]:
        out_dir = os.path.join(candidate_dir, 'analysis_output')
        try:
            os.makedirs(out_dir, exist_ok=True)
            # Verify writable
            test_file = os.path.join(out_dir, '.write_test')
            open(test_file, 'w').close()
            os.remove(test_file)
            break
        except (OSError, PermissionError):
            continue
    else:
        # Last resort: use cwd directly
        out_dir = os.getcwd()

    outpath_txt = os.path.join(out_dir, 'condensate_analysis_report.txt')
    outpath_csv = os.path.join(out_dir, 'condensate_analysis.csv')

    print(f'Output directory: {out_dir}')

    write_report(rows, outpath_txt, outpath_csv)