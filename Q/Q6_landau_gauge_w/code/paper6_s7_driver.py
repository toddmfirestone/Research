#!/usr/bin/env python3
"""
paper6_s7_driver.py
═══════════════════════════════════════════════════════════════════════════════
Driver script for Paper 6 §7 sign structure analysis.

Builds the ensemble_registry for all (N, β_eff) combinations in Scopes A+B+C,
then calls run_section7_analysis() from paper6_verification_s7.py.

Run this file — NOT paper6_verification_s7.py directly.

Usage:
  python paper6_s7_driver.py

Output files written to OUTPUT_DIR (set below):
  paper6_sign_analysis_results.txt
  paper6_beta_c_estimates.txt

═══════════════════════════════════════════════════════════════════════════════
SCOPE COVERAGE
─────────────────────────────────────────────────────────────────────────────
  Scope A: N ∈ {6, 8},       β_eff ∈ {5.0, 6.0, 9.0, 12.0, 20.0}
  Scope B: N ∈ {6, 8},       β_eff ∈ {9.0, 12.0, 20.0}     (subset of A)
  Scope C: N ∈ {12, 16},     β_eff ∈ {5.0, 6.0, 9.0, 12.0, 20.0}
  Scope D: β_eff ∈ {2.0, 2.5, 3.0, 3.5}  — DEFERRED; not run here.

  This driver runs Scopes A+B+C: all N ∈ {6, 8, 12, 16},
  all β_eff ∈ {5.0, 6.0, 9.0, 12.0, 20.0}.

SEED SCHEME (mirrors §3 / §4):
  Ensemble:   seed = 2000 + int(beta_eff * 100)
  Gauge-fix:  seed = cfg_idx * 100 + start + 222_000

ARRAY SHAPE (canonical, locked §5):
  (N_cfg, N, N, N, N, 4, 2, 2)

SIGNATURE BRIDGE:
  paper6_verification_s4.measure_G_cross_gf(configs_gf_stack, t)
    → returns shape (N_cfg,) for a single timeslice t.
  paper6_verification_s7.run_section7_analysis expects
    measure_G_cross_gf(configs_gf) → (G_cross_mean, G_cross_jk)
  The local wrapper _measure_G_cross_all_t() resolves this.
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import time
import numpy as np

# ── Configuration — edit paths here ──────────────────────────────────────────

OUTPUT_DIR = r'.'          # write output files to current directory
                            # change to e.g. r'D:\Claude\GA\papers\In Work\Paper 6'

# Ensemble parameters (mirrors §3/§4)
N_CFG     = 20
N_STARTS  = 5
N_THERM   = 500
N_DECORR  = 10
EPS_GAUGE = 1e-14
K_MAX_SD  = 10000
ALGORITHM = 'SD'

# Scope A+B+C
N_LIST       = [6, 8, 12, 16]
BETA_EFF_LIST = [5.0, 6.0, 9.0, 12.0, 20.0]

# ── Imports ───────────────────────────────────────────────────────────────────

try:
    from paper6_gauge_fix import (
        gauge_fix_and_check_D19,
        generate_ensemble,
    )
except ImportError as e:
    sys.exit(f"ERROR: Cannot import paper6_gauge_fix.py — {e}\n"
             f"Ensure paper6_gauge_fix.py is in the same directory.")

try:
    # s4.py's measure_G_cross_gf takes (configs_gf_stack, t) → (N_cfg,)
    from paper6_verification_s4 import measure_G_cross_gf as _measure_G_cross_t
except ImportError as e:
    sys.exit(f"ERROR: Cannot import paper6_verification_s4.py — {e}\n"
             f"Ensure paper6_verification_s4.py is in the same directory.")

try:
    from paper6_verification_s7 import run_section7_analysis
except ImportError as e:
    sys.exit(f"ERROR: Cannot import paper6_verification_s7.py — {e}\n"
             f"Ensure paper6_verification_s7.py is in the same directory.")


# ── Signature bridge ──────────────────────────────────────────────────────────

def _measure_G_cross_all_t(configs_gf):
    """
    Bridge between s4.py's per-timeslice measure_G_cross_gf(configs_gf_stack, t)
    and the (G_cross_mean, G_cross_jk) format expected by s7.py.

    Calls s4.py's function for each t, assembles:
      G_cross_per_cfg : (N_cfg, N_t)  — per-config per-timeslice values
      G_cross_mean    : (N_t,)        — ensemble mean
      G_cross_jk      : (N_cfg, N_t) — jackknife leave-one-out samples

    Parameters
    ----------
    configs_gf : np.ndarray, shape (N_cfg, N, N, N, N, 4, 2, 2)

    Returns
    -------
    G_cross_mean : np.ndarray, shape (N_t,)
    G_cross_jk   : np.ndarray, shape (N_cfg, N_t)
    """
    N_cfg_local = configs_gf.shape[0]
    N_t         = configs_gf.shape[1]   # hypercubic: N_t = N

    G_cross_per_cfg = np.zeros((N_cfg_local, N_t), dtype=np.float64)

    for t in range(N_t):
        # s4.py returns shape (N_cfg,) for timeslice t
        G_cross_per_cfg[:, t] = _measure_G_cross_t(configs_gf, t)

    G_cross_mean = np.mean(G_cross_per_cfg, axis=0)   # (N_t,)

    # Jackknife leave-one-out samples
    G_cross_jk = np.zeros((N_cfg_local, N_t), dtype=np.float64)
    for k in range(N_cfg_local):
        mask = np.ones(N_cfg_local, dtype=bool)
        mask[k] = False
        G_cross_jk[k, :] = np.mean(G_cross_per_cfg[mask, :], axis=0)

    return G_cross_mean, G_cross_jk


# ── Monkey-patch: inject bridge into s7 module ────────────────────────────────
# run_section7_analysis calls the module-level measure_G_cross_gf.
# We replace it with our bridge so the (N_cfg, N_t) format is produced correctly.

import paper6_verification_s7 as _s7_module
_s7_module.measure_G_cross_gf = _measure_G_cross_all_t


# ── Ensemble generation ───────────────────────────────────────────────────────

def build_gauge_fixed_ensemble(N, beta_eff, verbose=True):
    """
    Generate a thermalized SU(2) ensemble and apply best-copy Landau
    gauge fixing. Mirrors the §3/§4 seed scheme exactly.

    Parameters
    ----------
    N        : int   — lattice size (hypercubic N⁴)
    beta_eff : float — effective coupling

    Returns
    -------
    configs_gf : np.ndarray, shape (N_CFG, N, N, N, N, 4, 2, 2), complex128
    """
    ens_seed = 2000 + int(beta_eff * 100)

    if verbose:
        print(f"  [N={N}, β={beta_eff:.1f}] Generating ensemble "
              f"(seed={ens_seed}, N_therm={N_THERM}) ...", end='', flush=True)
    t0 = time.time()

    configs_raw, _ = generate_ensemble(
        beta_eff, N, N_CFG,
        N_therm=N_THERM, N_decorr=N_DECORR,
        seed=ens_seed
    )

    if verbose:
        print(f" {time.time()-t0:.1f}s", end='')

    # Best-copy gauge fixing
    if verbose:
        print(f"  gauge-fixing ({N_STARTS} starts) ...", end='', flush=True)
    t1 = time.time()

    configs_gf = np.empty((N_CFG, N, N, N, N, 4, 2, 2), dtype=np.complex128)

    for cfg_idx in range(N_CFG):
        U_raw   = configs_raw[cfg_idx]
        best_FL = -1e30
        best_Ugf = None

        for start in range(N_STARTS):
            gf_seed = cfg_idx * 100 + start + 222_000
            U_gf, _, FL, _, _, conv = gauge_fix_and_check_D19(
                U_raw, beta_eff, N,
                algorithm=ALGORITHM,
                eps_gauge=EPS_GAUGE,
                k_max_SD=K_MAX_SD,
                random_seed=gf_seed
            )
            if conv and FL > best_FL:
                best_FL  = FL
                best_Ugf = U_gf

        if best_Ugf is None:
            best_Ugf = U_gf   # fallback: last run if none converged
            print(f"\n  [WARN] (N={N}, β={beta_eff:.1f}, cfg={cfg_idx}): "
                  f"no gauge-fix start converged; using last run.")

        configs_gf[cfg_idx] = best_Ugf

    if verbose:
        print(f" {time.time()-t1:.1f}s  DONE")

    # Shape assertion (canonical)
    assert configs_gf.shape == (N_CFG, N, N, N, N, 4, 2, 2), \
        f"Shape error: got {configs_gf.shape}"

    return configs_gf


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print('═' * 72)
    print('PAPER 6 — §7 DRIVER: SIGN STRUCTURE AND β_c CHARACTERISATION')
    print(f'  N values  : {N_LIST}')
    print(f'  β_eff     : {BETA_EFF_LIST}')
    print(f'  N_cfg     : {N_CFG}  N_starts: {N_STARTS}')
    print(f'  N_therm   : {N_THERM}  N_decorr: {N_DECORR}')
    print(f'  Output dir: {os.path.abspath(OUTPUT_DIR)}')
    print(f'  D27b gate : scope_D_available = False  (Scope D deferred)')
    print('═' * 72)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Build ensemble registry ───────────────────────────────────────────
    ensemble_registry = {}
    t_total = time.time()

    n_total = len(N_LIST) * len(BETA_EFF_LIST)
    done    = 0

    for N in N_LIST:
        for beta_eff in BETA_EFF_LIST:
            done += 1
            print(f'\n[{done}/{n_total}] Building ensemble '
                  f'(N={N}, β_eff={beta_eff:.1f}) ...')

            configs_gf = build_gauge_fixed_ensemble(N, beta_eff, verbose=True)

            ensemble_registry[(N, beta_eff)] = {
                'configs_gf': configs_gf,
                'N':          N,
                'beta_eff':   beta_eff,
                'N_cfg':      N_CFG,
            }

    print(f'\n[Registry] {len(ensemble_registry)} ensembles built in '
          f'{time.time()-t_total:.1f}s total.')

    # ── Run §7 analysis ───────────────────────────────────────────────────
    print('\n' + '─' * 72)
    print('[§7] Running sign analysis ...')
    print('─' * 72)

    results = run_section7_analysis(ensemble_registry, OUTPUT_DIR)

    # ── Final report ──────────────────────────────────────────────────────
    print('\n' + '═' * 72)
    print('DRIVER COMPLETE')
    print(f'  D27a : {results["D27a_result"]}')
    print(f'  D27b : {results["D27b_result"]}')
    print(f'  Colour discrepancies: {len(results["colour_discrepancies"])}')

    b_c_inf = results['fit_params'].get('beta_c_inf')
    if b_c_inf is not None:
        sig = results['fit_params'].get('sigma_beta_c_inf', float('nan'))
        chi = results['fit_params'].get('chi2_dof', float('nan'))
        print(f'  β_c^∞ : {b_c_inf:.4f} ± {sig:.4f}  (χ²/dof={chi:.3f})')
    else:
        print(f'  β_c^∞ : {results["fit_params"]["note"]}')

    print(f'\n  Output files:')
    print(f'    {os.path.join(os.path.abspath(OUTPUT_DIR), "paper6_sign_analysis_results.txt")}')
    print(f'    {os.path.join(os.path.abspath(OUTPUT_DIR), "paper6_beta_c_estimates.txt")}')
    print('═' * 72)

    # Exit code: 0 if D27a passes (or partial), 1 if D27a fails
    if 'FAIL' in results['D27a_result']:
        sys.exit(1)
    sys.exit(0)


if __name__ == '__main__':
    main()
