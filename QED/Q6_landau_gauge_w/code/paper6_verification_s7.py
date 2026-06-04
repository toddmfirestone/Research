#!/usr/bin/env python3
"""
paper6_verification_s7.py
═══════════════════════════════════════════════════════════════════════════════
Paper 6 — §7 Verification: Sign Structure and β_eff Characterisation
Checks D27a (STANDARD, runnable), D27b (INVERTED, DEFERRED — Scope D only).

Functions implemented (paper-local numbering):
  P6-F6  measure_G_diag_gf          — diagonal correlator
  P6-F7  sign_classify_ensemble      — P/N/Z timeslice classification
  P6-F8  compute_D_tilde_W           — D̃_{W±}(ω=0) summation
  P6-F9  estimate_beta_c             — β_c(N) estimation + 1/N² fit

Support functions:
  estimate_plateau_and_onset         — plateau P_∞ and β_onset per N
  check_colour_symmetry              — G_cross vs G_diag type agreement
  check_D27a                         — sign positivity at weak coupling
  check_D27b                         — sign violation at strong coupling (DEFERRED)

Orchestration:
  run_section7_analysis              — top-level loop over ensemble_registry

Output files (UTF-8):
  paper6_sign_analysis_results.txt
  paper6_beta_c_estimates.txt

═══════════════════════════════════════════════════════════════════════════════
PREREQUISITES
─────────────────────────────────────────────────────────────────────────────
  configs_gf arrays must be pre-gauge-fixed (P6-F1/F2/F3 pipeline complete).
  §7 code does NOT re-run gauge fixing.

  Dependency: measure_G_cross_gf from paper6_verification_s4.py  (P6-F4)
  Dependency: jackknife_error    from paper6_verification_s5.py  (P6-F5 scope)

DEPENDENCY NOTE — paper6_verification_s5.py:
  The CR lists jackknife_error() from s5.py for operations not covered inline.
  All jackknife computations in P6-F6 through P6-F9 are implemented inline
  (standard leave-one-out estimator). paper6_verification_s5.py is imported
  below with a graceful fallback: if the file is absent the inline estimators
  are used throughout and a warning is printed. No correctness impact.

DO NOT RUN THIS FILE. Pass to Todd for processing.

ARRAY SHAPE (canonical, locked §5):
  (N_cfg, N, N, N, N, 4, 2, 2)
  Axes: [cfg, t, x, y, z, μ, row, col]
  a-element: configs_gf[c, t, x, y, z, μ, 0, 0]   (diagonal SU(2) entry)
  b-element: configs_gf[c, t, x, y, z, μ, 0, 1]   (off-diagonal; used by F4)

ENCODING: All file I/O uses encoding='utf-8'.

β_eff SCOPE:
  PHASE1 (runnable):  β_eff ∈ {5.0, 6.0, 9.0, 12.0, 20.0}, N ∈ {6,8,12,16}
  PHASE2 (deferred):  β_eff ∈ {2.0, 2.5, 3.0, 3.5}  — Scope D
  D27b gates on scope_D_available = False (see below).

═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import numpy as np
from datetime import datetime

# ---------------------------------------------------------------------------
# Dependency imports with graceful fallback
# ---------------------------------------------------------------------------
try:
    from paper6_verification_s4 import measure_G_cross_gf
except ImportError:
    print("[WARN] paper6_verification_s4.py not found. "
          "measure_G_cross_gf must be provided externally.")
    measure_G_cross_gf = None

try:
    from paper6_verification_s5 import jackknife_error
    _S5_AVAILABLE = True
except ImportError:
    print("[WARN] paper6_verification_s5.py not found. "
          "Inline jackknife estimators will be used throughout §7.")
    jackknife_error = None
    _S5_AVAILABLE = False

# ---------------------------------------------------------------------------
# D27b gate — set True ONLY after Scope D generation is confirmed.
# ---------------------------------------------------------------------------
scope_D_available = False   # HARD GATE — do not change without Scope D data.


# ═══════════════════════════════════════════════════════════════════════════
# P6-F6  measure_G_diag_gf
# ═══════════════════════════════════════════════════════════════════════════

def measure_G_diag_gf(configs_gf):
    """
    P6-F6: Measure G_diag^{gf}(t) = 8 Σ_{x,μ} Re[a_t^{gf}(x,μ) · ā_0^{gf}(x,μ)]

    Uses the a-element (diagonal SU(2) entry [0,0]), analogous to
    measure_G_cross_gf (P6-F4) which uses the b-element [0,1].

    Parameters
    ----------
    configs_gf : np.ndarray, shape (N_cfg, N, N, N, N, 4, 2, 2), dtype complex128
        Gauge-fixed link configurations.

    Returns
    -------
    G_diag_mean : np.ndarray, shape (N_t,)
        Ensemble mean of G_diag^{gf}(t).
    G_diag_jk   : np.ndarray, shape (N_cfg, N_t)
        Jackknife leave-one-out samples.
    """
    assert configs_gf.ndim == 8, \
        f"Expected 8-dimensional array (N_cfg,N,N,N,N,4,2,2); got ndim={configs_gf.ndim}"
    assert configs_gf.shape[5] == 4, \
        f"Axis 5 must be μ-directions (4); got {configs_gf.shape[5]}"
    assert configs_gf.shape[6] == 2, \
        f"Axis 6 must be SU(2) row index (2); got {configs_gf.shape[6]}"
    assert configs_gf.shape[7] == 2, \
        f"Axis 7 must be SU(2) col index (2); got {configs_gf.shape[7]}"
    assert configs_gf.shape[1] == configs_gf.shape[2] == \
           configs_gf.shape[3] == configs_gf.shape[4], \
        (f"Lattice must be hypercubic (N_t=Nx=Ny=Nz=N); "
         f"got shape {configs_gf.shape[1:5]}")

    N_cfg = configs_gf.shape[0]
    N_t   = configs_gf.shape[1]

    # a-element at reference timeslice t=0: shape (N_cfg, N, N, N, 4)
    a0 = configs_gf[:, 0, :, :, :, :, 0, 0]

    G_diag_per_cfg = np.zeros((N_cfg, N_t), dtype=np.float64)

    for t in range(N_t):
        # a-element at timeslice t: shape (N_cfg, N, N, N, 4)
        a_t = configs_gf[:, t, :, :, :, :, 0, 0]
        # Re[a_t · conj(a_0)]: shape (N_cfg, N, N, N, 4)
        correlator = np.real(a_t * np.conj(a0))
        # Sum over spatial sites and μ-directions; prefactor 8
        G_diag_per_cfg[:, t] = 8.0 * np.sum(correlator, axis=(1, 2, 3, 4))

    G_diag_mean = np.mean(G_diag_per_cfg, axis=0)   # shape (N_t,)

    # Jackknife leave-one-out samples
    G_diag_jk = np.zeros((N_cfg, N_t), dtype=np.float64)
    for k in range(N_cfg):
        mask = np.ones(N_cfg, dtype=bool)
        mask[k] = False
        G_diag_jk[k, :] = np.mean(G_diag_per_cfg[mask, :], axis=0)

    return G_diag_mean, G_diag_jk


# ═══════════════════════════════════════════════════════════════════════════
# P6-F7  sign_classify_ensemble
# ═══════════════════════════════════════════════════════════════════════════

def sign_classify_ensemble(G_mean, G_jk, sigma_threshold=2.0):
    """
    P6-F7: Classify each timeslice into sign categories P, N, Z (§7.1).
    Return ensemble-level type and jackknife σ.

    Timeslice categories (mutually exclusive):
      P (Positive) : G_mean[t]  >  +sigma_threshold * σ_JK(t)
      N (Negative) : G_mean[t]  <  -sigma_threshold * σ_JK(t)
      Z (Zero)     : |G_mean[t]| ≤  sigma_threshold * σ_JK(t)

    Ensemble-level types:
      'P' : All timeslices in category P.
      'N' : At least one N; none in P.
      'M' : At least one P and at least one N (mixed).
      'Z' : All timeslices in category Z.

    Parameters
    ----------
    G_mean          : np.ndarray, shape (N_t,)       — ensemble mean
    G_jk            : np.ndarray, shape (N_cfg, N_t) — jackknife samples
    sigma_threshold : float, default 2.0

    Returns
    -------
    timeslice_categories : list of str, length N_t  ('P', 'N', or 'Z')
    ensemble_type        : str  ('P', 'N', 'M', or 'Z')
    sigma_jk             : np.ndarray, shape (N_t,)  — jackknife standard error
    """
    N_cfg, N_t = G_jk.shape

    # Standard jackknife standard error (inline)
    sigma_jk = np.sqrt(
        (N_cfg - 1) / N_cfg *
        np.sum((G_jk - G_mean[np.newaxis, :]) ** 2, axis=0)
    )

    timeslice_categories = []
    for t in range(N_t):
        mu  = G_mean[t]
        sig = sigma_jk[t]
        if mu > sigma_threshold * sig:
            timeslice_categories.append('P')
        elif mu < -sigma_threshold * sig:
            timeslice_categories.append('N')
        else:
            timeslice_categories.append('Z')

    has_P = 'P' in timeslice_categories
    has_N = 'N' in timeslice_categories

    if has_P and not has_N:
        ensemble_type = 'P'
    elif has_N and not has_P:
        ensemble_type = 'N'
    elif has_P and has_N:
        ensemble_type = 'M'
    else:
        ensemble_type = 'Z'   # all timeslices within 2σ of zero

    return timeslice_categories, ensemble_type, sigma_jk


# ═══════════════════════════════════════════════════════════════════════════
# P6-F8  compute_D_tilde_W
# ═══════════════════════════════════════════════════════════════════════════

def compute_D_tilde_W(G_cross_mean, G_cross_jk, beta_eff, V):
    """
    P6-F8: Compute D̃_{W±}(ω=0) = Σ_t D_{W±}(t) from G_cross^{gf}.

    Uses Theorem 2 working form:
      D_{W±}(t) = G_cross^{gf}(t) × β_eff / (32 V)

    Both diagonal and full-covariance jackknife errors are computed.
    A NOTE is printed if they differ by more than 10%.

    Parameters
    ----------
    G_cross_mean : np.ndarray, shape (N_t,)       — ensemble mean of G_cross
    G_cross_jk   : np.ndarray, shape (N_cfg, N_t) — jackknife samples
    beta_eff     : float                           — effective coupling
    V            : int                             — spatial volume N³

    Returns
    -------
    D_tilde_mean       : float  — ensemble mean of D̃_{W±}(ω=0)
    D_tilde_jk         : np.ndarray, shape (N_cfg,) — jackknife samples
    D_tilde_sigma      : float  — diagonal-approximation jackknife error
    D_tilde_sigma_full : float  — full off-diagonal covariance jackknife error
    """
    N_cfg, N_t = G_cross_jk.shape
    scale = beta_eff / (32.0 * V)

    D_W_mean = G_cross_mean * scale          # shape (N_t,)
    D_W_jk   = G_cross_jk  * scale          # shape (N_cfg, N_t)

    D_tilde_mean = float(np.sum(D_W_mean))
    D_tilde_jk   = np.sum(D_W_jk, axis=1)   # shape (N_cfg,)

    # Diagonal-approximation jackknife error
    D_tilde_sigma = float(np.sqrt(
        (N_cfg - 1) / N_cfg *
        np.sum((D_tilde_jk - D_tilde_mean) ** 2)
    ))

    # Full-covariance jackknife error (off-diagonal timeslice correlations)
    dev        = D_W_jk - D_W_mean[np.newaxis, :]       # shape (N_cfg, N_t)
    cov_matrix = (N_cfg - 1) / N_cfg * (dev.T @ dev)    # shape (N_t, N_t)
    D_tilde_sigma_full = float(np.sqrt(np.sum(cov_matrix)))

    # Report if diagonal and full estimates differ by more than 10%
    denom = max(D_tilde_sigma, 1e-30)
    if abs(D_tilde_sigma_full - D_tilde_sigma) / denom > 0.10:
        print(f"[NOTE] D_tilde sigma: diag={D_tilde_sigma:.6e}, "
              f"full={D_tilde_sigma_full:.6e}  "
              f"(>10% difference; off-diagonal timeslice correlations "
              f"are non-negligible)")

    return D_tilde_mean, D_tilde_jk, D_tilde_sigma, D_tilde_sigma_full


# ═══════════════════════════════════════════════════════════════════════════
# P6-F9  estimate_beta_c
# ═══════════════════════════════════════════════════════════════════════════

def estimate_beta_c(ensemble_types, beta_eff_list, N_list):
    """
    P6-F9: Estimate β_c(N) for each N from the ensemble sign classification.

    Definition (§7.2):
      β_c(N) = sup{ β_eff : ensemble type ∈ {N, M, Z} }

    If all ensembles at lattice size N are Type-P, returns a strict upper
    bound string: '< β_eff_min'.
    If no Type-P at any β_eff, returns a lower bound string: '> β_eff_max'.
    If the sign transition is bracketed, interpolates the midpoint (float).

    A 1/N² fit for β_c^∞ is performed when at least 3 numeric (float)
    estimates are available:
      β_c(N) = β_c^∞ + A/N² + O(N^{-4})

    Acceptance criterion: χ²/dof < 3.0 (consistent with §5 FV fits).
    TENSION is flagged if χ²/dof ≥ 3 (FLAG-8 scope applies).

    Special case: if β_c^∞ is consistent with zero within 2σ_fit, this is
    noted as a physically significant secondary finding.

    Parameters
    ----------
    ensemble_types : dict  (N, β_eff) → str ('P','N','M','Z')
    beta_eff_list  : list of float, sorted ascending
    N_list         : list of int

    Returns
    -------
    beta_c_by_N : dict  N → float | str
    beta_c_inf  : float | None
    fit_params  : dict with keys 'beta_c_inf', 'A', 'chi2_dof', 'note',
                                 'sigma_beta_c_inf' (if fit performed)
    """
    beta_c_by_N = {}
    for N in N_list:
        types_at_N = [
            (b, ensemble_types.get((N, b), 'MISSING'))
            for b in sorted(beta_eff_list)
        ]

        non_P  = [b for b, t in types_at_N if t in ('N', 'M', 'Z')]
        P_vals = [b for b, t in types_at_N if t == 'P']

        if not non_P:
            # All Type-P: strict upper bound only
            beta_c_by_N[N] = f"< {min(beta_eff_list):.1f}"
        elif not P_vals:
            # No Type-P at any β_eff: lower bound
            beta_c_by_N[N] = f"> {max(beta_eff_list):.1f}"
        else:
            # Transition bracketed: interpolate midpoint
            # (fraction-weighted interpolation per §7.2 used when
            #  per-timeslice Category-P fractions are not separately
            #  supplied; midpoint is the conservative fallback)
            beta_hi = max(non_P)
            P_above_hi = [b for b in P_vals if b > beta_hi]
            if P_above_hi:
                beta_lo = min(P_above_hi)
                beta_c_by_N[N] = 0.5 * (beta_hi + beta_lo)
            else:
                beta_c_by_N[N] = f"< {min(beta_eff_list):.1f}"

    # 1/N² fit — requires at least 3 numeric (float) estimates
    N_fit  = [N for N in N_list
              if isinstance(beta_c_by_N.get(N), float)]
    bc_fit = [beta_c_by_N[N] for N in N_fit]

    if len(N_fit) >= 3:
        X = np.column_stack([np.ones(len(N_fit)),
                             1.0 / np.array(N_fit, dtype=float) ** 2])
        coeffs, residuals, _, _ = np.linalg.lstsq(X, bc_fit, rcond=None)
        beta_c_inf_val = float(coeffs[0])
        A_val          = float(coeffs[1])
        predicted      = X @ coeffs

        # χ²/dof (dof = n_points - 2 parameters)
        chi2_dof = float(
            np.sum((np.array(bc_fit) - predicted) ** 2) /
            max(len(N_fit) - 2, 1)
        )

        # Jackknife uncertainty on β_c^∞ (leave-one-out over N values)
        n_fit = len(N_fit)
        jk_bc_inf = []
        for i in range(n_fit):
            mask   = [j for j in range(n_fit) if j != i]
            X_jk   = X[mask, :]
            bc_jk  = [bc_fit[j] for j in mask]
            if len(mask) >= 2:
                c_jk, _, _, _ = np.linalg.lstsq(X_jk, bc_jk, rcond=None)
                jk_bc_inf.append(float(c_jk[0]))
        if jk_bc_inf:
            sigma_bc_inf = float(np.sqrt(
                (n_fit - 1) / n_fit *
                np.sum((np.array(jk_bc_inf) - beta_c_inf_val) ** 2)
            ))
        else:
            sigma_bc_inf = float('nan')

        # Check β_c^∞ consistent with zero (within 2σ_fit)
        zero_consistent = (
            abs(beta_c_inf_val) < 2.0 * sigma_bc_inf
            if not np.isnan(sigma_bc_inf) else False
        )

        note = 'OK'
        if chi2_dof >= 3.0:
            note = ('TENSION — χ²/dof ≥ 3 (FLAG-8 scope applies; '
                    'strong-coupling finite-volume effects may distort '
                    '1/N² form near β_c). β_c^∞ quoted as lower bound only.')
        if zero_consistent:
            note += (' | SECONDARY FINDING: β_c^∞ consistent with zero '
                     'within 2σ_fit. Sign violation may vanish in the '
                     'thermodynamic limit. Do not over-interpret without '
                     'Scope D confirmation.')

        fit_params = {
            'beta_c_inf':       beta_c_inf_val,
            'A':                A_val,
            'chi2_dof':         chi2_dof,
            'sigma_beta_c_inf': sigma_bc_inf,
            'zero_consistent':  zero_consistent,
            'note':             note,
        }
        beta_c_inf = beta_c_inf_val

    else:
        beta_c_inf = None
        fit_params = {
            'beta_c_inf':       None,
            'A':                None,
            'chi2_dof':         None,
            'sigma_beta_c_inf': None,
            'zero_consistent':  False,
            'note':             (f'Insufficient numeric estimates '
                                 f'({len(N_fit)} < 3) for 1/N² fit. '
                                 f'Increase Scope coverage to obtain '
                                 f'bracketed β_c estimates at more N values.'),
        }

    return beta_c_by_N, beta_c_inf, fit_params


# ═══════════════════════════════════════════════════════════════════════════
# estimate_plateau_and_onset
# ═══════════════════════════════════════════════════════════════════════════

def estimate_plateau_and_onset(D_tilde_results, N,
                               plateau_betas=(12.0, 20.0),
                               sigma_threshold=2.0):
    """
    Estimate the D̃_{W±}(ω=0) perturbative plateau P_∞ and the onset
    coupling β_onset at which non-perturbative suppression is first resolved.

    Plateau definition (§7.4):
      P_∞ = mean of D̃_{W±}(ω=0) over β_eff ∈ plateau_betas.
      Consistency check: β_eff = 9.0 should not move P_∞ by more than 1σ.

    β_onset definition:
      Smallest β_eff at which D̃_{W±}(ω=0) is still consistent with P_∞
      within sigma_threshold * combined_σ.
      Combined σ = √(σ_D̃² + σ_{P_∞}²).

    Parameters
    ----------
    D_tilde_results : dict  (N, β_eff) → {'D_tilde_mean', 'D_tilde_sigma',
                                           'D_tilde_sigma_full'}
    N               : int — lattice size
    plateau_betas   : tuple of float — β_eff values defining the plateau
    sigma_threshold : float, default 2.0

    Returns
    -------
    P_inf       : float | None
    sigma_P_inf : float | None
    beta_onset  : float | None
    """
    plateau_vals   = []
    plateau_sigmas = []
    for beta in plateau_betas:
        key = (N, beta)
        if key in D_tilde_results:
            plateau_vals.append(D_tilde_results[key]['D_tilde_mean'])
            plateau_sigmas.append(D_tilde_results[key]['D_tilde_sigma'])

    if not plateau_vals:
        return None, None, None

    P_inf       = float(np.mean(plateau_vals))
    sigma_P_inf = float(np.mean(plateau_sigmas))   # conservative average

    # Optional consistency check: β_eff = 9.0 inclusion
    key_9 = (N, 9.0)
    if key_9 in D_tilde_results:
        val_9   = D_tilde_results[key_9]['D_tilde_mean']
        sig_9   = D_tilde_results[key_9]['D_tilde_sigma']
        if abs(val_9 - P_inf) > sigma_P_inf:
            print(f"[NOTE] N={N}: β_eff=9.0 shifts P_∞ by "
                  f"{abs(val_9-P_inf)/sigma_P_inf:.2f}σ "
                  f"(threshold is 1σ; check plateau stability).")

    # β_onset: smallest β_eff (ascending) consistent with plateau
    beta_onset = None
    for beta in sorted(D_tilde_results.keys(), key=lambda k: k[1]):
        if beta[0] != N:
            continue
        val            = D_tilde_results[beta]['D_tilde_mean']
        sigma          = D_tilde_results[beta]['D_tilde_sigma']
        combined_sigma = np.sqrt(sigma ** 2 + sigma_P_inf ** 2)
        if abs(val - P_inf) < sigma_threshold * combined_sigma:
            beta_onset = beta[1]   # update; keep smallest at end of loop

    return P_inf, sigma_P_inf, beta_onset


# ═══════════════════════════════════════════════════════════════════════════
# check_colour_symmetry
# ═══════════════════════════════════════════════════════════════════════════

def check_colour_symmetry(sign_results):
    """
    Compare G_cross^{gf} and G_diag^{gf} ensemble types at each (N, β_eff).

    Expected agreement (§7.3): at weak coupling, colour symmetry forces
      D^{a=1}(p²) = D^{a=2}(p²) = D^{a=3}(p²)
    so G_diag and G_cross ensemble types must agree.

    Any disagreement is printed to stdout as a FLAG for investigation.
    Possible causes: lattice artefacts, gauge-fixing residuals, finite-N
    boundary effects. A discrepancy is an empirical finding — not
    necessarily a code error — and must not be suppressed.

    Parameters
    ----------
    sign_results : dict  (N, β_eff) → {'cross_type': str, 'diag_type': str, ...}

    Returns
    -------
    discrepancies : list of (N, β_eff) keys with type disagreement
    """
    discrepancies = []
    for key, res in sign_results.items():
        N, beta_eff = key
        ct = res['cross_type']
        dt = res['diag_type']
        if ct != dt:
            discrepancies.append(key)
            print(f"[FLAG] Colour symmetry discrepancy at "
                  f"(N={N}, β_eff={beta_eff:.1f}): "
                  f"G_cross type={ct}, G_diag type={dt}. "
                  f"Possible causes: lattice artefacts, gauge-fixing residuals, "
                  f"finite-N boundary effects. Investigate before publication.")
    return discrepancies


# ═══════════════════════════════════════════════════════════════════════════
# check_D27a — Sign positivity at weak coupling (STANDARD, runnable)
# ═══════════════════════════════════════════════════════════════════════════

def check_D27a(ensemble_types, N_list,
               beta_eff_weak=(9.0, 12.0, 20.0)):
    """
    D27a: Verify sign positivity of ⟨G_cross^{gf}(t)⟩ at weak coupling.

    Condition: For each N ∈ N_list and each β_eff ∈ beta_eff_weak,
      ensemble type must be 'P'.

    Sub-conditions per CR §7.6:
      D27a-1  through D27a-12  (see algebra output for full list).
      D27a-7 through D27a-12 (Scope C: N ∈ {12, 16}) may be DEFERRED if
      Scope C ensembles have not yet been generated.

    Pass  : All available ensembles are Type-P.
    Fail  : Any ensemble is Type-N or Type-M.
    Warning: Type-Z at isolated timeslices (reported; not a fail).
    Partial: Some ensembles DEFERRED (Scope C not yet generated).

    Physical interpretation: Confirms Corollary 6 Part (i) numerically.
    A fail requires revisiting the gauge-fixing pipeline (D19–D23).

    Parameters
    ----------
    ensemble_types : dict  (N, β_eff) → str
    N_list         : list of int
    beta_eff_weak  : tuple of float, default (9.0, 12.0, 20.0)

    Returns
    -------
    overall   : str  ('PASS', 'PARTIAL PASS (some ensembles DEFERRED)', 'FAIL')
    results   : dict  (N, β_eff) → str per sub-condition
    warnings  : list of str
    """
    results  = {}
    warnings = []
    any_fail = False

    for N in N_list:
        for beta in beta_eff_weak:
            key   = (N, beta)
            etype = ensemble_types.get(key, 'MISSING')
            if etype == 'MISSING':
                results[key] = 'DEFERRED'
            elif etype == 'P':
                results[key] = 'PASS'
            elif etype == 'Z':
                results[key] = 'WARNING'
                warnings.append(
                    f"D27a WARNING: (N={N}, β_eff={beta:.1f}) — Type-Z "
                    f"(ensemble mean within 2σ of zero at ≥1 timeslice). "
                    f"Report timeslice index and ensemble key. Not a fail.")
            else:   # 'N' or 'M'
                results[key] = 'FAIL'
                any_fail     = True
                warnings.append(
                    f"D27a FAIL: (N={N}, β_eff={beta:.1f}) — "
                    f"ensemble type='{etype}'. "
                    f"Negative or mixed sign at weak coupling. "
                    f"Revisit gauge-fixing pipeline (D19–D23).")

    any_deferred = any(v == 'DEFERRED' for v in results.values())

    if any_fail:
        overall = 'FAIL'
    elif any_deferred:
        overall = 'PARTIAL PASS (some ensembles DEFERRED)'
    else:
        overall = 'PASS'

    return overall, results, warnings


# ═══════════════════════════════════════════════════════════════════════════
# check_D27b — Sign violation at strong coupling (INVERTED, DEFERRED)
# ═══════════════════════════════════════════════════════════════════════════

def check_D27b(ensemble_types, N_list, beta_eff_strong=2.0):
    """
    D27b (INVERTED): Verify sign violation of ⟨G_cross^{gf}(t)⟩ at strong
    coupling (β_eff = 2.0, Scope D).

    STATUS: DEFERRED. Activation requires scope_D_available = True.

    PASS (INVERTED) = negative sign OBSERVED at ≥1 timeslice for each N.
    This inverted sense reflects Corollary 6 Part (ii): sign-definiteness
    is expected to fail at strong coupling.

    QS5 JUSTIFICATION (mandatory; reproduced verbatim when Type-P observed):
    ─────────────────────────────────────────────────────────────────────────
    A positive result at β_eff = 2.0 (all timeslices Type-P) would indicate
    that non-perturbative IR suppression has not been reached at our available
    lattice volumes. This is not a code error and does not invalidate the
    framework. It implies β_c(N) < 2.0 for the simulated N values, with the
    physical sign violation accessible only at stronger coupling or larger
    volume. In that case, D27b is reported as PARTIAL PASS and β_c(N) is
    bounded: β_c(N) < 2.0 for all N.
    ─────────────────────────────────────────────────────────────────────────

    Parameters
    ----------
    ensemble_types  : dict  (N, β_eff) → str
    N_list          : list of int
    beta_eff_strong : float, default 2.0

    Returns
    -------
    overall : str
    results : dict  (N, β_eff) → str
    msg     : str   (QS5 justification if applicable, else empty)
    """
    if not scope_D_available:
        return (
            'DEFERRED',
            {},
            'Scope D not yet generated. D27b will be activated upon '
            'confirmation that β_eff ∈ {2.0, 2.5, 3.0, 3.5} ensembles '
            'are complete. Set scope_D_available = True to enable.'
        )

    results      = {}
    all_negative = True
    qs5_triggered = False

    QS5_TEXT = (
        "A positive result at β_eff = 2.0 (all timeslices Type-P) would "
        "indicate that non-perturbative IR suppression has not been reached "
        "at our available lattice volumes. This is not a code error and does "
        "not invalidate the framework. It implies β_c(N) < 2.0 for the "
        "simulated N values, with the physical sign violation accessible only "
        "at stronger coupling or larger volume. In that case, D27b is reported "
        "as PARTIAL PASS and β_c(N) is bounded: β_c(N) < 2.0 for all N."
    )

    for N in N_list:
        key   = (N, beta_eff_strong)
        etype = ensemble_types.get(key, 'MISSING')

        if etype in ('N', 'M'):
            results[key] = 'PASS (INVERTED: negative sign observed)'
        elif etype == 'P':
            results[key] = (
                f'QS5: β_eff={beta_eff_strong:.1f}, N={N} — Type-P. '
                f'Implies β_c(N={N}) < {beta_eff_strong:.1f}. '
                f'Not a code error; report as upper bound.'
            )
            all_negative   = False
            qs5_triggered  = True
        elif etype == 'Z':
            results[key] = (f'PARTIAL (borderline; Type-Z at '
                            f'β_eff={beta_eff_strong:.1f})')
            all_negative  = False
        else:
            results[key] = f'MISSING — ensemble ({N}, {beta_eff_strong:.1f}) not found'
            all_negative  = False

    overall = ('PASS (INVERTED)'
               if all_negative
               else 'PARTIAL PASS (QS5 applies; see per-N detail)')
    msg = QS5_TEXT if qs5_triggered else ''

    return overall, results, msg


# ═══════════════════════════════════════════════════════════════════════════
# Output file writers
# ═══════════════════════════════════════════════════════════════════════════

def write_sign_analysis_results(sign_results, D_tilde_results,
                                colour_discrepancies,
                                D27a_result, D27a_detail, D27a_warns,
                                D27b_result, D27b_detail, D27b_msg,
                                filepath):
    """
    Write paper6_sign_analysis_results.txt (UTF-8).

    Sections:
      1 — Ensemble classification table
      2 — Per-timeslice sign detail (non-P ensembles or N_t ≤ 16)
      3 — D̃_{W±}(ω=0) table
      4 — Colour symmetry discrepancies
      5 — D27a result
      6 — D27b result
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    N_list    = sorted(set(k[0] for k in sign_results))
    b_list    = sorted(set(k[1] for k in sign_results))

    lines = []
    hdr = '═' * 72

    lines.append(hdr)
    lines.append('PAPER 6 — §7 SIGN ANALYSIS RESULTS')
    lines.append(f'Generated : {timestamp}')
    lines.append(f'N values  : {N_list}')
    lines.append(f'β_eff     : {b_list}')
    lines.append(hdr)
    lines.append('')

    # ── Section 1: Ensemble classification table ──────────────────────────
    lines.append('SECTION 1 — ENSEMBLE CLASSIFICATION')
    lines.append('─' * 72)
    hdr1 = (f"{'N':>4}  {'β_eff':>6}  {'Type_cross':>10}  "
            f"{'Type_diag':>9}  {'Colour_agree':>12}  {'D27a_status':>14}")
    lines.append(hdr1)
    lines.append('─' * 72)

    for N in N_list:
        for beta in b_list:
            key = (N, beta)
            if key not in sign_results:
                continue
            res   = sign_results[key]
            ct    = res['cross_type']
            dt    = res['diag_type']
            agree = 'Y' if ct == dt else 'N'
            d27a  = D27a_detail.get(key, 'N/A')
            lines.append(
                f"{N:>4}  {beta:>6.1f}  {ct:>10}  "
                f"{dt:>9}  {agree:>12}  {d27a:>14}"
            )
    lines.append('')

    # ── Section 2: Per-timeslice sign detail ─────────────────────────────
    lines.append('SECTION 2 — PER-TIMESLICE SIGN DETAIL')
    lines.append('─' * 72)
    lines.append('(Shown for non-P ensembles; all ensembles if N_t ≤ 16)')
    lines.append(
        f"{'N':>4}  {'β_eff':>6}  {'t':>4}  "
        f"{'G_cross_mean':>14}  {'G_cross_σ':>12}  {'Cat':>4}"
    )
    lines.append('─' * 72)

    for N in N_list:
        for beta in b_list:
            key = (N, beta)
            if key not in sign_results:
                continue
            res  = sign_results[key]
            ct   = res['cross_type']
            cats = res['cross_cats']
            gmn  = res['G_cross_mean']
            gsig = res['G_cross_sig']
            N_t  = len(gmn)
            # Print if non-P or N_t ≤ 16
            if ct != 'P' or N_t <= 16:
                for t in range(N_t):
                    lines.append(
                        f"{N:>4}  {beta:>6.1f}  {t:>4}  "
                        f"{gmn[t]:>14.6e}  {gsig[t]:>12.4e}  "
                        f"{cats[t]:>4}"
                    )
                lines.append('')
    lines.append('')

    # ── Section 3: D̃_{W±}(ω=0) table ─────────────────────────────────────
    lines.append('SECTION 3 — D̃_{W±}(ω=0) vs β_eff')
    lines.append('─' * 72)
    lines.append(
        f"{'N':>4}  {'β_eff':>6}  {'β_eff⁻¹':>8}  "
        f"{'D̃_mean':>14}  {'D̃_σ_diag':>12}  {'D̃_σ_full':>12}"
    )
    lines.append('─' * 72)

    for N in N_list:
        for beta in b_list:
            key = (N, beta)
            if key not in D_tilde_results:
                continue
            dr = D_tilde_results[key]
            lines.append(
                f"{N:>4}  {beta:>6.1f}  {1.0/beta:>8.5f}  "
                f"{dr['D_tilde_mean']:>14.6e}  "
                f"{dr['D_tilde_sigma']:>12.4e}  "
                f"{dr['D_tilde_sigma_full']:>12.4e}"
            )
    lines.append('')

    # ── Section 4: Colour symmetry discrepancies ──────────────────────────
    lines.append('SECTION 4 — COLOUR SYMMETRY DISCREPANCIES')
    lines.append('─' * 72)
    if not colour_discrepancies:
        lines.append('No colour symmetry discrepancies detected.')
        lines.append('G_cross and G_diag ensemble types agree at all (N, β_eff).')
    else:
        lines.append(f'{len(colour_discrepancies)} discrepancy/discrepancies found:')
        for key in colour_discrepancies:
            N, beta = key
            res = sign_results[key]
            lines.append(f'  (N={N}, β_eff={beta:.1f}): '
                         f'G_cross={res["cross_type"]}, '
                         f'G_diag={res["diag_type"]}')
        lines.append('')
        lines.append('Possible causes: lattice artefacts, gauge-fixing residuals,')
        lines.append('finite-N boundary effects. Investigate before publication.')
    lines.append('')

    # ── Section 5: D27a ───────────────────────────────────────────────────
    lines.append('SECTION 5 — D27a: SIGN POSITIVITY AT WEAK COUPLING')
    lines.append('─' * 72)
    lines.append(f'Overall D27a result: {D27a_result}')
    lines.append('')
    lines.append(f"{'N':>4}  {'β_eff':>6}  {'Sub-condition':>16}  {'Status':>30}")
    lines.append('─' * 72)

    # Map (N, beta) to sub-condition number for display
    sub_idx = 1
    beta_weak = (9.0, 12.0, 20.0)
    for N in sorted(set(k[0] for k in D27a_detail)):
        for beta in beta_weak:
            key = (N, beta)
            if key in D27a_detail:
                lines.append(
                    f"{N:>4}  {beta:>6.1f}  "
                    f"{'D27a-'+str(sub_idx):>16}  "
                    f"{D27a_detail[key]:>30}"
                )
                sub_idx += 1

    if D27a_warns:
        lines.append('')
        lines.append('Warnings:')
        for w in D27a_warns:
            lines.append(f'  {w}')
    lines.append('')

    # ── Section 6: D27b ───────────────────────────────────────────────────
    lines.append('SECTION 6 — D27b: SIGN VIOLATION AT STRONG COUPLING (INVERTED)')
    lines.append('─' * 72)
    lines.append(f'Overall D27b result: {D27b_result}')
    if D27b_detail:
        lines.append('')
        for key, status in D27b_detail.items():
            N, beta = key
            lines.append(f'  (N={N}, β_eff={beta:.1f}): {status}')
    if D27b_msg:
        lines.append('')
        lines.append('QS5 JUSTIFICATION (verbatim):')
        lines.append(D27b_msg)
    lines.append('')
    lines.append(hdr)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

    print(f"[OUTPUT] Written: {filepath}")


def write_beta_c_estimates(beta_c_by_N, beta_c_inf, fit_params,
                           onset_results, filepath):
    """
    Write paper6_beta_c_estimates.txt (UTF-8).

    Sections:
      1 — β_c(N) per lattice size
      2 — 1/N² fit results
      3 — β_onset per N
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    lines = []
    hdr   = '═' * 72

    lines.append(hdr)
    lines.append('PAPER 6 — §7 β_c ESTIMATES')
    lines.append(f'Generated : {timestamp}')
    lines.append(hdr)
    lines.append('')

    # ── Section 1: β_c(N) per lattice size ───────────────────────────────
    lines.append('SECTION 1 — β_c(N) PER LATTICE SIZE')
    lines.append('─' * 72)
    lines.append(f"{'N':>4}  {'β_c estimate':>14}  {'Type':>8}  "
                 f"{'Method':>18}  Notes")
    lines.append('─' * 72)

    for N, val in sorted(beta_c_by_N.items()):
        if isinstance(val, float):
            typ    = 'float'
            method = 'midpoint interp.'
            note   = ''
        elif isinstance(val, str) and val.startswith('<'):
            typ    = 'upper bound'
            method = 'all Type-P'
            note   = 'β_c < smallest simulated β_eff'
        elif isinstance(val, str) and val.startswith('>'):
            typ    = 'lower bound'
            method = 'no Type-P found'
            note   = 'β_c > largest simulated β_eff'
        else:
            typ    = 'unknown'
            method = '—'
            note   = str(val)
        lines.append(f"{N:>4}  {str(val):>14}  {typ:>8}  {method:>18}  {note}")
    lines.append('')

    # ── Section 2: 1/N² fit ───────────────────────────────────────────────
    lines.append('SECTION 2 — 1/N² FIT: β_c(N) = β_c^∞ + A/N²')
    lines.append('─' * 72)

    fp = fit_params
    if fp.get('beta_c_inf') is not None:
        lines.append(f"β_c^∞         = {fp['beta_c_inf']:.6f}  "
                     f"± {fp.get('sigma_beta_c_inf', float('nan')):.6f}")
        lines.append(f"A             = {fp['A']:.6f}")
        lines.append(f"χ²/dof        = {fp['chi2_dof']:.4f}")
        lines.append(f"Fit status    : {fp['note']}")
        if fp.get('zero_consistent'):
            lines.append('')
            lines.append('SECONDARY FINDING: β_c^∞ is consistent with zero '
                         'within 2σ_fit.')
            lines.append('  → G_cross^{gf} may be sign-positive in the '
                         'infinite-volume limit.')
            lines.append('  → Defer physical interpretation pending Scope D.')
    else:
        lines.append(f"Fit not performed. Reason: {fp['note']}")
    lines.append('')

    # ── Section 3: β_onset per N ──────────────────────────────────────────
    lines.append('SECTION 3 — β_onset PER LATTICE SIZE')
    lines.append('─' * 72)
    lines.append(f"{'N':>4}  {'P_∞':>14}  {'σ(P_∞)':>10}  "
                 f"{'β_onset':>10}  vs β_c(N)")
    lines.append('─' * 72)

    for N in sorted(onset_results.keys()):
        res    = onset_results[N]
        P_inf  = res['P_inf']
        sig_P  = res['sigma_P_inf']
        b_on   = res['beta_onset']
        b_c    = beta_c_by_N.get(N, 'N/A')

        P_str  = f'{P_inf:.6e}' if P_inf  is not None else 'N/A'
        s_str  = f'{sig_P:.4e}' if sig_P  is not None else 'N/A'
        bo_str = f'{b_on:.1f}'  if b_on   is not None else 'N/A'

        # Comparison annotation
        if isinstance(b_c, float) and b_on is not None:
            diff = abs(b_on - b_c)
            # One β_eff step is typically ~1.0–3.0 in current grid
            note = 'consistent' if diff < 2.0 else 'DISCREPANCY (> 1 step)'
        else:
            note = '—'

        lines.append(f"{N:>4}  {P_str:>14}  {s_str:>10}  "
                     f"{bo_str:>10}  {note}")
    lines.append('')
    lines.append(hdr)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

    print(f"[OUTPUT] Written: {filepath}")


# ═══════════════════════════════════════════════════════════════════════════
# Main orchestration loop
# ═══════════════════════════════════════════════════════════════════════════

def run_section7_analysis(ensemble_registry, output_dir):
    """
    Top-level orchestration for Paper 6 §7 sign analysis.

    ensemble_registry : dict keyed by (N, β_eff):
      {
        'configs_gf' : np.ndarray  shape (N_cfg, N, N, N, N, 4, 2, 2),
        'N'          : int,
        'beta_eff'   : float,
        'N_cfg'      : int
      }

    output_dir : str — directory for output file writing.

    Output files:
      <output_dir>/paper6_sign_analysis_results.txt
      <output_dir>/paper6_beta_c_estimates.txt

    DO NOT RUN — pass to Todd for processing.
    """
    if measure_G_cross_gf is None:
        raise RuntimeError(
            "measure_G_cross_gf (P6-F4) not available. "
            "Ensure paper6_verification_s4.py is on the Python path."
        )

    sign_results    = {}
    D_tilde_results = {}

    for key, ens in ensemble_registry.items():
        N, beta_eff = key
        configs_gf  = ens['configs_gf']

        # Shape assertion at entry (canonical, locked §5)
        assert configs_gf.shape == (ens['N_cfg'], N, N, N, N, 4, 2, 2), (
            f"Shape mismatch at {key}: "
            f"expected ({ens['N_cfg']},{N},{N},{N},{N},4,2,2), "
            f"got {configs_gf.shape}"
        )

        V = N ** 3   # spatial volume

        print(f"[§7] Processing (N={N}, β_eff={beta_eff:.1f}) ...")

        # ── Observable 1: G_cross^{gf} (P6-F4) ──────────────────────────
        G_cross_mean, G_cross_jk = measure_G_cross_gf(configs_gf)
        tc_cross, et_cross, sig_cross = sign_classify_ensemble(
            G_cross_mean, G_cross_jk)

        # ── G_diag^{gf} — colour symmetry check (P6-F6) ─────────────────
        G_diag_mean, G_diag_jk = measure_G_diag_gf(configs_gf)
        tc_diag, et_diag, sig_diag = sign_classify_ensemble(
            G_diag_mean, G_diag_jk)

        # ── D̃_{W±}(ω=0) (P6-F8) ─────────────────────────────────────────
        D_tm, D_tjk, D_ts, D_ts_full = compute_D_tilde_W(
            G_cross_mean, G_cross_jk, beta_eff, V)

        sign_results[key] = {
            'cross_type'  : et_cross,
            'diag_type'   : et_diag,
            'cross_cats'  : tc_cross,
            'diag_cats'   : tc_diag,
            'G_cross_mean': G_cross_mean,
            'G_cross_sig' : sig_cross,
            'G_diag_mean' : G_diag_mean,
            'G_diag_sig'  : sig_diag,
        }
        D_tilde_results[key] = {
            'D_tilde_mean'      : D_tm,
            'D_tilde_sigma'     : D_ts,
            'D_tilde_sigma_full': D_ts_full,
        }

    # ── Colour symmetry check ─────────────────────────────────────────────
    colour_discrepancies = check_colour_symmetry(sign_results)

    # ── β_c estimation (P6-F9) ────────────────────────────────────────────
    ensemble_types = {k: v['cross_type'] for k, v in sign_results.items()}
    beta_eff_list  = sorted(set(k[1] for k in ensemble_registry))
    N_list         = sorted(set(k[0] for k in ensemble_registry))

    beta_c_by_N, beta_c_inf, fit_params = estimate_beta_c(
        ensemble_types, beta_eff_list, N_list)

    # ── D̃ plateau and β_onset per N ──────────────────────────────────────
    onset_results = {}
    for N in N_list:
        P_inf, sig_P, beta_onset = estimate_plateau_and_onset(
            D_tilde_results, N)
        onset_results[N] = {
            'P_inf'      : P_inf,
            'sigma_P_inf': sig_P,
            'beta_onset' : beta_onset,
        }

    # ── D27a check ────────────────────────────────────────────────────────
    D27a_result, D27a_detail, D27a_warns = check_D27a(
        ensemble_types, N_list)

    # ── D27b check (DEFERRED unless scope_D_available = True) ─────────────
    D27b_result, D27b_detail, D27b_msg = check_D27b(
        ensemble_types, N_list)

    # ── Write output files ────────────────────────────────────────────────
    write_sign_analysis_results(
        sign_results, D_tilde_results, colour_discrepancies,
        D27a_result, D27a_detail, D27a_warns,
        D27b_result, D27b_detail, D27b_msg,
        os.path.join(output_dir, 'paper6_sign_analysis_results.txt')
    )

    write_beta_c_estimates(
        beta_c_by_N, beta_c_inf, fit_params, onset_results,
        os.path.join(output_dir, 'paper6_beta_c_estimates.txt')
    )

    # ── Console summary ───────────────────────────────────────────────────
    print('')
    print('─' * 60)
    print(f'[D27a] {D27a_result}')
    print(f'[D27b] {D27b_result}')
    if D27a_warns:
        for w in D27a_warns:
            print(f'  {w}')
    print(f'[Colour discrepancies] {len(colour_discrepancies)} found.')
    if beta_c_inf is not None:
        sigma_str = (
            f"± {fit_params['sigma_beta_c_inf']:.4f}"
            if fit_params.get('sigma_beta_c_inf') is not None else ''
        )
        print(f'[β_c^∞] {beta_c_inf:.4f} {sigma_str}  '
              f'(χ²/dof={fit_params["chi2_dof"]:.3f})')
    else:
        print(f'[β_c^∞] Fit not performed — {fit_params["note"]}')
    print('─' * 60)

    return {
        'sign_results'        : sign_results,
        'D_tilde_results'     : D_tilde_results,
        'colour_discrepancies': colour_discrepancies,
        'beta_c_by_N'         : beta_c_by_N,
        'beta_c_inf'          : beta_c_inf,
        'fit_params'          : fit_params,
        'onset_results'       : onset_results,
        'D27a_result'         : D27a_result,
        'D27a_detail'         : D27a_detail,
        'D27b_result'         : D27b_result,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Entry point (informational only — DO NOT RUN directly)
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("paper6_verification_s7.py — Paper 6 §7 Sign Structure Analysis")
    print("DO NOT RUN this file directly.")
    print("Pass to Todd for processing on the local Windows machine.")
    print("")
    print("Functions defined:")
    print("  P6-F6  measure_G_diag_gf(configs_gf)")
    print("  P6-F7  sign_classify_ensemble(G_mean, G_jk, sigma_threshold=2.0)")
    print("  P6-F8  compute_D_tilde_W(G_cross_mean, G_cross_jk, beta_eff, V)")
    print("  P6-F9  estimate_beta_c(ensemble_types, beta_eff_list, N_list)")
    print("         estimate_plateau_and_onset(D_tilde_results, N, ...)")
    print("         check_colour_symmetry(sign_results)")
    print("         check_D27a(ensemble_types, N_list, ...)")
    print("         check_D27b(ensemble_types, N_list, ...)  [DEFERRED]")
    print("  run_section7_analysis(ensemble_registry, output_dir)")
    print("")
    print(f"scope_D_available = {scope_D_available}  (D27b gated OFF)")
