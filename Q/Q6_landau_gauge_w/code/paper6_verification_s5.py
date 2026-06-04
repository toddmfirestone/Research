#!/usr/bin/env python3
"""
paper6_verification_s5.py
═══════════════════════════════════════════════════════════════════════════════
Paper 6 — §5 Verification: Statistical Convergence of ⟨G_cross^{gf}⟩
CR: Paper6_CodeRequirements_S5.txt  [v1.2]
Check IDs: D26-PHASE1 (D21c as deferred check now discharged)

SCOPE FOR THIS RUN (operator decisions 2026-05-06):
  N ∈ {6, 12}           — N=8 and N=16 deferred (computational scope)
  β_eff ∈ {5.0,6.0,9.0,12.0}  — Scope A (N=6) + Scope C partial (N=12)
  β_eff = 20.0           — Scope B (N=6 only; unblocks D25 β=20)
  Scope D (β∈{2,2.5,3,3.5})   — DEFERRED (Group E1; not yet generated)

ARRAY SHAPE CONVENTION (locked v1.2):
  Single config:  (N, N, N, N, 4, 2, 2)   axes [t,x,y,z,mu,row,col]
  Full ensemble:  (N_cfg, N, N, N, N, 4, 2, 2)

D21c DISCHARGE:
  D21c (DEFERRED since §2) runs here using N=6 and N=12 data.
  Check: mean(F_L^gf; N=6) / mean(F_L^gf; N=12) ≈ V(6)/V(12) = 1/16
  within ±5% relative tolerance.

D25 β=20.0 STATUS:
  CR §5 says this "transitions from DEFERRED to OPEN."
  D25 β=20.0 was already completed in §4 (PARTIAL PASS, FINDING-4).
  The append step is SKIPPED; existing entry in paper6_verification_summary_s4.txt
  is authoritative.

FV SCALING NOTE:
  With N ∈ {6, 12} only (2 data points), the D(N) = D_inf + a/N² fit is
  DEGENERATE: DOF = len(N_values) - 2 = 0. The fit uniquely determines
  (D_inf, a) with no chi²/dof. STATUS always reported as "DEGENERATE";
  ACCEPT/TENSION criterion suspended until N=8 and N=16 data are available.

SEED SCHEME (non-overlapping with §2–§4):
  §2/§3 ensemble seeds:    2000 + int(β×100)      [N=6, N_cfg=20]
  §5 N=6  ensemble seeds:  5000 + int(β×100)      [N=6, N_cfg=100, new]
  §5 N=12 ensemble seeds: 12000 + int(β×100)      [N=12, new]
  §5 gauge-fix seeds:  cfg×100 + start + N×10000 + 500000

DEPENDENCIES:
  paper6_gauge_fix.py       — gauge fixing, SU(2) utilities
  su2_corrected_T3.py       — Metropolis sweep, plaquette
  paper6_verification_s4.py — optional; run independently for §4 results.
                              measure_G_cross_gf is defined inline here to
                              avoid spawn-method circular import on Windows.
  numpy, scipy              — numerics

OUTPUT FILES:
  paper6_verification_summary_s5.txt  — main verification summary
  paper6_coupling_table.txt           — Appendix B plaquette/coupling data
  paper6_fv_scaling_results.txt       — FV fit results (2-pt, degenerate)
"""

import sys
import time
import math
import multiprocessing
import numpy as np
from scipy.linalg import solve

# ── Import Paper 6 infrastructure ─────────────────────────────────────────────
try:
    from paper6_gauge_fix import (
        gauge_fix_and_check_D19,
        compute_FL,
        random_su2,
    )
except ImportError as e:
    sys.exit(f"ERROR: Cannot import paper6_gauge_fix.py — {e}")

try:
    from su2_corrected_T3 import metropolis_sweep
    from su2_corrected_T3 import plaquette as _plaquette_raw
    def plaquette_fn(L):
        return float(_plaquette_raw(L))
except ImportError as e:
    sys.exit(f"ERROR: Cannot import su2_corrected_T3.py — {e}")

try:
    from paper6_verification_s4 import (
        run_D24,
        run_D25,
        _get_best_copy_configs,
    )
    _S4_AVAILABLE = True
except ImportError:
    _S4_AVAILABLE = False   # s5 core checks (D21c, D26) do not require s4


# measure_G_cross_gf defined here directly (not imported from s4) to avoid
# spawn-method circular import on Windows.  Definition is identical to the
# one in paper6_verification_s4.py — both derive from the §4 canonical formula.
def measure_G_cross_gf(configs_gf_stack, t):
    """
    F8: G_cross^{gf}(t; c) = 8 * Σ_{x,μ} Re[ b_t(x,μ;c) · conj(b_0(x,μ;c)) ]
    where b(x,μ;c) = configs_gf_stack[c, t, x, y, z, μ, 0, 1].

    Parameters
    ----------
    configs_gf_stack : ndarray (N_cfg, N, N, N, N, 4, 2, 2)
    t                : int, target timeslice

    Returns
    -------
    G_values : ndarray (N_cfg,), float64
    """
    b_t = configs_gf_stack[:, t, :, :, :, :, 0, 1]   # (N_cfg, N, N, N, 4)
    b_0 = configs_gf_stack[:, 0, :, :, :, :, 0, 1]   # reference slice
    return 8.0 * np.sum(np.real(b_t * np.conj(b_0)), axis=(1, 2, 3, 4))


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS AND SCOPE DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

# β_eff scopes (CR v1.2 §β_eff Scope Definition)
BETA_SCOPE_A  = [5.0, 6.0, 9.0, 12.0]   # base ensembles, N=6, N_cfg=100
BETA_SCOPE_B  = [20.0]                    # new, N=6 only this run
BETA_SCOPE_D  = [2.0, 2.5, 3.0, 3.5]    # DEFERRED — Group E1

# N values for this run (operator decision: N=6 and N=12 only)
N_RUN        = [6, 12]

# Ensemble sizes
N_CFG        = 100          # per CR §1 minimum
N_STARTS     = 5            # Gribov copy starts (F7)
N_THERM      = 500          # thermalisation sweeps
N_SEP        = 10           # sweeps between stored configs

# Gauge fixing
EPS_GAUGE    = 1e-14
METROPOLIS_EPS = 0.27       # Metropolis step (SU(2) standard; ~50% acceptance)

# D26 pass threshold
CV_THRESHOLD = 0.05         # coefficient of variation < 5% = PASS

# D21b R_min values (from §2 CR, implemented values — not description text)
R_MIN = {5.0: 0.65, 6.0: 0.70, 9.0: 0.80, 12.0: 0.88, 20.0: 0.93}

# D21c extensive scaling tolerance
D21C_TOL = 0.05             # ±5% relative to V(6)/V(12)

# Autocorrelation
TAU_MAX    = 50             # max lag for ρ(τ)

# Parallelism
N_WORKERS  = 12             # CPU cores available; used by Pool in gauge_fix_ensemble


# ═══════════════════════════════════════════════════════════════════════════════
# SEED AND ALGORITHM SELECTION
# ═══════════════════════════════════════════════════════════════════════════════

def ensemble_seed(N, beta_eff):
    """
    Ensemble RNG seed for §5 (non-overlapping with §2–§4).
    §2/§3 used 2000+int(β×100) for N=6, N_cfg=20.
    §5 uses distinct ranges for N=6 and N=12.
    """
    if N == 6:
        return 5000 + int(beta_eff * 100)    # range 5500–5700+
    elif N == 12:
        return 12000 + int(beta_eff * 100)   # range 12500–12700+
    else:
        return N * 1000 + int(beta_eff * 100)


def gf_seed(N, cfg_idx, start):
    """
    Gauge-fixing seed for §5.
    Formula: cfg × 100 + start + N × 10000 + 500000
    For N=6:  cfg×100 + start + 560000   (560000–562000 range)
    For N=12: cfg×100 + start + 620000   (620000–622000 range)
    No overlap with §2/§3 (222000+) or §3 perturb (333000+).
    """
    return cfg_idx * 100 + start + N * 10000 + 500000


def algorithm_and_kmax(N, beta_eff):
    """
    Select gauge-fixing algorithm and max-iteration counts per CR §1.

    Rules (CR v1.2 §1):
      β_eff ≤ 8.0 OR N ≤ 8:
        F5 (SD). k_max = 1000 for N≤8; (N/6)²×800 for N>8.
      β_eff ≥ 9.0 AND N ≥ 12:
        F6 (FA). k_max = 200 × (N/6)^{0.5}.

    Returns: (algo, k_max_SD, k_max_FA)
      Pass both to gauge_fix_and_check_D19; only the relevant one is used.
    """
    if N >= 12:
        algo = 'FA'
        k_fa = int(200 * math.sqrt(N / 6.0))
        k_sd = 10000   # unused but required by function signature
    else:
        algo = 'SD'
        k_fa = 2000    # unused
        if N <= 8:
            k_sd = 1000
        else:
            k_sd = int((N / 6.0) ** 2 * 800)
    return algo, k_sd, k_fa


# ═══════════════════════════════════════════════════════════════════════════════
# MULTIPROCESSING WORKER — must be at module level for pickling
# ═══════════════════════════════════════════════════════════════════════════════

def _gf_single_config(args):
    """
    Best-copy Landau gauge-fix for a single configuration.

    Module-level so multiprocessing.Pool can pickle it on all platforms
    (required for 'spawn' start method on macOS/Windows; also works with
    'fork' on Linux).  Explicit imports inside the function guarantee the
    worker process finds the modules regardless of start method.

    Parameters (passed as a single tuple for Pool.map compatibility)
    ----------
    args : tuple
        (U_raw, N, beta_eff, cfg_idx, N_starts, algo, k_sd, k_fa)
        U_raw    : ndarray (N,N,N,N,4,2,2) — raw link config
        N        : int — lattice side
        beta_eff : float
        cfg_idx  : int — config index (returned unchanged for reassembly)
        N_starts : int — Gribov copy starts
        algo     : str — 'SD' or 'FA'
        k_sd     : int — k_max for SD
        k_fa     : int — k_max for FA

    Returns
    -------
    tuple : (cfg_idx, best_Ugf, FL_raw, best_FL, best_Theta, best_conv, best_k)
    """
    U_raw, N, beta_eff, cfg_idx, N_starts, algo, k_sd, k_fa = args

    # Explicit imports for spawn-safe workers
    from paper6_gauge_fix import gauge_fix_and_check_D19, compute_FL

    FL_raw     = compute_FL(U_raw)
    best_FL    = -1e30
    best_Ugf   = None
    best_Theta = 1.0
    best_conv  = False
    best_k     = k_sd
    last_Ugf   = None   # fallback if no start converges

    for start in range(N_starts):
        # Seed mirrors gf_seed(N, cfg_idx, start) — inlined to avoid import
        seed_val = cfg_idx * 100 + start + N * 10000 + 500_000
        U_gf, _, FL, Theta, k, conv = gauge_fix_and_check_D19(
            U_raw, beta_eff, N,
            algorithm=algo,
            eps_gauge=1e-14,
            k_max_SD=k_sd,
            k_max_FA=k_fa,
            random_seed=seed_val,
        )
        last_Ugf = U_gf
        if conv and FL > best_FL:
            best_FL    = FL
            best_Ugf   = U_gf
            best_Theta = Theta
            best_conv  = conv
            best_k     = k

    if best_Ugf is None:
        # No start converged; use last result as fallback
        best_Ugf  = last_Ugf
        best_FL   = compute_FL(last_Ugf)
        best_conv = False

    return cfg_idx, best_Ugf, FL_raw, best_FL, best_Theta, best_conv, best_k


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — generate_extended_ensemble
# ═══════════════════════════════════════════════════════════════════════════════

def generate_extended_ensemble(N, beta_eff, N_cfg=100, N_therm=500,
                               n_sep=10, n_gribov_starts=5,
                               verbose=True):
    """
    Generate N_cfg gauge configurations on an N^4 lattice at coupling β_eff.
    Applies Metropolis heatbath sweeps with thermalisation stability check.

    Parameters
    ----------
    N           : int,   lattice side length (hypercubic N^4)
    beta_eff    : float, SU(2) gauge coupling
    N_cfg       : int,   configurations to store (≥ 100 per CR)
    N_therm     : int,   thermalisation sweeps (doubled once if unstable)
    n_sep       : int,   sweeps between stored configurations
    n_gribov_starts : int, passed through for documentation; not used here
                    (gauge fixing is separate)
    verbose     : bool,  print progress

    Returns
    -------
    configs    : ndarray (N_cfg, N, N, N, N, 4, 2, 2), complex128
                 Raw (un-gauge-fixed) SU(2) link arrays.
                 Canonical shape per CR v1.2.
    plaquettes : ndarray (N_cfg,), float64
                 Per-configuration plaquette ⟨(1/2)Tr P⟩.
    therm_info : dict
                 Thermalisation diagnostics:
                   'N_therm_actual': int — sweeps actually run
                   'plaq_mean': float   — mean plaquette after thermalisation
                   'plaq_std' : float   — std of thermalisation plaquette
                   'therm_extended': bool — True if N_therm was doubled

    Algorithm:
      (1) Initialise: cold start (identity links) for β≥9; hot start for β<9.
      (2) Run N_therm Metropolis sweeps (METROPOLIS_EPS=0.27).
          Collect plaquette values in the second half of thermalisation.
      (3) Thermalisation stability check:
            running_mean := cumsum(P_second_half) / arange(1, len+1)
            stable := all( |running_mean[i] - mean(P_second_half)| < σ_naive )
            where σ_naive = std(P_second_half) / sqrt(len(P_second_half))
          If not stable: run N_therm more sweeps and proceed.
      (4) Collect N_cfg configurations at n_sep-sweep intervals.
          Measure plaquette on each stored configuration.

    Cold/hot start rules (CR §1):
      β_eff ≥ 9.0 → cold start (identity links, minimal thermalisation needed)
      β_eff < 9.0 → hot start (random Haar-distributed SU(2) links)
      (No cold-vs-hot distinction at intermediate β; 9.0 is the dividing line.)
    """
    ens_seed = ensemble_seed(N, beta_eff)
    rng = np.random.default_rng(ens_seed)

    if verbose:
        print(f"  [gen] N={N} β={beta_eff} N_cfg={N_cfg} "
              f"N_therm={N_therm} n_sep={n_sep} seed={ens_seed}")

    # ── Step 1: Initialise link array ─────────────────────────────────────────
    if beta_eff >= 9.0:
        # Cold start: all links = identity
        L = np.zeros((N, N, N, N, 4, 2, 2), dtype=complex)
        L[..., 0, 0] = 1.0
        L[..., 1, 1] = 1.0
        if verbose:
            print(f"  [gen]   Cold start (β={beta_eff} ≥ 9.0)")
    else:
        # Hot start: random SU(2) links
        hot_seed = int(rng.integers(1_000_000))
        L = random_su2((N, N, N, N, 4), seed=hot_seed)
        if verbose:
            print(f"  [gen]   Hot start (β={beta_eff} < 9.0)")

    # ── Step 2: Thermalisation ────────────────────────────────────────────────
    plaq_therm = []
    for sweep in range(N_therm):
        L, _ = metropolis_sweep(L, beta_eff, METROPOLIS_EPS)
        if sweep >= N_therm // 2:
            plaq_therm.append(plaquette_fn(L))

    # ── Step 3: Stability check ───────────────────────────────────────────────
    plaq_arr  = np.array(plaq_therm)
    plaq_mean = float(np.mean(plaq_arr))
    plaq_std  = float(np.std(plaq_arr, ddof=1))
    sigma_naive = plaq_std / math.sqrt(len(plaq_arr)) if len(plaq_arr) > 1 else 1.0

    running = np.cumsum(plaq_arr) / np.arange(1, len(plaq_arr) + 1)
    stable  = bool(np.all(np.abs(running - plaq_mean) < sigma_naive))
    therm_extended = False

    if not stable:
        if verbose:
            print(f"  [gen]   Thermalisation unstable — extending by {N_therm} sweeps")
        for _ in range(N_therm):
            L, _ = metropolis_sweep(L, beta_eff, METROPOLIS_EPS)
        therm_extended = True

    N_therm_actual = N_therm * (2 if therm_extended else 1)

    if verbose:
        print(f"  [gen]   Thermalisation: {N_therm_actual} sweeps, "
              f"⟨P⟩={plaq_mean:.5f} ± {sigma_naive:.5f} "
              f"{'[EXTENDED]' if therm_extended else '[stable]'}")

    # ── Step 4: Collect configurations ───────────────────────────────────────
    configs    = np.empty((N_cfg, N, N, N, N, 4, 2, 2), dtype=complex)
    plaquettes = np.empty(N_cfg, dtype=float)

    for c in range(N_cfg):
        for _ in range(n_sep):
            L, _ = metropolis_sweep(L, beta_eff, METROPOLIS_EPS)
        configs[c]    = L
        plaquettes[c] = plaquette_fn(L)

    if verbose:
        print(f"  [gen]   Collected {N_cfg} configs. "
              f"⟨P⟩_ensemble={np.mean(plaquettes):.5f}")

    therm_info = {
        'N_therm_actual': N_therm_actual,
        'plaq_mean':      plaq_mean,
        'plaq_std':       plaq_std,
        'therm_extended': therm_extended,
        'plaq_ensemble_mean': float(np.mean(plaquettes)),
        'plaq_ensemble_std':  float(np.std(plaquettes, ddof=1)),
    }
    return configs, plaquettes, therm_info


# ═══════════════════════════════════════════════════════════════════════════════
# GAUGE FIXING PIPELINE (D19–D21b per config; D21c across volumes)
# ═══════════════════════════════════════════════════════════════════════════════

def gauge_fix_ensemble(configs_raw, N, beta_eff, verbose=True):
    """
    Apply best-copy Landau gauge fixing to all N_cfg configurations.
    D19, D21a, D21b checked per configuration.

    Parallelism: the cfg_idx loop is distributed across N_WORKERS processes
    using multiprocessing.Pool.imap_unordered.  Each worker calls
    _gf_single_config (module-level) for one configuration, running all
    N_STARTS gauge-fixing starts internally.  Results arrive out of order
    and are re-indexed via the returned cfg_idx.

    Data movement per config:
      Sent:     U_raw shape (N,N,N,N,4,2,2) ≈ 320 KB (N=6), 5.3 MB (N=12)
      Returned: best_Ugf same shape + scalars
    Total pickle overhead: ~2×N_cfg×config_size; negligible vs GF compute.

    Returns
    -------
    configs_gf : ndarray (N_cfg, N, N, N, N, 4, 2, 2)
    gf_stats   : dict — per-ensemble diagnostics
    """
    algo, k_sd, k_fa = algorithm_and_kmax(N, beta_eff)
    N_cfg = len(configs_raw)
    V     = N ** 4
    R_min = R_MIN.get(beta_eff, 0.60)

    if verbose:
        print(f"  [gf]  N={N} β={beta_eff} algo={algo} "
              f"k_SD={k_sd} k_FA={k_fa} "
              f"N_cfg={N_cfg} workers={N_WORKERS}")

    # Build argument list — one tuple per configuration
    args_list = [
        (configs_raw[i], N, beta_eff, i, N_STARTS, algo, k_sd, k_fa)
        for i in range(N_cfg)
    ]

    # ── Parallel gauge fixing ─────────────────────────────────────────────────
    configs_gf = np.empty((N_cfg, N, N, N, N, 4, 2, 2), dtype=complex)
    FL_raw_arr = np.empty(N_cfg)
    FL_best    = np.empty(N_cfg)
    R_best     = np.empty(N_cfg)
    Theta_best = np.empty(N_cfg)
    D19_pass   = np.zeros(N_cfg, dtype=bool)
    D21a_pass  = np.zeros(N_cfg, dtype=bool)
    D21b_pass  = np.zeros(N_cfg, dtype=bool)
    k_conv_list = [0] * N_cfg

    t_gf = time.time()
    done = 0
    with multiprocessing.Pool(processes=N_WORKERS) as pool:
        for result in pool.imap_unordered(_gf_single_config, args_list,
                                          chunksize=1):
            cfg_idx, best_Ugf, FL_raw, best_FL_c, best_Theta, best_conv, best_k = result
            configs_gf[cfg_idx]  = best_Ugf
            FL_raw_arr[cfg_idx]  = FL_raw
            FL_best[cfg_idx]     = best_FL_c
            R_best[cfg_idx]      = best_FL_c / (8.0 * V)
            Theta_best[cfg_idx]  = best_Theta
            D19_pass[cfg_idx]    = best_conv
            D21a_pass[cfg_idx]   = (best_FL_c > FL_raw)
            D21b_pass[cfg_idx]   = (R_best[cfg_idx] >= R_min)
            k_conv_list[cfg_idx] = best_k
            done += 1
            if verbose and (done % 10 == 0 or done == N_cfg):
                elapsed = time.time() - t_gf
                print(f"  [gf]    {done:3d}/{N_cfg} done  "
                      f"({elapsed:.1f}s,  ~{elapsed/done*(N_cfg-done):.0f}s rem)")

    gf_stats = {
        'FL_best':     FL_best,
        'R_best':      R_best,
        'D19_pass':    D19_pass,
        'D21a_pass':   D21a_pass,
        'D21b_pass':   D21b_pass,
        'Theta_best':  Theta_best,
        'k_conv_mean': float(np.mean(k_conv_list)),
        'D19_frac':    float(np.mean(D19_pass)),
        'D21a_frac':   float(np.mean(D21a_pass)),
        'D21b_frac':   float(np.mean(D21b_pass)),
        'R_mean':      float(np.mean(R_best)),
        'R_std':       float(np.std(R_best, ddof=1)),
    }
    if verbose:
        print(f"  [gf]  Done ({time.time()-t_gf:.1f}s): "
              f"D19={gf_stats['D19_frac']*100:.0f}% "
              f"D21a={gf_stats['D21a_frac']*100:.0f}% "
              f"D21b={gf_stats['D21b_frac']*100:.0f}% "
              f"R_mean={gf_stats['R_mean']:.5f}")
    return configs_gf, gf_stats


def check_D21c(gf_stats_N6, gf_stats_N12, beta_eff):
    """
    D21c: Extensive volume scaling check (CR §2 D21(c)).

    Verifies: mean(F_L^gf; N=6) / mean(F_L^gf; N=12) ≈ V(6)/V(12) = 1/16
    within ±5% relative tolerance.

    F_L is an extensive quantity (sum over V links); this ratio must
    equal V(6)/V(12) = 6^4/12^4 = 1296/20736 = 1/16 modulo O(1/N²)
    finite-volume corrections. A ±5% window accommodates those corrections.

    Parameters
    ----------
    gf_stats_N6  : dict from gauge_fix_ensemble at N=6
    gf_stats_N12 : dict from gauge_fix_ensemble at N=12
    beta_eff     : float (for reporting)

    Returns
    -------
    result : dict
        'actual_ratio'   : float, mean(FL_N6) / mean(FL_N12)
        'expected_ratio' : float, 6**4 / 12**4 = 1/16
        'rel_error'      : float, |actual - expected| / expected
        'passed'         : bool, rel_error < D21C_TOL = 0.05
        'mean_FL_N6'     : float
        'mean_FL_N12'    : float
    """
    mean_FL_N6  = float(np.mean(gf_stats_N6['FL_best']))
    mean_FL_N12 = float(np.mean(gf_stats_N12['FL_best']))
    expected    = (6 ** 4) / (12 ** 4)     # = 1/16 = 0.0625
    actual      = mean_FL_N6 / mean_FL_N12 if mean_FL_N12 != 0 else float('nan')
    rel_error   = abs(actual - expected) / expected

    return {
        'beta_eff':       beta_eff,
        'actual_ratio':   actual,
        'expected_ratio': expected,
        'rel_error':      rel_error,
        'tolerance':      D21C_TOL,
        'passed':         bool(rel_error < D21C_TOL),
        'mean_FL_N6':     mean_FL_N6,
        'mean_FL_N12':    mean_FL_N12,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — compute_autocorrelation
# ═══════════════════════════════════════════════════════════════════════════════

def compute_autocorrelation(O_array, tau_max=TAU_MAX):
    """
    Integrated autocorrelation time τ_int for observable O_array.

    Algorithm (CR v1.2 §2):
      mu  = mean(O_array)
      var = np.var(O_array, ddof=0)
      rho[tau] = mean( (O[:N-tau]-mu)*(O[tau:]-mu) ) / var
      tau_cut  = first tau where |rho[tau]| < 2/sqrt(N)
      tau_int  = 0.5 + sum(rho[1:tau_cut])

    If var ≈ 0 (constant observable): tau_int = 0.5 (trivial decorrelation).

    Parameters
    ----------
    O_array : ndarray (N_cfg,), observable values on sequential configs
    tau_max : int, maximum lag to compute

    Returns
    -------
    tau_int : float
    rho     : ndarray, normalised ACF ρ(τ) for τ = 1, …, tau_cut
    tau_cut : int
    """
    O   = np.asarray(O_array, dtype=float)
    N   = len(O)
    mu  = float(np.mean(O))
    var = float(np.var(O, ddof=0))

    if var < 1e-30 * (abs(mu) + 1.0):
        # Constant observable: trivially uncorrelated
        return 0.5, np.array([]), 0

    threshold = 2.0 / math.sqrt(N)
    tau_max_actual = min(tau_max, N // 4)
    rho_list = []
    tau_cut = tau_max_actual   # default: never dropped below threshold

    for tau in range(1, tau_max_actual + 1):
        c_tau = float(np.mean((O[:N - tau] - mu) * (O[tau:] - mu))) / var
        rho_list.append(c_tau)
        if abs(c_tau) < threshold:
            tau_cut = tau
            break

    rho     = np.array(rho_list)
    tau_int = 0.5 + float(np.sum(rho[:-1]))   # exclude the cut point itself
    return tau_int, rho, tau_cut


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — jackknife_error
# ═══════════════════════════════════════════════════════════════════════════════

def jackknife_error(O_array):
    """
    Jackknife error estimate for the ensemble mean.

    Algorithm (CR v1.2 §3):
      N        = len(O_array)
      total    = sum(O_array)
      mean_O   = total / N
      jk_means = (total - O_array) / (N - 1)     [leave-one-out means]
      sigma_JK = sqrt( (N-1)/N * sum((jk_means - mean_O)²) )

    Equivalent to: sigma_JK = sqrt((N-1)/N) * std(jk_means) * sqrt(N)
                             = std_of_jk_distribution * sqrt(N-1)
    The factor (N-1) vs N difference from the standard error propagates
    the bias-corrected jackknife variance.

    Parameters
    ----------
    O_array : ndarray (N,), observable values

    Returns
    -------
    mean_O  : float, ensemble mean
    sigma_JK: float, jackknife error on the mean
    """
    O      = np.asarray(O_array, dtype=float)
    N      = len(O)
    total  = float(np.sum(O))
    mean_O = total / N
    jk_means = (total - O) / (N - 1)
    sigma_JK = math.sqrt((N - 1) / N * float(np.sum((jk_means - mean_O) ** 2)))
    return mean_O, sigma_JK


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — fv_scaling_fit (appears before D26 in CR §7)
# ═══════════════════════════════════════════════════════════════════════════════

def fv_scaling_fit(D_vs_N, sigma_vs_N, N_values, t):
    """
    Weighted least-squares fit of D_{W±}(t; N) = D_inf + a / N² (CR §7).

    Model:  D(N) = D_inf + a / N²   (linear in two parameters)
    Design: X = [[1, 1/N_i²] for each N_i], weights W_ii = 1/sigma_i²

    Computation:
      Normal equations: (X^T W X) θ = X^T W d
      θ = [D_inf, a_coeff]
      Covariance: cov = (X^T W X)^{-1}
      χ² = sum( (D_i - D_inf - a/N_i²)² / sigma_i² )
      χ²/dof: dof = len(N_values) - 2

    DEGENERACY NOTE:
      With only 2 data points and 2 parameters (this run: N ∈ {6, 12}),
      dof = 0 and χ² = 0 exactly (perfect fit through 2 points).
      chi2_dof is returned as float('nan') and STATUS = 'DEGENERATE'.
      The parameter covariance is still well-defined and meaningful
      (propagation of measurement errors σ_i to parameter uncertainties).

    Parameters
    ----------
    D_vs_N    : array-like, D_{W±}(t; N_i) for each N_i
    sigma_vs_N: array-like, jackknife errors sigma_i for each N_i
    N_values  : list of int, lattice sizes
    t         : int, timeslice (for labelling only)

    Returns
    -------
    dict with:
      D_inf, sigma_D_inf : float — infinite-volume limit and error
      a_coeff, sigma_a   : float — 1/N² coefficient and error
      chi2_dof           : float — χ²/dof (nan if dof=0)
      delta_FV           : float — |a_coeff| / 16² (residual at N=16)
      status             : str — 'ACCEPT', 'TENSION', or 'DEGENERATE'
      chi2               : float — raw χ²
      dof                : int
    """
    D_arr     = np.asarray(D_vs_N,     dtype=float)
    sig_arr   = np.asarray(sigma_vs_N, dtype=float)
    N_arr     = np.asarray(N_values,   dtype=float)
    n_pts     = len(N_arr)

    # Design matrix: columns [1, 1/N²]
    X = np.column_stack([np.ones(n_pts), 1.0 / N_arr ** 2])

    # Weights: 1/sigma²; guard against zero sigma
    sig_safe = np.where(sig_arr > 1e-30, sig_arr, 1e-30)
    W_diag   = 1.0 / sig_safe ** 2

    # Weighted normal equations: (X^T W X) θ = X^T W D
    A   = X.T @ (W_diag[:, None] * X)      # 2×2 matrix
    b_v = X.T @ (W_diag * D_arr)            # 2-vector
    theta = np.linalg.solve(A, b_v)
    D_inf, a_coeff = float(theta[0]), float(theta[1])

    # Covariance matrix
    cov = np.linalg.inv(A)
    sigma_D_inf = float(math.sqrt(max(cov[0, 0], 0.0)))
    sigma_a     = float(math.sqrt(max(cov[1, 1], 0.0)))

    # χ²
    residuals = D_arr - D_inf - a_coeff / N_arr ** 2
    chi2      = float(np.sum((residuals / sig_safe) ** 2))
    dof       = n_pts - 2

    if dof <= 0:
        chi2_dof = float('nan')
        status   = 'DEGENERATE'
    elif chi2 / dof < 3.0:
        chi2_dof = chi2 / dof
        status   = 'ACCEPT'
    else:
        chi2_dof = chi2 / dof
        status   = 'TENSION'

    delta_FV = abs(a_coeff) / (16.0 ** 2)

    return {
        't':           t,
        'D_inf':       D_inf,
        'sigma_D_inf': sigma_D_inf,
        'a_coeff':     a_coeff,
        'sigma_a':     sigma_a,
        'chi2':        chi2,
        'dof':         dof,
        'chi2_dof':    chi2_dof,
        'delta_FV':    delta_FV,
        'status':      status,
        'N_values':    list(N_values),
        'D_vs_N':      list(D_arr),
        'sigma_vs_N':  list(sig_arr),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — run_D26_phase1
# ═══════════════════════════════════════════════════════════════════════════════

def run_D26_phase1(configs_gf, autocorr_results, n_sep=N_SEP):
    """
    D26-PHASE1: Statistical convergence of ⟨G_cross^{gf}(t=1)⟩ (CR §5).

    Scope: Scopes A+B+C (β ∈ {5.0,6.0,9.0,12.0,20.0}, N ∈ {6,12}).

    For each (N, β_eff) in configs_gf:
      G_array   = measure_G_cross_gf(configs_gf[(N,β)], t=1)  → (N_cfg,)
      mean_G, σ_JK = jackknife_error(G_array)
      CV        = σ_JK / |mean_G|
      running   = cumsum(G_array) / arange(1, N_cfg+1)
      stable    = all( |running[n] - mean_G| < 2*σ_JK  for n ≥ N_cfg//2 )

    PASS: CV < CV_THRESHOLD AND stable = True for ALL entries.
    FAIL: CV ≥ CV_THRESHOLD OR stable = False for ANY entry.

    No PARTIAL PASS in Phase 1 (Scope D absent; CR §5 §NO PARTIAL PASS).

    Parameters
    ----------
    configs_gf      : dict (N, beta_eff) → ndarray (N_cfg,N,N,N,N,4,2,2)
    autocorr_results: dict (N, beta_eff) → {'tau_int':…, 'tau_cut':…, …}
    n_sep           : int, sweeps between configs (for n_sep/2 comparison)

    Returns
    -------
    results  : dict (N, beta_eff) → per-entry result dict
    D26_pass : bool, overall PHASE1 verdict
    """
    results = {}

    for (N, beta_eff), U_stack in sorted(configs_gf.items()):
        N_cfg_local = U_stack.shape[0]

        G_array = measure_G_cross_gf(U_stack, t=1)
        mean_G, sig_JK = jackknife_error(G_array)

        CV = (sig_JK / abs(mean_G)) if abs(mean_G) > 1e-10 else float('inf')

        running  = np.cumsum(G_array) / np.arange(1, N_cfg_local + 1)
        stable   = bool(np.all(
            np.abs(running[N_cfg_local // 2:] - mean_G) < 2.0 * sig_JK
        ))

        entry_pass = (CV < CV_THRESHOLD) and stable

        # Autocorrelation check: τ_int vs n_sep/2
        tau_int = autocorr_results.get((N, beta_eff), {}).get('tau_int', 0.5)
        nsep_warn = tau_int > n_sep / 2.0

        results[(N, beta_eff)] = {
            'N':           N,
            'beta_eff':    beta_eff,
            'N_cfg':       N_cfg_local,
            'mean_G':      float(mean_G),
            'sigma_JK':    float(sig_JK),
            'CV':          float(CV),
            'stable':      stable,
            'tau_int':     float(tau_int),
            'nsep_warn':   nsep_warn,
            'pass':        entry_pass,
        }

        flag = '' if entry_pass else '  *** FAIL ***'
        warn = '  [τ_int > n_sep/2 — regenerate recommended]' if nsep_warn else ''
        print(f"  D26  N={N:2d}  β={beta_eff:5.1f}:  "
              f"mean={mean_G:9.3f}  σ={sig_JK:.3f}  "
              f"CV={CV:.4f}  stable={stable}  τ={tau_int:.2f}"
              f"{flag}{warn}")

    D26_pass = all(r['pass'] for r in results.values())
    return results, D26_pass


# ═══════════════════════════════════════════════════════════════════════════════
# PLAQUETTE COUPLING TABLE (CR §8)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_coupling_table(plaquette_dict):
    """
    Compute plaquette / coupling data for Appendix B Table (CR §8).

    For each (N, beta_eff):
      mean_P    = mean(plaquettes)
      g2_meas   = 4 * (1 - mean_P)
      g2_pert   = 4 / beta_eff
      delta_rel = |g2_meas - g2_pert| / g2_pert

    Flag if delta_rel > 0.10 at β=20 (expected ~ 0.01–0.05 at WC).

    Parameters
    ----------
    plaquette_dict : dict (N, beta_eff) → ndarray (N_cfg,)

    Returns
    -------
    table : list of dicts, one per (N, beta_eff)
    """
    table = []
    for (N, beta_eff), plaq in sorted(plaquette_dict.items()):
        mean_P    = float(np.mean(plaq))
        std_P     = float(np.std(plaq, ddof=1))
        g2_meas   = 4.0 * (1.0 - mean_P)
        g2_pert   = 4.0 / beta_eff
        delta_rel = abs(g2_meas - g2_pert) / g2_pert
        flag = ''
        if beta_eff == 20.0 and delta_rel > 0.10:
            flag = 'ANOMALOUS'
        elif beta_eff == 20.0:
            flag = 'OK'
        table.append({
            'N':        N,
            'beta_eff': beta_eff,
            'mean_P':   mean_P,
            'std_P':    std_P,
            'g2_meas':  g2_meas,
            'g2_pert':  g2_pert,
            'delta_rel': delta_rel,
            'flag':     flag,
        })
    return table


# ═══════════════════════════════════════════════════════════════════════════════
# FV SCALING ANALYSIS (CR §7)
# ═══════════════════════════════════════════════════════════════════════════════

def run_fv_scaling(configs_gf, plaquette_dict, beta_list, N_list):
    """
    Finite-volume scaling fit of D_{W±}(t; N) over available N values.

    For each (β_eff, t), collect D_{W±}(t; N_i) and σ_i via jackknife,
    then fit D(N) = D_inf + a/N².

    With N_list = [6, 12] (this run): DOF = 0 → DEGENERATE for all fits.

    Parameters
    ----------
    configs_gf     : dict (N, beta_eff) → stacked gauge-fixed arrays
    plaquette_dict : dict (N, beta_eff) → plaquette arrays
    beta_list      : list of float — β values to fit (Scopes A only: no β=20)
    N_list         : list of int   — N values available

    Returns
    -------
    fv_results : dict (beta_eff, t) → fv_scaling_fit output dict
    """
    fv_results = {}

    for beta_eff in beta_list:
        # Collect D_{W±}(t) and errors for each N
        # Using Method A: D_A[t] = G_avg[t] * beta / (32 * N^3)
        # (CR §3 jackknife propagation formula: G * beta / (32*N^3))
        D_by_N     = {}
        sigma_by_N = {}

        for N_val in N_list:
            key = (N_val, beta_eff)
            if key not in configs_gf:
                continue
            U_stack = configs_gf[key]
            N_t     = N_val    # hypercubic
            V_sp    = N_val ** 3

            for t in range(1, N_t // 2 + 1):
                G_arr = measure_G_cross_gf(U_stack, t=t)
                # Propagate to D_A via jackknife
                D_arr = G_arr * beta_eff / (32.0 * V_sp)
                mean_D, sig_D = jackknife_error(D_arr)
                if t not in D_by_N:
                    D_by_N[t]     = {}
                    sigma_by_N[t] = {}
                D_by_N[t][N_val]     = mean_D
                sigma_by_N[t][N_val] = sig_D

        # Run fit for each timeslice
        for t, D_dict in sorted(D_by_N.items()):
            avail_N = sorted(D_dict.keys())
            if len(avail_N) < 2:
                continue
            D_arr_t   = np.array([D_dict[n]     for n in avail_N])
            sig_arr_t = np.array([sigma_by_N[t][n] for n in avail_N])
            # Guard: replace zero sigma with small value
            sig_arr_t = np.where(sig_arr_t > 1e-30, sig_arr_t, 1e-10)

            fit = fv_scaling_fit(D_arr_t, sig_arr_t, avail_N, t)
            fv_results[(beta_eff, t)] = fit

    return fv_results


# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT FILE WRITERS
# ═══════════════════════════════════════════════════════════════════════════════

def write_coupling_table(table, fname='paper6_coupling_table.txt'):
    lines = []
    w = lines.append
    w("# Paper 6 — Plaquette / Coupling Table (Appendix B)")
    w("# Generated by paper6_verification_s5.py")
    w("# CR §8: Scope A+B+C ensembles (Scope D appended when generated)")
    w("#")
    w(f"# {'beta_eff':>8}  {'N':>4}  {'mean_P':>10}  {'std_P':>9}  "
      f"{'g2_meas':>9}  {'g2_pert':>9}  {'delta_rel':>10}  {'flag'}")
    for r in table:
        w(f"  {r['beta_eff']:>8.1f}  {r['N']:>4d}  {r['mean_P']:>10.7f}  "
          f"{r['std_P']:>9.7f}  {r['g2_meas']:>9.6f}  {r['g2_pert']:>9.6f}  "
          f"{r['delta_rel']:>10.6f}  {r['flag']}")
    with open(fname, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  Written: {fname}")


def write_fv_results(fv_results, fname='paper6_fv_scaling_results.txt'):
    lines = []
    w = lines.append
    w("# Paper 6 — Finite-Volume Scaling Results")
    w("# Generated by paper6_verification_s5.py")
    w("# CR §7: D(N) = D_inf + a/N² fit")
    w("# NOTE: N_values=[6,12] only (this run); DOF=0 everywhere → DEGENERATE.")
    w("# chi2_dof = nan for all rows. ACCEPT/TENSION suspended.")
    w("# Full 4-volume fit (N=6,8,12,16) deferred pending N=8,16 data.")
    w("#")
    w(f"# {'beta_eff':>8}  {'t':>3}  {'D_inf':>12}  {'sigma_Dinf':>12}  "
      f"{'a_coeff':>12}  {'sigma_a':>10}  {'chi2_dof':>10}  "
      f"{'delta_FV':>10}  {'N_values':>12}  {'STATUS'}")
    for (beta_eff, t), r in sorted(fv_results.items()):
        chi2_str = f"{r['chi2_dof']:.4f}" if not math.isnan(r['chi2_dof']) else 'nan'
        w(f"  {beta_eff:>8.1f}  {t:>3d}  {r['D_inf']:>12.7f}  "
          f"{r['sigma_D_inf']:>12.7f}  {r['a_coeff']:>12.5f}  "
          f"{r['sigma_a']:>10.5f}  {chi2_str:>10}  "
          f"{r['delta_FV']:>10.7f}  "
          f"{str(r['N_values']):>12}  {r['status']}")
    with open(fname, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  Written: {fname}")


def write_summary(d26_results, D26_pass, d21c_results, autocorr_all,
                  gf_stats_all, coupling_table, fv_results,
                  elapsed, fname='paper6_verification_summary_s5.txt'):
    lines = []
    w = lines.append

    w("══════════════════════════════════════════════════════════════════════════")
    w("PAPER 6 — VERIFICATION SUMMARY")
    w("§5 — Statistical Convergence of ⟨G_cross^{gf}⟩ with N_cfg")
    w("     CR: Paper6_CodeRequirements_S5.txt [v1.2]")
    w("══════════════════════════════════════════════════════════════════════════")
    w("")
    d21c_all_pass = all(r['passed'] for r in d21c_results.values())
    w(f"STATUS:")
    w(f"  D21c : {'PASS' if d21c_all_pass else 'FAIL'}  "
      f"(extensive volume scaling; DEFERRED since §2, discharged here)")
    w(f"  D26-PHASE1 : {'PASS' if D26_pass else 'FAIL'}  "
      f"(CV < {CV_THRESHOLD} AND stable for all Scope A+B+C entries)")
    w(f"  D26-PHASE2 : DEFERRED  (Scope D; Group E1 not yet generated)")
    w(f"  FV scaling : DEGENERATE  (N∈{{6,12}} only; DOF=0)")
    w("")

    # ── Run parameters ─────────────────────────────────────────────────────────
    w("──────────────────────────────────────────────────────────────────────────")
    w("RUN PARAMETERS")
    w("──────────────────────────────────────────────────────────────────────────")
    w(f"  N values: {N_RUN}  (N=8, N=16 deferred — operator decision)")
    w(f"  N_cfg={N_CFG}  N_starts={N_STARTS}  N_therm={N_THERM}  n_sep={N_SEP}")
    w(f"  eps_gauge={EPS_GAUGE:.0e}  CV_threshold={CV_THRESHOLD}")
    w(f"  Scope A: N=6,  β∈{BETA_SCOPE_A}")
    w(f"  Scope B: N=6,  β=20.0")
    w(f"  Scope C: N=12, β∈{BETA_SCOPE_A}  (N=8,16 deferred)")
    w(f"  Scope D: DEFERRED  β∈{BETA_SCOPE_D}")
    w(f"  Seed scheme: N=6 → 5000+int(β×100); N=12 → 12000+int(β×100)")
    w(f"  Wall time: {elapsed:.1f}s")
    w("")

    # ── D21c ───────────────────────────────────────────────────────────────────
    w("──────────────────────────────────────────────────────────────────────────")
    w("D21c — EXTENSIVE VOLUME SCALING (DISCHARGED FROM §2 DEFERRAL)")
    w("──────────────────────────────────────────────────────────────────────────")
    w("")
    w("  Check: mean(F_L^gf; N=6) / mean(F_L^gf; N=12) ≈ V(6)/V(12) = 1/16")
    w(f"  Tolerance: ±{D21C_TOL*100:.0f}% relative")
    w(f"  Expected: 6^4/12^4 = {6**4}/{12**4} = {6**4/12**4:.6f}")
    w("")
    w(f"  {'beta':>6}  {'mean_FL_N6':>12}  {'mean_FL_N12':>13}  "
      f"{'actual':>10}  {'expected':>10}  {'rel_err':>9}  {'D21c'}")
    w("  " + "─" * 72)
    for beta_eff, r in sorted(d21c_results.items()):
        pf = 'PASS' if r['passed'] else 'FAIL'
        w(f"  {beta_eff:>6.1f}  {r['mean_FL_N6']:>12.3f}  {r['mean_FL_N12']:>13.3f}  "
          f"{r['actual_ratio']:>10.6f}  {r['expected_ratio']:>10.6f}  "
          f"{r['rel_error']:>9.5f}  {pf}")
    w(f"  D21c OVERALL: {'PASS' if d21c_all_pass else 'FAIL'}")
    w("")

    # ── Gauge fixing quality ───────────────────────────────────────────────────
    w("──────────────────────────────────────────────────────────────────────────")
    w("GAUGE FIXING QUALITY (D19, D21a, D21b) — §5 ensembles")
    w("──────────────────────────────────────────────────────────────────────────")
    w("")
    w(f"  {'N':>4}  {'beta':>6}  {'D19%':>6}  {'D21a%':>6}  {'D21b%':>6}  "
      f"{'R_mean':>8}  {'R_std':>7}  {'k_conv':>7}")
    w("  " + "─" * 60)
    for (N, beta_eff), st in sorted(gf_stats_all.items()):
        w(f"  {N:>4d}  {beta_eff:>6.1f}  "
          f"{st['D19_frac']*100:>6.1f}  {st['D21a_frac']*100:>6.1f}  "
          f"{st['D21b_frac']*100:>6.1f}  "
          f"{st['R_mean']:>8.5f}  {st['R_std']:>7.5f}  {st['k_conv_mean']:>7.1f}")
    w("")

    # ── Autocorrelation ────────────────────────────────────────────────────────
    w("──────────────────────────────────────────────────────────────────────────")
    w("AUTOCORRELATION ANALYSIS (Appendix B Table B.2)")
    w("──────────────────────────────────────────────────────────────────────────")
    w("")
    w(f"  Observable: G_cross^{{gf}}(t=1; c)")
    w(f"  n_sep={N_SEP}; τ_int > {N_SEP/2} triggers regeneration warning")
    w("")
    w(f"  {'N':>4}  {'beta':>6}  {'tau_int':>9}  {'tau_cut':>8}  {'nsep_ok':>8}")
    w("  " + "─" * 44)
    for (N, beta_eff), ac in sorted(autocorr_all.items()):
        nsep_ok = 'OK' if ac['tau_int'] <= N_SEP / 2 else 'WARN'
        w(f"  {N:>4d}  {beta_eff:>6.1f}  {ac['tau_int']:>9.3f}  "
          f"{ac['tau_cut']:>8d}  {nsep_ok:>8}")
    w("")

    # ── D26-PHASE1 ─────────────────────────────────────────────────────────────
    w("──────────────────────────────────────────────────────────────────────────")
    w("D26-PHASE1 — STATISTICAL CONVERGENCE")
    w("──────────────────────────────────────────────────────────────────────────")
    w("")
    w(f"  Scope: A+B+C — β∈{{5.0,6.0,9.0,12.0,20.0}}, N∈{{{','.join(map(str,N_RUN))}}}")
    w(f"  Pass condition: CV < {CV_THRESHOLD} AND stable=True for ALL entries")
    w(f"  No PARTIAL PASS in Phase 1 (Scope D absent)")
    w("")
    w(f"  {'N':>4}  {'beta':>6}  {'mean_G':>10}  {'sigma_JK':>9}  "
      f"{'CV':>8}  {'stable':>7}  {'tau_int':>8}  {'D26'}")
    w("  " + "─" * 72)
    for (N, beta_eff), r in sorted(d26_results.items()):
        pf = 'PASS' if r['pass'] else 'FAIL'
        warn = ' [τ>n_sep/2]' if r['nsep_warn'] else ''
        w(f"  {N:>4d}  {beta_eff:>6.1f}  {r['mean_G']:>10.3f}  "
          f"{r['sigma_JK']:>9.4f}  {r['CV']:>8.5f}  "
          f"{str(r['stable']):>7}  {r['tau_int']:>8.3f}  {pf}{warn}")
    n_pass = sum(r['pass'] for r in d26_results.values())
    n_total = len(d26_results)
    w(f"  D26-PHASE1: {n_pass}/{n_total} PASS → {'PASS' if D26_pass else 'FAIL'}")
    w("")
    w("  D26-PHASE2 (DEFERRED):")
    w("    Scope D: β∈{2.0,2.5,3.0,3.5} not yet generated (Group E1).")
    w("    Expected outcome when generated: PARTIAL PASS at β≤2.5")
    w("    (large Gribov variation; N_cfg increase recommended before §7).")
    w("")

    # ── FV scaling ─────────────────────────────────────────────────────────────
    w("──────────────────────────────────────────────────────────────────────────")
    w("FINITE-VOLUME SCALING (CR §7) — 2-POINT DETERMINATION")
    w("──────────────────────────────────────────────────────────────────────────")
    w("")
    w("  NOTE: With N∈{6,12} only, DOF=0 for all fits. chi2/dof undefined.")
    w("  Full 4-volume fit (N=6,8,12,16) deferred pending N=8,16 data.")
    w("  Values reported are exact 2-point determinations, not tested fits.")
    w("")
    w(f"  {'beta':>6}  {'t':>3}  {'D_inf':>12}  {'sigma_Dinf':>12}  "
      f"{'a_coeff':>10}  {'delta_FV':>10}  {'status'}")
    w("  " + "─" * 68)
    for (beta_eff, t), r in sorted(fv_results.items()):
        w(f"  {beta_eff:>6.1f}  {t:>3d}  {r['D_inf']:>12.7f}  "
          f"{r['sigma_D_inf']:>12.7f}  {r['a_coeff']:>10.5f}  "
          f"{r['delta_FV']:>10.7f}  {r['status']}")
    w("")

    # ── Cumulative registry ────────────────────────────────────────────────────
    w("──────────────────────────────────────────────────────────────────────────")
    w("CUMULATIVE CHECK REGISTRY (D1 through D26)")
    w("──────────────────────────────────────────────────────────────────────────")
    w("")
    w("  S1-9  D1-D18       Paper 5   85/85    PASS")
    w("  S2    D19          Paper 6  400/400   PASS")
    w("  S2    D20          Paper 6  400/400   PASS")
    w("  S2    D21a         Paper 6  400/400   PASS")
    w("  S2    D21b         Paper 6  400/400   PASS")
    d21c_str = 'PASS' if d21c_all_pass else 'FAIL'
    w(f"  S5    D21c         Paper 6            {d21c_str}  "
      f"[discharged from §2 deferral; N=6 vs N=12]")
    w("  S3    D22          Paper 6  100/100   PASS  (INVERTED)")
    w("  S3    D23a         Paper 6  100/100   PASS")
    w("  S3    D23b-strict  Paper 6    0/100   FAIL  [FINDING-3]")
    w("  S3    D23b-rel     Paper 6  100/100   PASS")
    w("  S4    D24          Paper 6    5/5     PASS  (at FP floor)")
    w("  S4    D25 b=12.0   Paper 6            PARTIAL PASS  [FINDING-4]")
    w("  S4    D25 b=20.0   Paper 6            PARTIAL PASS  [FINDING-4]")
    d26_str = 'PASS' if D26_pass else 'FAIL'
    w(f"  S5    D26-PHASE1   Paper 6   {n_pass}/{n_total}     {d26_str}")
    w("  S5    D26-PHASE2   Paper 6            DEFERRED  [Group E1]")
    w("")
    w("  NEXT AVAILABLE CHECK ID: D27 (assigned to §7)")
    w("")
    w("══════════════════════════════════════════════════════════════════════════")
    w("END OF §5 VERIFICATION SUMMARY")
    w("══════════════════════════════════════════════════════════════════════════")

    with open(fname, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  Written: {fname}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN BLOCK
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("═" * 80)
    print("PAPER 6 — §5 STATISTICAL CONVERGENCE  (D21c, D26-PHASE1)")
    print(f"  N values this run : {N_RUN}")
    print(f"  N_cfg={N_CFG}  N_starts={N_STARTS}  N_therm={N_THERM}  n_sep={N_SEP}")
    print(f"  Scope A: N=6,  β∈{BETA_SCOPE_A}")
    print(f"  Scope B: N=6,  β={BETA_SCOPE_B}")
    print(f"  Scope C: N=12, β∈{BETA_SCOPE_A}  (N=8,16 deferred)")
    print(f"  Scope D: DEFERRED  β∈{BETA_SCOPE_D}")
    print("═" * 80)

    t_start = time.time()

    # ── Storage ────────────────────────────────────────────────────────────────
    configs_raw_all  = {}   # (N, beta) → raw configs
    configs_gf_all   = {}   # (N, beta) → gauge-fixed configs
    plaquette_all    = {}   # (N, beta) → plaquette array
    gf_stats_all     = {}   # (N, beta) → gauge fixing statistics
    autocorr_all     = {}   # (N, beta) → autocorrelation info
    d21c_results     = {}   # beta → D21c result

    # ── Build ensemble generation targets in priority order ────────────────────
    # Priority 1: Scope B (β=20.0, N=6)
    # Priority 2: Scope A (base β, N=6)
    # Priority 3: Scope C partial (base β, N=12)
    generation_targets = []
    for beta in BETA_SCOPE_B:
        generation_targets.append((6, beta, 'B'))
    for beta in BETA_SCOPE_A:
        generation_targets.append((6, beta, 'A'))
    for beta in BETA_SCOPE_A:
        generation_targets.append((12, beta, 'C'))

    # ── (a)–(b): Generate ensembles and apply D19-D21a/b pipeline ─────────────
    for (N, beta_eff, scope) in generation_targets:
        print(f"\n{'─'*80}")
        print(f"  Scope {scope}: N={N}  β={beta_eff}")
        print(f"{'─'*80}")

        # Generate raw ensemble
        configs_raw, plaquettes, therm_info = generate_extended_ensemble(
            N=N, beta_eff=beta_eff, N_cfg=N_CFG,
            N_therm=N_THERM, n_sep=N_SEP
        )
        configs_raw_all[(N, beta_eff)] = configs_raw
        plaquette_all[(N, beta_eff)]   = plaquettes

        # Gauge fix (D19, D21a, D21b per config)
        configs_gf, gf_stats = gauge_fix_ensemble(
            configs_raw, N=N, beta_eff=beta_eff
        )
        configs_gf_all[(N, beta_eff)] = configs_gf
        gf_stats_all[(N, beta_eff)]   = gf_stats

        # Assert D19 pass rate acceptable (>= 95%)
        if gf_stats['D19_frac'] < 0.95:
            print(f"  *** WARNING: D19 pass rate {gf_stats['D19_frac']*100:.1f}% "
                  f"< 95% at N={N}, β={beta_eff} ***")

        # D21b failure report
        if gf_stats['D21b_frac'] < 1.0:
            fail_n = int((1 - gf_stats['D21b_frac']) * N_CFG)
            print(f"  *** WARNING: D21b fails for {fail_n} configs: "
                  f"R_mean={gf_stats['R_mean']:.5f} "
                  f"R_min={R_MIN.get(beta_eff, 0.60):.2f} ***")

    # ── (c): D21c — extensive volume scaling ──────────────────────────────────
    print(f"\n{'─'*80}")
    print("  D21c — EXTENSIVE VOLUME SCALING")
    print(f"{'─'*80}")
    for beta_eff in BETA_SCOPE_A:
        key6  = (6,  beta_eff)
        key12 = (12, beta_eff)
        if key6 in gf_stats_all and key12 in gf_stats_all:
            r21c = check_D21c(gf_stats_all[key6], gf_stats_all[key12], beta_eff)
            d21c_results[beta_eff] = r21c
            pf = 'PASS' if r21c['passed'] else 'FAIL'
            print(f"  β={beta_eff}: ratio={r21c['actual_ratio']:.6f}  "
                  f"expected={r21c['expected_ratio']:.6f}  "
                  f"rel_err={r21c['rel_error']:.5f}  → {pf}")
        else:
            print(f"  β={beta_eff}: SKIP (missing N=6 or N=12 data)")

    # ── (d): Autocorrelation — check τ_int vs n_sep/2 ─────────────────────────
    print(f"\n{'─'*80}")
    print("  AUTOCORRELATION ANALYSIS")
    print(f"{'─'*80}")
    regenerate_flags = {}
    for (N, beta_eff), U_stack in sorted(configs_gf_all.items()):
        G_array = measure_G_cross_gf(U_stack, t=1)
        tau_int, rho, tau_cut = compute_autocorrelation(G_array)
        nsep_warn = tau_int > N_SEP / 2.0
        autocorr_all[(N, beta_eff)] = {
            'tau_int': tau_int,
            'tau_cut': tau_cut,
            'rho_len': len(rho),
        }
        flag = '  [τ > n_sep/2 — regenerate with n_sep=' \
               f'{2*math.ceil(tau_int)}]' if nsep_warn else ''
        print(f"  N={N:2d}  β={beta_eff:5.1f}: τ_int={tau_int:.3f}  "
              f"τ_cut={tau_cut}  n_sep/2={N_SEP/2:.1f}{flag}")
        if nsep_warn:
            regenerate_flags[(N, beta_eff)] = 2 * math.ceil(tau_int)

    if regenerate_flags:
        print(f"\n  *** REGENERATION RECOMMENDED for: {list(regenerate_flags.keys())}")
        print("  *** Run with updated n_sep values; re-run §5 for those ensembles.")

    # ── (e): D26-PHASE1 ───────────────────────────────────────────────────────
    print(f"\n{'─'*80}")
    print("  D26-PHASE1 — STATISTICAL CONVERGENCE")
    print(f"{'─'*80}")
    d26_results, D26_pass = run_D26_phase1(configs_gf_all, autocorr_all)
    print(f"\n  D26-PHASE1 RESULT: {'PASS' if D26_pass else 'FAIL'}")

    # ── (f): Plaquette coupling table ─────────────────────────────────────────
    coupling_table = compute_coupling_table(plaquette_all)

    # ── (g): Finite-volume scaling fit ────────────────────────────────────────
    print(f"\n{'─'*80}")
    print("  FINITE-VOLUME SCALING (2-point, DEGENERATE)")
    print(f"{'─'*80}")
    # Only Scopes A+C (not β=20 unless all N available)
    fv_beta_list = BETA_SCOPE_A   # β∈{5,6,9,12} with N=6 and N=12
    fv_results = run_fv_scaling(configs_gf_all, plaquette_all,
                                fv_beta_list, N_RUN)
    for (beta_eff, t), r in sorted(fv_results.items()):
        print(f"  β={beta_eff}  t={t}: D_inf={r['D_inf']:.6f}±{r['sigma_D_inf']:.6f}"
              f"  a={r['a_coeff']:.5f}  δ_FV={r['delta_FV']:.6f}  {r['status']}")

    # ── (h)–(k): Write output files ───────────────────────────────────────────
    elapsed = time.time() - t_start
    write_coupling_table(coupling_table)
    write_fv_results(fv_results)
    write_summary(d26_results, D26_pass, d21c_results, autocorr_all,
                  gf_stats_all, coupling_table, fv_results, elapsed)

    print(f"\n  Total wall time: {elapsed:.1f}s")
    print(f"\n  Output files:")
    print(f"    paper6_verification_summary_s5.txt")
    print(f"    paper6_coupling_table.txt")
    print(f"    paper6_fv_scaling_results.txt")

    return D26_pass, all(r['passed'] for r in d21c_results.values())


if __name__ == "__main__":
    d26_ok, d21c_ok = main()
    sys.exit(0 if (d26_ok and d21c_ok) else 1)