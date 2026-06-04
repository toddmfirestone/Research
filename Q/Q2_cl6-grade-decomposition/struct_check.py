#!/usr/bin/env python3
# =============================================================================
#  paper2_error_unit_check.py
#
#  Cl(6) Lattice QCD, Paper 2  --  grade-fraction ERROR-PROPAGATION UNIT check
#  Todd M. Firestone  /  diagnostic packaged for local execution
#
#  PURPOSE
#  -------
#  The grade fraction f_r(t) = C_pi^(r)(t) / C_pi(t) is a RATIO of correlators.
#  The only correct statistical error on it is the config-to-config scatter,
#  propagated by jackknifing the RATIO over gauge configurations.
#
#  The suspect alternative (the Q1 trap) is to pool all N_cfg * N_src
#  source-measurements as if they were independent samples, which understates
#  the error by up to ~sqrt(N_src) whenever there is genuine between-config
#  variance.
#
#  This script computes the error BOTH ways, performs a variance-components
#  (one-way ANOVA) decomposition, and -- decisively -- compares both estimates
#  to the QUOTED paper error so we can tell which unit the paper actually used.
#
#  It runs out of the box in --demo mode (synthetic data with known variance
#  structure) so you can confirm the machinery before wiring in real data.
#
#  HOW TO RUN
#  ----------
#    Demonstration (no data needed, proves the script works):
#        python paper2_error_unit_check.py --demo
#
#    Real data, one beta at a time:
#        python paper2_error_unit_check.py --data grade_data_b4p5.npz --quoted 0.000084
#        python paper2_error_unit_check.py --data grade_data_b6p0.npz --quoted 0.000028
#
#  INPUT CONTRACT  (an .npz file produced by your analysis pipeline)
#  -----------------------------------------------------------------
#  PREFERRED (full information -- enables every test):
#      C_r   : float array, shape (N_cfg, N_src, N_t, 4)
#              grade-projected correlators C_pi^(r)(t) per
#              (configuration, point-source, timeslice, channel).
#              Channel order MUST be [ 3 , 3bar , 1a , 1b ].
#      (C_tot is optional; if absent it is taken as C_r.sum(axis=-1),
#       which also lets us re-verify the completeness identity.)
#
#  ACCEPTABLE FALLBACK (per-config, already source-averaged):
#      C_r   : float array, shape (N_cfg, N_t, 4)
#              In this case the source axis is gone, so the naive
#              per-source test CANNOT be run; the script will still
#              compare the config-level jackknife error to the quoted
#              value, which by itself reveals whether the quoted error
#              is the config-to-config scatter.
#
#  Optional keys in the .npz:
#      C_tot : matching shape without the trailing 4   (true denominator)
#      beta  : scalar, for labeling only
#
#  Save with, e.g.:
#      np.savez('grade_data_b4p5.npz', C_r=C_r, C_tot=C_tot, beta=4.5)
#
#  OUTPUT
#  ------
#  Prints a report to the console AND writes 'error_unit_report.txt'
#  in the current directory.  Send that file back.
#
#  Environment: Python 3.11.x, NumPy >= 2.0 (tested target NumPy 2.2.5).
#  No SciPy required.
# =============================================================================

import argparse
import sys
import numpy as np

CH_NAMES = ["3 (triplet)", "3bar (anti-triplet)", "1a (singlet -i)", "1b (singlet +i)"]
CH_DIM   = np.array([3, 3, 1, 1], dtype=float)        # dim(r)
FREE_REF = CH_DIM / 8.0                                # dim(r)/8 = [.375,.375,.125,.125]

# Tolerance (relative) for declaring an SE estimate "matches" the quoted value.
MATCH_TOL = 0.15


# -----------------------------------------------------------------------------
#  Synthetic data for --demo : known between/within variance so the
#  understatement effect is provable.
# -----------------------------------------------------------------------------
def make_demo(N_cfg=20, N_src=24, N_t=6, sigma_b=8.0e-5, sigma_w=4.0e-4, seed=12345):
    """Fabricate C_r with a real between-config component (sigma_b) and a
    larger within-config source-noise component (sigma_w). With sigma_b > 0
    the naive per-source error MUST understate the true config error."""
    rng = np.random.default_rng(seed)
    # per-config offsets (shared by all sources in a config) -> between variance
    cfg_off = rng.normal(0.0, sigma_b, size=(N_cfg, 1, N_t, 4))
    # per-source noise -> within variance
    src_noise = rng.normal(0.0, sigma_w, size=(N_cfg, N_src, N_t, 4))
    f = FREE_REF[None, None, None, :] + cfg_off + src_noise
    # enforce the completeness constraint f sums to 1 over channels per (c,s,t)
    f = f / f.sum(axis=-1, keepdims=True)
    # turn fractions into pseudo-correlators with a positive common scale
    C_tot = np.abs(rng.lognormal(mean=0.0, sigma=0.3, size=(N_cfg, N_src, N_t)))
    C_r = f * C_tot[..., None]
    return C_r, C_tot, None


# -----------------------------------------------------------------------------
#  Core estimators
# -----------------------------------------------------------------------------
def ratio_of_means(Cr_sum, Ct_sum):
    """f_r = (sum over configs of C_r) / (sum over configs of C_tot).
    Cr_sum: (..., 4) ; Ct_sum: (...) -> returns (..., 4)."""
    return Cr_sum / Ct_sum[..., None]


def jackknife_ratio(Cr_cfg, Ct_cfg):
    """Delete-one jackknife of the ratio over the configuration axis (axis 0).
    Cr_cfg: (N_cfg, 4) summed/averaged over sources at a fixed t.
    Ct_cfg: (N_cfg,)   summed/averaged over sources at a fixed t.
    Returns (f_hat[4], se_jk[4])."""
    N = Cr_cfg.shape[0]
    tot_Cr = Cr_cfg.sum(axis=0)          # (4,)
    tot_Ct = Ct_cfg.sum(axis=0)          # scalar
    # leave-one-out sums
    loo_Cr = tot_Cr[None, :] - Cr_cfg    # (N,4)
    loo_Ct = tot_Ct - Ct_cfg            # (N,)
    f_jk = loo_Cr / loo_Ct[:, None]     # (N,4) pseudo-estimates
    f_bar = f_jk.mean(axis=0)
    var_jk = (N - 1) / N * np.sum((f_jk - f_bar[None, :]) ** 2, axis=0)
    f_hat = tot_Cr / tot_Ct             # ratio-of-means central value
    return f_hat, np.sqrt(var_jk)


def bootstrap_ratio(Cr_cfg, Ct_cfg, B=20000, seed=2024):
    """Bootstrap over configurations; SE = std of resampled ratio-of-means."""
    rng = np.random.default_rng(seed)
    N = Cr_cfg.shape[0]
    idx = rng.integers(0, N, size=(B, N))
    boot_Cr = Cr_cfg[idx].sum(axis=1)            # (B,4)
    boot_Ct = Ct_cfg[idx].sum(axis=1)            # (B,)
    f_star = boot_Cr / boot_Ct[:, None]          # (B,4)
    return f_star.std(axis=0, ddof=1)


def variance_components(f_cs):
    """One-way ANOVA, factor = configuration, balanced design.
    f_cs: (N_cfg, N_src, 4) per-source fractions at a fixed t.
    Returns dict with sigma2_b, sigma2_w (each shape (4,))."""
    N_cfg, N_src, _ = f_cs.shape
    grand = f_cs.mean(axis=(0, 1))                          # (4,)
    cfg_mean = f_cs.mean(axis=1)                            # (N_cfg,4)
    ss_between = N_src * np.sum((cfg_mean - grand[None, :]) ** 2, axis=0)
    ss_within = np.sum((f_cs - cfg_mean[:, None, :]) ** 2, axis=(0, 1))
    df_between = N_cfg - 1
    df_within = N_cfg * (N_src - 1)
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    sigma2_w = ms_within
    sigma2_b = np.maximum((ms_between - ms_within) / N_src, 0.0)  # clip tiny negatives
    return dict(sigma2_b=sigma2_b, sigma2_w=sigma2_w,
                ms_between=ms_between, ms_within=ms_within)


def predicted_understatement(sigma2_b, sigma2_w, N_src):
    """ratio = SE_naive / SE_config  predicted from variance components.
    -> sqrt( (sb2 + sw2) / (N_src*sb2 + sw2) ).  ->1 if sb2=0; ->1/sqrt(N_src) if sb2>>sw2."""
    num = sigma2_b + sigma2_w
    den = N_src * sigma2_b + sigma2_w
    return np.sqrt(num / np.where(den > 0, den, np.nan))


# -----------------------------------------------------------------------------
#  Reporting helpers
# -----------------------------------------------------------------------------
def fmt(x):
    return f"{x: .3e}"


def analyze_timeslice(C_r, C_tot, t, quoted, lines):
    has_src = (C_r.ndim == 4)
    p = lines.append

    if has_src:
        N_cfg, N_src = C_r.shape[0], C_r.shape[1]
        Cr_t = C_r[:, :, t, :]                # (N_cfg, N_src, 4)
        Ct_t = C_tot[:, :, t]                 # (N_cfg, N_src)
        # per-config source-averaged correlators
        Cr_cfg = Cr_t.mean(axis=1)            # (N_cfg, 4)
        Ct_cfg = Ct_t.mean(axis=1)            # (N_cfg,)
        # per-source fractions (for ANOVA + naive)
        f_cs = Cr_t / Ct_t[..., None]         # (N_cfg, N_src, 4)
    else:
        N_cfg = C_r.shape[0]
        N_src = 1
        Cr_cfg = C_r[:, t, :]                 # (N_cfg, 4)
        Ct_cfg = C_tot[:, t]                  # (N_cfg,)
        f_cs = None

    # ---- central value & the correct (config) errors ----
    f_hat, se_jk = jackknife_ratio(Cr_cfg, Ct_cfg)
    se_boot = bootstrap_ratio(Cr_cfg, Ct_cfg)
    # analytic config SE from per-config fractions (cross-check of jackknife)
    f_cfg = Cr_cfg / Ct_cfg[:, None]          # (N_cfg,4)
    se_cfg_analytic = f_cfg.std(axis=0, ddof=1) / np.sqrt(N_cfg)

    p(f"\n  Timeslice t = {t}    (N_cfg = {N_cfg}, N_src = {N_src})")
    p(f"  {'channel':<22}{'f_hat':>12}{'dim/8':>10}{'SE_jackknife':>16}"
      f"{'SE_bootstrap':>16}{'SE_cfg(anal)':>16}")
    for r in range(4):
        p(f"  {CH_NAMES[r]:<22}{f_hat[r]:>12.6f}{FREE_REF[r]:>10.3f}"
          f"{fmt(se_jk[r]):>16}{fmt(se_boot[r]):>16}{fmt(se_cfg_analytic[r]):>16}")

    # ---- naive per-source error + variance components (only if sources present)
    if has_src:
        f_pool = f_cs.reshape(-1, 4)
        se_naive = f_pool.std(axis=0, ddof=1) / np.sqrt(N_cfg * N_src)
        vc = variance_components(f_cs)
        pred_ratio = predicted_understatement(vc['sigma2_b'], vc['sigma2_w'], N_src)
        obs_ratio = se_naive / np.where(se_jk > 0, se_jk, np.nan)

        p("")
        p(f"  {'channel':<22}{'SE_naive(pool)':>16}{'sigma_b':>14}{'sigma_w':>14}"
          f"{'pred ratio':>12}{'obs ratio':>12}")
        for r in range(4):
            p(f"  {CH_NAMES[r]:<22}{fmt(se_naive[r]):>16}"
              f"{fmt(np.sqrt(vc['sigma2_b'][r])):>14}{fmt(np.sqrt(vc['sigma2_w'][r])):>14}"
              f"{pred_ratio[r]:>12.3f}{obs_ratio[r]:>12.3f}")
        p(f"\n  Reference: 1/sqrt(N_src) = {1.0/np.sqrt(N_src):.3f}  "
          f"(naive understates the config error by ~this factor when sigma_b dominates)")
    else:
        se_naive = None
        p("\n  [per-source data absent: naive-pooling test and ANOVA skipped]")

    # ---- verdict vs quoted ----
    if quoted is not None:
        p("\n  ---- COMPARISON TO QUOTED PAPER ERROR ----")
        p(f"  quoted SE (this beta) = {quoted: .3e}")
        # use the worst (largest) jackknife SE across channels as the representative
        rep_jk = float(np.max(se_jk))
        rel_jk = abs(rep_jk - quoted) / quoted
        p(f"  config jackknife SE (max over channels) = {rep_jk: .3e}  "
          f"(relative diff {rel_jk*100:5.1f}%)")
        if has_src:
            rep_naive = float(np.max(se_naive))
            rel_naive = abs(rep_naive - quoted) / quoted
            p(f"  naive pooled  SE (max over channels) = {rep_naive: .3e}  "
              f"(relative diff {rel_naive*100:5.1f}%)")

        verdict = "INCONCLUSIVE -- inspect numbers above by hand."
        if rel_jk < MATCH_TOL:
            verdict = ("PASS -- quoted error matches the CONFIG jackknife. "
                       "The paper used the gauge configuration as the unit. Correct.")
        elif has_src and rel_naive < MATCH_TOL and rep_jk > quoted * (1 + MATCH_TOL):
            factor = rep_jk / quoted
            verdict = (f"FAIL -- quoted error matches the NAIVE source-pooled estimate, "
                       f"and the correct config error is ~{factor:.1f}x larger. "
                       f"The quoted +/-{quoted:.0e} UNDERSTATES the error; "
                       f"the significances (0.6 sigma, 3.6 sigma) must be recomputed.")
        p(f"\n  VERDICT (t={t}): {verdict}")
    return


def completeness_check(C_r, C_tot, lines):
    p = lines.append
    recon = C_r.sum(axis=-1)
    denom = np.where(np.abs(C_tot) > 0, np.abs(C_tot), np.nan)
    delta = np.abs(C_tot - recon) / denom
    p("\n  Completeness identity  sum_r C_r = C_tot :")
    p(f"    max relative residual delta_max = {np.nanmax(delta): .3e}  "
      f"(target < 1e-10; expect ~machine epsilon)")


def subsample_scaling(C_r, C_tot, t, lines):
    """Self-consistency: does the config-level SE shrink as 1/sqrt(N_cfg')?
    Subsamples the existing configs (cannot reach N=200; that needs new runs)."""
    p = lines.append
    has_src = (C_r.ndim == 4)
    if has_src:
        Cr_cfg = C_r[:, :, t, :].mean(axis=1)
        Ct_cfg = C_tot[:, :, t].mean(axis=1)
    else:
        Cr_cfg = C_r[:, t, :]
        Ct_cfg = C_tot[:, t]
    N = Cr_cfg.shape[0]
    p("\n  ---- N_cfg SUBSAMPLING SCALING (self-consistency only) ----")
    p("  Expectation: SE ~ 1/sqrt(N_cfg').  This CANNOT reach N=200 with the")
    p("  present configs; demonstrating the central deviation shrinks toward")
    p("  zero at large N genuinely requires new ensembles.")
    p(f"  {'N_cfg':>8}{'SE_3 (jk)':>16}{'SE * sqrt(N)':>18}")
    grid = [n for n in (5, 10, 15, N) if n <= N]
    rng = np.random.default_rng(7)
    for n in grid:
        # average SE over a few random size-n subsets for stability
        ses = []
        reps = 1 if n == N else 8
        for _ in range(reps):
            sel = rng.choice(N, size=n, replace=False) if n < N else np.arange(N)
            _, se = jackknife_ratio(Cr_cfg[sel], Ct_cfg[sel])
            ses.append(se[0])               # channel 3 as representative
        se_mean = float(np.mean(ses))
        p(f"  {n:>8}{fmt(se_mean):>16}{se_mean*np.sqrt(n):>18.3e}")
    p("  If 'SE * sqrt(N)' is roughly constant down the column, the error")
    p("  estimate scales correctly and the N^-1/2 claim is self-consistent.")


# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Paper 2 grade-fraction error-unit check")
    ap.add_argument("--data", type=str, default=None,
                    help=".npz with C_r (and optionally C_tot, beta)")
    ap.add_argument("--quoted", type=float, default=None,
                    help="quoted paper SE for THIS beta (e.g. 0.000084 at b=4.5)")
    ap.add_argument("--t", type=int, default=1, help="timeslice to analyze (default 1)")
    ap.add_argument("--all-t", action="store_true", help="analyze every timeslice")
    ap.add_argument("--scaling", action="store_true",
                    help="also run the N_cfg subsampling scaling diagnostic")
    ap.add_argument("--demo", action="store_true",
                    help="run on synthetic data with known variance structure")
    args = ap.parse_args()

    lines = []
    p = lines.append
    p("=" * 78)
    p("  Cl(6) Paper 2 -- grade-fraction ERROR-PROPAGATION UNIT diagnostic")
    p("=" * 78)

    if args.demo:
        p("\n  MODE: --demo  (synthetic data; sigma_b = 8e-5, sigma_w = 4e-4)")
        p("  Expectation: naive per-source error should UNDERSTATE the config")
        p("  jackknife error, because sigma_b > 0 means configs genuinely differ.")
        C_r, C_tot, beta = make_demo()
        quoted = args.quoted  # may be None
    elif args.data:
        d = np.load(args.data)
        C_r = np.asarray(d["C_r"], dtype=float)
        C_tot = np.asarray(d["C_tot"], dtype=float) if "C_tot" in d else C_r.sum(axis=-1)
        beta = float(d["beta"]) if "beta" in d else None
        quoted = args.quoted
        p(f"\n  MODE: data file '{args.data}'   beta = {beta}")
        p(f"  C_r shape = {C_r.shape}   "
          f"({'per-source' if C_r.ndim == 4 else 'per-config (no source axis)'})")
    else:
        p("\n  No --data and no --demo given.  Run with --demo first, then --data.")
        print("\n".join(lines)); sys.exit(0)

    N_t = C_r.shape[-2] if C_r.ndim == 4 else C_r.shape[1]
    completeness_check(C_r, C_tot, lines)

    t_list = range(N_t) if args.all_t else [args.t]
    for t in t_list:
        if t >= N_t:
            p(f"\n  (skipping t={t}: only {N_t} timeslices present)")
            continue
        analyze_timeslice(C_r, C_tot, t, quoted, lines)

    if args.scaling:
        subsample_scaling(C_r, C_tot, args.t, lines)

    p("\n" + "=" * 78)
    p("  Send back: this report (error_unit_report.txt).")
    p("  Key line is the VERDICT for t=1 vs the quoted error for that beta.")
    p("=" * 78)

    report = "\n".join(lines)
    print(report)
    with open("error_unit_report.txt", "w", encoding="utf-8") as fh:
        fh.write(report + "\n")


if __name__ == "__main__":
    main()
