#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
error_treatment.py
=====================================================================
Honest error bars for the Cl(6) lattice ensembles.  Addresses three
findings:

  M4 (autocorrelation): the naive std/sqrt(N) assumes independent
      configs.  We measure the integrated autocorrelation time tau_int
      (Sokal automatic windowing) and inflate:  err -> err*sqrt(2*tau).

  M6 (blocked bootstrap): derived quantities (GOR gap dA, R_AL,
      sigma_eff) are resampled with a MOVING-BLOCK bootstrap whose block
      length L ~ ceil(2*tau_int), so the resampling respects
      autocorrelation instead of pretending configs are independent.

  M1 (run-to-run systematic): no single chain can see this.  If you
      provide several INDEPENDENT-SEED runs of the same (beta) point,
      this script reports the across-seed spread as a systematic and
      combines it in quadrature:  err_total = sqrt(err_stat^2 + sys^2).

This is a PURE-ANALYSIS script: it reads existing per-config data and
recomputes errors.  It runs nothing expensive and starts no chains.

USAGE
  Edit the paths in CONFIG, then:  python error_treatment.py
  - SWEEP_DIR : folder of sweep_*.txt fermion files (per-config m_eff)
  - GAUGE_GLOB: glob of gauge npz files w/ per-config 'plaquette','W11','W22'
  - For the systematic, point SEED_RUNS at >=2 npz of the same beta
    produced with different seeds (see rerun_N6_longtherm.py BASE_SEED).

OUTPUT
  A table per observable: naive err, tau_int, autocorr-corrected err,
  and (if seeds supplied) the run-to-run systematic and combined error.
=====================================================================
"""

import os, re, glob
import numpy as np

# ============================ CONFIG ============================
SWEEP_DIR  = "sweep_results"                 # fermion per-config files
GAUGE_GLOB = "rerun_N6_longtherm/wloop_N6_b*.npz"
# Optional: independent-seed npz per beta for the M1 systematic, e.g.
#   {6.0: ["seedA_b6.0.npz","seedB_b6.0.npz","seedC_b6.0.npz"]}
SEED_RUNS  = {}
CONFINED, DECONFINED = 4.5, 6.0
N_BOOT = 5000
SEED   = 12345
# Mass points per N for the GOR gap (edit to match your fits)
MASS_POINTS = {4:[0.05,0.1,0.2,0.3,0.5], 6:[0.02,0.03,0.04,0.05],
               8:[0.001,0.005,0.008,0.05,0.1],
               10:[0.001,0.002,0.003,0.004,0.005,0.01]}
# ===============================================================


# ---------------- integrated autocorrelation time ----------------
def tau_int(x, c=5.0):
    """Sokal automatic-windowing integrated autocorrelation time (in
    units of the sample spacing).  Returns (tau, window)."""
    x = np.asarray(x, float); n = len(x)
    if n < 4: return 0.5, 0
    x = x - x.mean()
    f = np.fft.fft(x, 2 * n)
    acf = np.fft.ifft(f * np.conj(f))[:n].real
    if acf[0] == 0: return 0.5, 0
    acf /= acf[0]
    tau, W = 0.5, n // 3
    for t in range(1, n // 3):
        tau += acf[t]
        if t >= c * tau:
            W = t; break
    return max(tau, 0.5), W


def autocorr_error(x):
    """Return (mean, naive_err, tau, corrected_err)."""
    x = np.asarray(x, float); n = len(x)
    naive = x.std(ddof=1) / np.sqrt(n)
    tau, _ = tau_int(x)
    return x.mean(), naive, tau, naive * np.sqrt(2 * tau)


# ---------------- moving-block bootstrap helpers ----------------
def _block_resample(a, L, rng):
    n = len(a); nbl = int(np.ceil(n / L)); idx = []
    hi = max(1, n - L + 1)
    for _ in range(nbl):
        s = rng.integers(0, hi); idx.extend(range(s, s + L))
    return a[np.array(idx[:n])]


def _gor_intercept(mpts, series):
    xs = np.array(mpts); ys = np.array([s.mean() for s in series])
    A = np.vstack([xs, np.ones_like(xs)]).T
    return np.linalg.lstsq(A, ys, rcond=None)[0][1]   # intercept = chiral-limit m_pi^2


def blocked_gap(N, mpts, m2_conf, m2_deconf, rng):
    """Blocked-bootstrap error on dA = intercept(conf) - intercept(deconf).
    m2_*: list (per mass point) of per-config m_pi^2 arrays."""
    # block length from the worst tau over all involved series
    taus = [tau_int(a)[0] for a in m2_conf + m2_deconf]
    L = max(1, int(np.ceil(2 * max(taus))))
    central = _gor_intercept(mpts, m2_conf) - _gor_intercept(mpts, m2_deconf)
    bs = []
    for _ in range(N_BOOT):
        cb = [_block_resample(a, L, rng) for a in m2_conf]
        db = [_block_resample(a, L, rng) for a in m2_deconf]
        bs.append(_gor_intercept(mpts, cb) - _gor_intercept(mpts, db))
    return central, np.std(bs), L


# ---------------- parse fermion sweep files ----------------
def load_fermion(sweep_dir):
    data = {}
    for fn in glob.glob(os.path.join(sweep_dir, "sweep_*.txt")):
        txt = open(fn, encoding="utf-8", errors="replace").read()
        m = re.search(r"beta=([\d.]+)\s+m0=([\d.]+)\s+N=(\d+)", txt)
        if not m: continue
        beta, m0, N = float(m.group(1)), float(m.group(2)), int(m.group(3))
        rows = []
        for line in txt.splitlines():
            tk = line.split()
            if len(tk) >= 4 and re.match(r"^\d+$", tk[0]):
                try: nums = [float(z) for z in tk[1:]]
                except ValueError: continue
                if abs(nums[0]) < 2 and 0 < abs(nums[-1]) < 100:
                    rows.append(nums[-1])          # last col = m_eff
        if rows:
            data.setdefault((N, beta, m0), []).extend(rows)
    return data


# ============================ main ============================
def main():
    rng = np.random.default_rng(SEED)

    # ---- GAUGE: per-config autocorrelation-corrected errors ----
    print("=" * 74)
    print("GAUGE observables: autocorrelation-corrected errors")
    print("=" * 74)
    print(f"{'file':<26}{'<P>':>9}{'naive':>9}{'tau':>6}{'corr err':>10}{'x':>5}")
    for f in sorted(glob.glob(GAUGE_GLOB)):
        d = np.load(f)
        if "plaquette" not in d: continue
        mean, naive, tau, corr = autocorr_error(d["plaquette"])
        print(f"{os.path.basename(f):<26}{mean:>9.5f}{naive:>9.5f}{tau:>6.2f}"
              f"{corr:>10.5f}{corr/naive:>5.1f}")

    # ---- run-to-run systematic from independent seeds (M1) ----
    if SEED_RUNS:
        print("\n" + "=" * 74)
        print("RUN-TO-RUN SYSTEMATIC (across independent seeds)  ->  M1")
        print("=" * 74)
        for beta, files in SEED_RUNS.items():
            means = []
            for f in files:
                d = np.load(f); means.append(float(d["plaquette"].mean()))
            means = np.array(means)
            sys = means.std(ddof=1) if len(means) > 1 else float("nan")
            # combine with a representative autocorr-corrected stat error
            d0 = np.load(files[0]); _, _, _, corr = autocorr_error(d0["plaquette"])
            tot = np.sqrt(corr ** 2 + sys ** 2)
            print(f"  beta={beta}: {len(means)} seeds  mean={means.mean():.5f}  "
                  f"sys(across-seed)={sys:.5f}  stat(corr)={corr:.5f}  "
                  f"=> TOTAL={tot:.5f}")
    else:
        print("\n[M1] No SEED_RUNS supplied -> run-to-run systematic not quantified.")
        print("     Provide >=2 independent-seed npz per beta to fill this in.")

    # ---- FERMION: GOR gap dA with blocked bootstrap (M4/M6) ----
    print("\n" + "=" * 74)
    print("GOR intercept gap dA: blocked-bootstrap (autocorr-aware) errors")
    print("=" * 74)
    data = load_fermion(SWEEP_DIR)
    if not data:
        print(f"  (no sweep_*.txt found in {SWEEP_DIR})")
        return
    print(f"{'N':>3}{'dA':>10}{'block':>7}{'err_blk':>10}{'sig':>8}   (naive sig in parens)")
    for N, mpts in MASS_POINTS.items():
        try:
            C = [np.array(data[(N, CONFINED,  m0)]) ** 2 for m0 in mpts]
            D = [np.array(data[(N, DECONFINED, m0)]) ** 2 for m0 in mpts]
        except KeyError:
            print(f"{N:>3}  (missing mass points)"); continue
        dA, err, L = blocked_gap(N, mpts, C, D, rng)
        # naive (block=1) for comparison
        L1 = 1
        bs = [_gor_intercept(mpts, [_block_resample(a, L1, rng) for a in C])
              - _gor_intercept(mpts, [_block_resample(a, L1, rng) for a in D])
              for _ in range(N_BOOT)]
        err1 = np.std(bs)
        print(f"{N:>3}{dA:>10.4f}{L:>7d}{err:>10.4f}{dA/err:>7.1f}σ"
              f"   ({dA/err1:.1f}σ)")
    print("\nNote: blocked errors include autocorrelation but NOT the run-to-run")
    print("systematic (M1) or the linear-GOR / single-slice-m_eff systematics.")
    print("dA is a phase DIFFERENCE, so common biases largely cancel -- it is the")
    print("most robust quantity; absolute plaquette/sigma carry the full systematic.")


if __name__ == "__main__":
    main()
