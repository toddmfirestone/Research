#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rerun_N6_longtherm.py
=====================================================================
Long-thermalization, crash-resilient, multi-core re-run of the N=6
pure-gauge plaquette / Wilson-loop sweep, built on ga_su3_lattice_v2.

WHY THIS EXISTS
  A fresh sweep reproduced Table 1/2 ONLY at the endpoints (beta=4.5,
  6.5).  The interior couplings (5.0, 5.5, 5.69, 6.0) came out many
  sigma off, and their thermalization traces were still drifting at
  500 sweeps.  This driver re-runs the interior points with much
  longer thermalization and a drift check, to test whether <P> and
  the Wilson loops converge to the table values.

WHAT IT GUARANTEES
  * Incremental, flushed, fsync'd per-config writes  -> a reboot loses
    at most the configuration currently in flight.
  * Atomic checkpoints of the FULL gauge field + RNG state, during
    BOTH thermalization and measurement.  Re-running the same command
    after a crash resumes the Markov chain bit-identically.
  * One beta per process (embarrassingly parallel), BLAS threads pinned
    to 1 so the workers don't oversubscribe your cores.
  * A plaquette-slope drift check on the thermalization tail AND the
    measurement window, with an equilibration verdict.

USAGE
  Put this file in the SAME folder as ga_su3_lattice_v2.py, then:
      python rerun_N6_longtherm.py
  If your machine reboots mid-run, just run that again.  Completed
  betas are skipped instantly; a half-finished beta resumes.

OUTPUT  (folder OUTDIR below)
  log_b{beta}.txt        full text log for that beta (live, line-buffered)
  data_b{beta}.csv       per-config rows: cfg,plaquette,W11,W22  (fsync'd)
  ckpt_b{beta}.pkl       checkpoint: U + RNG state + counters (atomic)
  wloop_N6_b{beta}.npz   final raw arrays + summary scalars (= "done" flag)
  summary_b{beta}.txt    one-line human summary
  master_summary.csv     combined table, written as betas finish
=====================================================================
"""

# --- pin BLAS/OpenMP threads to 1 BEFORE numpy is imported anywhere ---
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import sys
import time
import pickle
import contextlib
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

# ======================  KNOBS  ======================
# Default = the four interior couplings that failed to reproduce.
# For a clean apples-to-apples table, consider adding 4.5 and 6.5 so
# all six rows share identical settings:  BETAS = [4.5,5.0,5.5,5.69,6.0,6.5]
BETAS        = [5.0, 5.5, 5.69, 6.0]

N            = 6        # lattice extent (N^4)
D            = 4
N_THERM      = 3000     # thermalization sweeps (was 500 -> too short for weak coupling)
N_CONFIGS    = 100      # measured configurations
DECORR       = 5        # sweeps between measurements (matches the original)
METRO_EPS    = 0.25     # Metropolis step size (matches the endpoint runs)
SU3_EPS      = 0.3      # SU3Algebra eps (matches the original construction)
BASE_SEED    = 12345    # per-beta seed = BASE_SEED + index  (reproducible & resumable)

CKPT_EVERY_THERM = 100  # checkpoint every this many thermalization sweeps
REC_THERM_EVERY  = 25   # record <P> for the thermalization drift trace this often
CKPT_EVERY_CFG   = 1    # checkpoint every this many measured configs (1 = max safety)

OUTDIR       = "rerun_N6_longtherm"
MAX_WORKERS  = None     # None -> min(len(BETAS), os.cpu_count())
# =====================================================


# ----------------------- small helpers -----------------------
def _atomic_pickle(obj, path):
    """Write a pickle atomically: tmp file -> fsync -> os.replace."""
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)            # atomic on Windows and POSIX


def _drift_report(idx, vals, label, log):
    """Linear-fit a series, print slope +/- err and an equilibration verdict."""
    idx  = np.asarray(idx, dtype=float)
    vals = np.asarray(vals, dtype=float)
    n = len(idx)
    if n < 4:
        log(f"  [{label}] too few points for a drift fit (n={n})")
        return
    slope, intercept = np.polyfit(idx, vals, 1)
    resid = vals - (slope * idx + intercept)
    sxx   = np.sum((idx - idx.mean()) ** 2)
    s_err = np.sqrt(np.sum(resid ** 2) / (n - 2) / sxx) if sxx > 0 else np.inf
    total = slope * (idx[-1] - idx[0])
    mean  = vals.mean()
    stat  = vals.std(ddof=1) / np.sqrt(n)
    if abs(slope) < 2 * s_err:
        verdict = "no significant slope  -> EQUILIBRATED"
    elif abs(total) < stat:
        verdict = "slope nonzero but total drift < stat error -> acceptable"
    else:
        verdict = "STILL DRIFTING  (total drift exceeds stat error)"
    log(f"  [{label}] slope = {slope:+.3e} +/- {s_err:.3e} per step")
    log(f"  [{label}] total drift over window = {total:+.5f}   "
        f"(mean = {mean:.5f} +/- {stat:.5f})")
    log(f"  [{label}] VERDICT: {verdict}")


# ----------------------- the worker -----------------------
def run_one_beta(args):
    """Run (or resume) a single beta.  Returns a summary dict."""
    beta, seed = args

    # Import the framework INSIDE the worker so spawn-based start methods
    # (Windows / macOS) re-import cleanly.  ga_su3_lattice_v2's __main__
    # blocks are guarded, so importing it has no side effects.
    import ga_su3_lattice_v2 as ga

    os.makedirs(OUTDIR, exist_ok=True)
    btag    = f"{beta}"
    logpath = os.path.join(OUTDIR, f"log_b{btag}.txt")
    csvpath = os.path.join(OUTDIR, f"data_b{btag}.csv")
    ckpath  = os.path.join(OUTDIR, f"ckpt_b{btag}.pkl")
    npzpath = os.path.join(OUTDIR, f"wloop_N6_b{btag}.npz")
    sumpath = os.path.join(OUTDIR, f"summary_b{btag}.txt")

    # Already finished?  Skip and report from the npz.
    if os.path.exists(npzpath):
        d = np.load(npzpath)
        return dict(beta=float(d["beta"]), P_mean=float(d["P_mean"]),
                    P_err=float(d["P_err"]), W11=float(d["W11_mean"]),
                    W22=float(d["W22_mean"]), R_AL=float(d["R_AL"]),
                    sigma_eff=float(d["sigma_eff"]),
                    acceptance=float(d["acceptance"]), status="already-done")

    logf = open(logpath, "a", buffering=1, encoding="utf-8")   # line-buffered
    def log(msg=""):
        logf.write(msg + "\n")

    with contextlib.redirect_stdout(logf):   # capture framework banners too
        t0 = time.time()
        log("=" * 60)
        log(f"BETA {beta}  |  N={N}^4  n_therm={N_THERM}  n_configs={N_CONFIGS}"
            f"  eps={METRO_EPS}  seed={seed}")
        log(f"started {time.strftime('%Y-%m-%d %H:%M:%S')}")
        log("=" * 60)

        # Build the (cheap) framework objects in this process.
        cl  = ga.CliffordCl6()
        su3 = ga.SU3Algebra(eps=SU3_EPS, g=1.0)
        lat = ga.Lattice(N=N, D=D)

        # ---- restore from checkpoint, or start fresh ----
        if os.path.exists(ckpath):
            with open(ckpath, "rb") as f:
                ck = pickle.load(f)
            np.random.set_state(ck["rng_state"])
            fields = ga.Fields(lat, cl, su3, mode="cold")
            fields.U = ck["U"]                       # overwrite with saved field
            metro = ga.Metropolis(fields, beta=beta, eps=METRO_EPS)
            metro._n_proposed = ck["n_proposed"]
            metro._n_accepted = ck["n_accepted"]
            phase        = ck["phase"]
            sweep_done   = ck["sweep_done"]
            cfg_done     = ck["cfg_done"]
            therm_trace  = ck["therm_trace"]
            measurements = ck["measurements"]
            log(f"RESUMED from checkpoint: phase={phase} "
                f"sweep_done={sweep_done} cfg_done={cfg_done}")
            # keep the CSV exactly in sync with the checkpoint
            with open(csvpath, "w", encoding="utf-8") as cf:
                cf.write("cfg,plaquette,W11,W22\n")
                for i, m in enumerate(measurements, 1):
                    cf.write(f"{i},{m[0]:.10f},{m[1]:.10f},{m[2]:.10f}\n")
        else:
            np.random.seed(seed)
            fields = ga.Fields(lat, cl, su3, mode="hot")
            metro  = ga.Metropolis(fields, beta=beta, eps=METRO_EPS)
            phase, sweep_done, cfg_done = "therm", 0, 0
            therm_trace, measurements = [], []
            with open(csvpath, "w", encoding="utf-8") as cf:
                cf.write("cfg,plaquette,W11,W22\n")

        obs = ga.Observables(metro.F)
        wil = ga.WilsonLoop(metro.F)

        def save_ckpt():
            _atomic_pickle(dict(
                beta=beta, phase=phase, sweep_done=sweep_done, cfg_done=cfg_done,
                U=metro.F.U, rng_state=np.random.get_state(),
                therm_trace=therm_trace, measurements=measurements,
                n_proposed=metro._n_proposed, n_accepted=metro._n_accepted,
            ), ckpath)

        # ----------------- THERMALIZATION (resumable) -----------------
        if phase == "therm":
            log(f"\nThermalizing from sweep {sweep_done+1} to {N_THERM} ...")
            while sweep_done < N_THERM:
                metro.sweep()
                sweep_done += 1
                if sweep_done % REC_THERM_EVERY == 0 or sweep_done == 1:
                    p = obs.avg_plaquette()
                    therm_trace.append([sweep_done, p])
                    log(f"  therm sweep {sweep_done:>6}  <P>={p:.6f}  "
                        f"acc={metro.acceptance_rate:.4f}")
                if sweep_done % CKPT_EVERY_THERM == 0:
                    save_ckpt()
            # drift check on the last 25% of the thermalization trace
            if therm_trace:
                tr = np.array(therm_trace)
                tail = tr[tr[:, 0] >= 0.75 * N_THERM]
                if len(tail) >= 4:
                    log("")
                    _drift_report(tail[:, 0], tail[:, 1],
                                  "therm-tail", log)
            phase, cfg_done = "measure", 0
            save_ckpt()

        # ----------------- MEASUREMENT (resumable, incremental) -----------------
        log(f"\nMeasuring configs from {cfg_done+1} to {N_CONFIGS} "
            f"[{DECORR} sweeps between] ...")
        cf = open(csvpath, "a", buffering=1, encoding="utf-8")
        while cfg_done < N_CONFIGS:
            for _ in range(DECORR):
                metro.sweep()
            p   = obs.avg_plaquette()
            w11 = wil.avg_loop(1, 1)
            w22 = wil.avg_loop(2, 2)
            cfg_done += 1
            measurements.append([p, w11, w22])
            cf.write(f"{cfg_done},{p:.10f},{w11:.10f},{w22:.10f}\n")
            cf.flush(); os.fsync(cf.fileno())          # survive a reboot
            if cfg_done % 10 == 0 or cfg_done == N_CONFIGS:
                arr = np.array(measurements)
                log(f"  cfg {cfg_done:>4}  <P>={arr[:,0].mean():.5f}  "
                    f"W11={arr[:,1].mean():.5f}  W22={arr[:,2].mean():.5f}")
            if cfg_done % CKPT_EVERY_CFG == 0:
                save_ckpt()                            # CSV written first, then ckpt
        cf.close()

        # ----------------- finalize -----------------
        arr = np.array(measurements)
        P, W11a, W22a = arr[:, 0], arr[:, 1], arr[:, 2]
        P_mean = P.mean();  P_err = P.std(ddof=1) / np.sqrt(len(P))
        w11 = W11a.mean();  w22 = W22a.mean()
        R_AL = w22 / w11 ** 2
        sigma_eff = -(1.0 / 3.0) * np.log(w22 / w11)
        acc = metro.acceptance_rate

        log("\n" + "-" * 60)
        log("MEASUREMENT-PHASE DRIFT CHECK")
        _drift_report(np.arange(1, len(P) + 1), P, "measure", log)
        log("-" * 60)
        log("PHYSICS REPORT")
        log(f"  beta        = {beta}")
        log(f"  N configs   = {len(P)}")
        log(f"  acceptance  = {acc:.4f}")
        log(f"  <P>         = {P_mean:.5f} +/- {P_err:.5f}")
        log(f"  <W(1,1)>    = {w11:.5f}")
        log(f"  <W(2,2)>    = {w22:.5f}")
        log(f"  R_AL        = {R_AL:.4f}")
        log(f"  sigma_eff   = {sigma_eff:.4f}")
        log(f"  wall time   = {(time.time()-t0)/60:.1f} min")

        np.savez(npzpath, beta=beta, plaquette=P, W11=W11a, W22=W22a,
                 P_mean=P_mean, P_err=P_err, W11_mean=w11, W22_mean=w22,
                 R_AL=R_AL, sigma_eff=sigma_eff, acceptance=acc,
                 n_therm=N_THERM, n_configs=N_CONFIGS, seed=seed)
        with open(sumpath, "w", encoding="utf-8") as sf:
            sf.write(f"beta={beta}  <P>={P_mean:.5f}+/-{P_err:.5f}  "
                     f"W11={w11:.5f}  W22={w22:.5f}  R_AL={R_AL:.4f}  "
                     f"sigma_eff={sigma_eff:.4f}  acc={acc:.4f}\n")

        # leave the checkpoint in place; npz presence marks completion
    logf.close()
    return dict(beta=beta, P_mean=P_mean, P_err=P_err, W11=w11, W22=w22,
                R_AL=R_AL, sigma_eff=sigma_eff, acceptance=acc, status="done")


# ----------------------- orchestrator -----------------------
def main():
    os.makedirs(OUTDIR, exist_ok=True)
    workers = MAX_WORKERS or min(len(BETAS), os.cpu_count() or 1)
    print(f"Launching {len(BETAS)} beta(s) across {workers} worker process(es).")
    print(f"Watch progress live in {OUTDIR}/log_b*.txt")
    print("Safe to Ctrl-C / reboot and re-run: completed betas skip, "
          "partial betas resume.\n")

    jobs = [(b, BASE_SEED + i) for i, b in enumerate(BETAS)]
    results = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run_one_beta, j): j[0] for j in jobs}
        for fut in as_completed(futs):
            b = futs[fut]
            try:
                r = results.append(fut.result()) or results[-1]
                print(f"[done] beta={r['beta']}  <P>={r['P_mean']:.5f}"
                      f"+/-{r['P_err']:.5f}  R_AL={r['R_AL']:.4f}  ({r['status']})")
            except Exception as e:
                print(f"[FAIL] beta={b}: {e!r}  -- re-run to resume this beta")

    # combined table from whatever npz files exist
    print("\n" + "=" * 72)
    print("  beta   <P>        +/-       W(1,1)    W(2,2)    R_AL      sig_eff   acc")
    rows = []
    for b in sorted(BETAS):
        p = os.path.join(OUTDIR, f"wloop_N6_b{b}.npz")
        if not os.path.exists(p):
            print(f"  {b:<5} (incomplete -- re-run to finish)")
            continue
        d = np.load(p)
        print(f"  {b:<5} {float(d['P_mean']):.5f}  {float(d['P_err']):.5f}  "
              f"{float(d['W11_mean']):.5f}  {float(d['W22_mean']):.5f}  "
              f"{float(d['R_AL']):.4f}  {float(d['sigma_eff']):.4f}  "
              f"{float(d['acceptance']):.4f}")
        rows.append([b, float(d['P_mean']), float(d['P_err']),
                     float(d['W11_mean']), float(d['W22_mean']),
                     float(d['R_AL']), float(d['sigma_eff']),
                     float(d['acceptance'])])
    print("=" * 72)
    if rows:
        with open(os.path.join(OUTDIR, "master_summary.csv"), "w",
                  encoding="utf-8") as f:
            f.write("beta,P_mean,P_err,W11,W22,R_AL,sigma_eff,acceptance\n")
            for r in rows:
                f.write(",".join(f"{x:.6f}" for x in r) + "\n")
        print(f"Combined table written to {OUTDIR}/master_summary.csv")


if __name__ == "__main__":
    main()
