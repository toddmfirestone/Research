#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rerun_N6_multiseed.py   (supersedes rerun_N6_longtherm.py)
=====================================================================
Same crash-safe, resumable, drift-checked long-thermalization driver,
now running N_SEEDS INDEPENDENT chains per beta.  The spread of <P>
(and W's) ACROSS seeds is the run-to-run systematic (finding M1) that
no single chain can see.

Each (beta, seed) is an independent job with its own files, so:
  * reboot-safe + resumable per job (re-run the command to continue),
  * fully parallel across all beta x seed jobs,
  * the final block prints, per beta, the across-seed mean +/- systematic,
    and a ready-to-paste SEED_RUNS = {...} dict for error_treatment.py.

USAGE
  Put this in the same folder as ga_su3_lattice_v2.py, then:
      python rerun_N6_multiseed.py
  Re-run after any reboot; finished jobs skip, partial jobs resume.

OUTPUT  (folder OUTDIR)
  log_b{beta}_s{seed}.txt     live text log per job
  data_b{beta}_s{seed}.csv    per-config rows (fsync'd)
  ckpt_b{beta}_s{seed}.pkl    checkpoint: U + RNG + counters (atomic)
  wloop_N6_b{beta}_s{seed}.npz   final raw arrays + scalars (= done flag)
  systematic_summary.csv      per-beta across-seed mean & systematic
=====================================================================
"""

# --- pin BLAS/OpenMP threads to 1 BEFORE numpy is imported anywhere ---
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import time
import pickle
import contextlib
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

# ======================  KNOBS  ======================
BETAS        = [4.5, 5.0, 5.5, 5.69, 6.0, 6.5]   # all six for a uniform table
N_SEEDS      = 5        # independent chains per beta -> across-seed systematic
BASE_SEED    = 20260603

N            = 6
D            = 4
N_THERM      = 3000
N_CONFIGS    = 100
DECORR       = 5
METRO_EPS    = 0.25
SU3_EPS      = 0.3

CKPT_EVERY_THERM = 100
REC_THERM_EVERY  = 25
CKPT_EVERY_CFG   = 1

OUTDIR       = "rerun_N6_multiseed"
MAX_WORKERS  = None     # None -> min(n_jobs, os.cpu_count())
# =====================================================


# ----------------------- helpers -----------------------
def _atomic_pickle(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def _drift_report(idx, vals, label, log):
    idx = np.asarray(idx, float); vals = np.asarray(vals, float); n = len(idx)
    if n < 4:
        log(f"  [{label}] too few points for a drift fit (n={n})"); return
    slope, intercept = np.polyfit(idx, vals, 1)
    resid = vals - (slope * idx + intercept)
    sxx = np.sum((idx - idx.mean()) ** 2)
    s_err = np.sqrt(np.sum(resid ** 2) / (n - 2) / sxx) if sxx > 0 else np.inf
    total = slope * (idx[-1] - idx[0]); mean = vals.mean()
    stat = vals.std(ddof=1) / np.sqrt(n)
    if abs(slope) < 2 * s_err:
        verdict = "no significant slope  -> EQUILIBRATED"
    elif abs(total) < stat:
        verdict = "slope nonzero but total drift < stat error -> acceptable"
    else:
        verdict = "STILL DRIFTING  (total drift exceeds stat error)"
    log(f"  [{label}] slope = {slope:+.3e} +/- {s_err:.3e} per step")
    log(f"  [{label}] total drift = {total:+.5f}   (mean = {mean:.5f} +/- {stat:.5f})")
    log(f"  [{label}] VERDICT: {verdict}")


# ----------------------- the worker -----------------------
def run_job(args):
    """Run (or resume) one (beta, seed) chain.  Returns a summary dict."""
    beta, seed = args
    import ga_su3_lattice_v2 as ga      # spawn-safe; __main__ guarded on import

    os.makedirs(OUTDIR, exist_ok=True)
    tag     = f"b{beta}_s{seed}"
    logpath = os.path.join(OUTDIR, f"log_{tag}.txt")
    csvpath = os.path.join(OUTDIR, f"data_{tag}.csv")
    ckpath  = os.path.join(OUTDIR, f"ckpt_{tag}.pkl")
    npzpath = os.path.join(OUTDIR, f"wloop_N6_{tag}.npz")

    if os.path.exists(npzpath):
        d = np.load(npzpath)
        return dict(beta=float(d["beta"]), seed=int(d["seed"]),
                    P_mean=float(d["P_mean"]), R_AL=float(d["R_AL"]),
                    npz=npzpath, status="already-done")

    logf = open(logpath, "a", buffering=1, encoding="utf-8")
    def log(m=""): logf.write(m + "\n")

    with contextlib.redirect_stdout(logf):
        t0 = time.time()
        log("=" * 60)
        log(f"BETA {beta}  SEED {seed}  |  N={N}^4  n_therm={N_THERM}  "
            f"n_configs={N_CONFIGS}  eps={METRO_EPS}")
        log(f"started {time.strftime('%Y-%m-%d %H:%M:%S')}")
        log("=" * 60)

        cl  = ga.CliffordCl6()
        su3 = ga.SU3Algebra(eps=SU3_EPS, g=1.0)
        lat = ga.Lattice(N=N, D=D)

        if os.path.exists(ckpath):
            with open(ckpath, "rb") as f: ck = pickle.load(f)
            np.random.set_state(ck["rng_state"])
            fields = ga.Fields(lat, cl, su3, mode="cold"); fields.U = ck["U"]
            metro = ga.Metropolis(fields, beta=beta, eps=METRO_EPS)
            metro._n_proposed = ck["n_proposed"]; metro._n_accepted = ck["n_accepted"]
            phase, sweep_done, cfg_done = ck["phase"], ck["sweep_done"], ck["cfg_done"]
            therm_trace, measurements = ck["therm_trace"], ck["measurements"]
            log(f"RESUMED: phase={phase} sweep_done={sweep_done} cfg_done={cfg_done}")
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

        obs = ga.Observables(metro.F); wil = ga.WilsonLoop(metro.F)

        def save_ckpt():
            _atomic_pickle(dict(beta=beta, phase=phase, sweep_done=sweep_done,
                cfg_done=cfg_done, U=metro.F.U, rng_state=np.random.get_state(),
                therm_trace=therm_trace, measurements=measurements,
                n_proposed=metro._n_proposed, n_accepted=metro._n_accepted), ckpath)

        # ---- thermalization (resumable) ----
        if phase == "therm":
            log(f"\nThermalizing from sweep {sweep_done+1} to {N_THERM} ...")
            while sweep_done < N_THERM:
                metro.sweep(); sweep_done += 1
                if sweep_done % REC_THERM_EVERY == 0 or sweep_done == 1:
                    p = obs.avg_plaquette(); therm_trace.append([sweep_done, p])
                    log(f"  therm {sweep_done:>6}  <P>={p:.6f}  acc={metro.acceptance_rate:.4f}")
                if sweep_done % CKPT_EVERY_THERM == 0: save_ckpt()
            if therm_trace:
                tr = np.array(therm_trace); tail = tr[tr[:, 0] >= 0.75 * N_THERM]
                if len(tail) >= 4:
                    log(""); _drift_report(tail[:, 0], tail[:, 1], "therm-tail", log)
            phase, cfg_done = "measure", 0; save_ckpt()

        # ---- measurement (resumable, incremental) ----
        log(f"\nMeasuring configs {cfg_done+1}..{N_CONFIGS} [{DECORR} sweeps between] ...")
        cf = open(csvpath, "a", buffering=1, encoding="utf-8")
        while cfg_done < N_CONFIGS:
            for _ in range(DECORR): metro.sweep()
            p = obs.avg_plaquette(); w11 = wil.avg_loop(1, 1); w22 = wil.avg_loop(2, 2)
            cfg_done += 1; measurements.append([p, w11, w22])
            cf.write(f"{cfg_done},{p:.10f},{w11:.10f},{w22:.10f}\n")
            cf.flush(); os.fsync(cf.fileno())
            if cfg_done % 25 == 0 or cfg_done == N_CONFIGS:
                a = np.array(measurements)
                log(f"  cfg {cfg_done:>4}  <P>={a[:,0].mean():.5f}  W22={a[:,2].mean():.5f}")
            if cfg_done % CKPT_EVERY_CFG == 0: save_ckpt()
        cf.close()

        # ---- finalize ----
        a = np.array(measurements); P, W11a, W22a = a[:, 0], a[:, 1], a[:, 2]
        P_mean = P.mean(); P_err = P.std(ddof=1) / np.sqrt(len(P))
        w11, w22 = W11a.mean(), W22a.mean()
        R_AL = w22 / w11 ** 2; sigma_eff = -(1.0 / 3.0) * np.log(w22 / w11)
        log("\nMEASUREMENT DRIFT CHECK")
        _drift_report(np.arange(1, len(P) + 1), P, "measure", log)
        log(f"PHYSICS: beta={beta} seed={seed}  <P>={P_mean:.5f}+/-{P_err:.5f}  "
            f"R_AL={R_AL:.4f}  acc={metro.acceptance_rate:.4f}  "
            f"wall={(time.time()-t0)/60:.1f}min")
        np.savez(npzpath, beta=beta, seed=seed, plaquette=P, W11=W11a, W22=W22a,
                 P_mean=P_mean, P_err=P_err, W11_mean=w11, W22_mean=w22,
                 R_AL=R_AL, sigma_eff=sigma_eff, acceptance=metro.acceptance_rate,
                 n_therm=N_THERM, n_configs=N_CONFIGS)
    logf.close()
    return dict(beta=beta, seed=seed, P_mean=P_mean, R_AL=R_AL,
                npz=npzpath, status="done")


# ----------------------- orchestrator -----------------------
def main():
    os.makedirs(OUTDIR, exist_ok=True)
    # distinct seed per (beta, seed-index); independent across all jobs
    jobs = []
    for i, b in enumerate(BETAS):
        for s in range(N_SEEDS):
            jobs.append((b, BASE_SEED + 1000 * i + s))
    workers = MAX_WORKERS or min(len(jobs), os.cpu_count() or 1)
    print(f"{len(jobs)} jobs ({len(BETAS)} beta x {N_SEEDS} seeds) "
          f"across {workers} workers. Watch {OUTDIR}/log_*.txt")
    print("Reboot-safe: re-run to resume.\n")

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run_job, j): j for j in jobs}
        for fut in as_completed(futs):
            b, s = futs[fut]
            try:
                r = fut.result()
                print(f"[done] beta={r['beta']} seed={r['seed']} "
                      f"<P>={r['P_mean']:.5f} ({r['status']})")
            except Exception as e:
                print(f"[FAIL] beta={b} seed={s}: {e!r} -- re-run to resume")

    # ---- across-seed aggregation = the M1 systematic ----
    print("\n" + "=" * 78)
    print("ACROSS-SEED SYSTEMATIC (run-to-run spread)  ->  M1")
    print("=" * 78)
    print(f"{'beta':>5}{'#seeds':>7}{'<P> mean':>11}{'sys(P)':>10}"
          f"{'within-run':>12}{'R_AL mean':>11}{'sys(R_AL)':>11}")
    rows = []
    seed_runs = {}
    for b in sorted(BETAS):
        files = sorted(__import__("glob").glob(
            os.path.join(OUTDIR, f"wloop_N6_b{b}_s*.npz")))
        if not files:
            print(f"{b:>5}  (no completed seeds)"); continue
        Pm, Rm, wre = [], [], []
        for f in files:
            d = np.load(f)
            Pm.append(float(d["P_mean"])); Rm.append(float(d["R_AL"]))
            wre.append(float(d["P_err"]))
        Pm, Rm = np.array(Pm), np.array(Rm)
        sysP = Pm.std(ddof=1) if len(Pm) > 1 else float("nan")
        sysR = Rm.std(ddof=1) if len(Rm) > 1 else float("nan")
        print(f"{b:>5}{len(files):>7}{Pm.mean():>11.5f}{sysP:>10.5f}"
              f"{np.mean(wre):>12.5f}{Rm.mean():>11.4f}{sysR:>11.4f}")
        rows.append([b, len(files), Pm.mean(), sysP, np.mean(wre), Rm.mean(), sysR])
        seed_runs[b] = files

    if rows:
        with open(os.path.join(OUTDIR, "systematic_summary.csv"), "w",
                  encoding="utf-8") as f:
            f.write("beta,n_seeds,P_mean,sys_P,within_run_err,R_AL_mean,sys_R_AL\n")
            for r in rows:
                f.write(f"{r[0]},{r[1]},{r[2]:.6f},{r[3]:.6f},{r[4]:.6f},"
                        f"{r[5]:.6f},{r[6]:.6f}\n")
        print(f"\nWrote {OUTDIR}/systematic_summary.csv")
        # ready-to-paste block for error_treatment.py
        print("\nPaste into error_treatment.py to combine stat (+) sys:")
        print("SEED_RUNS = {")
        for b, files in seed_runs.items():
            inner = ", ".join(repr(os.path.join(OUTDIR, os.path.basename(x)))
                              for x in files)
            print(f"    {b}: [{inner}],")
        print("}")
        print("\nReminder: sys(P) here is the run-to-run band. error_treatment.py")
        print("combines it with the autocorrelation-corrected statistical error.")


if __name__ == "__main__":
    main()
