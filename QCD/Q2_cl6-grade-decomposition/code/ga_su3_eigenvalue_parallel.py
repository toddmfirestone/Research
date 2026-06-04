#!/usr/bin/env python3
"""
GA-SU(3) DIRAC EIGENVALUES — HARDENED PARALLEL RUNNER
=====================================================
Replacement for ga_su3_eigenvalue.py, built to survive interruptions
(e.g. a Windows update reboot) and to use a multi-core machine.

WHAT'S NEW vs the original
--------------------------
1. INCREMENTAL SAVE + RESUME. Every config's eigenvalues are written to
   their own file the instant they finish. A reboot/crash costs at most
   one in-flight config per worker -- never the whole run. Re-running the
   script counts what's already saved and produces ONLY the deficit.
2. PARALLEL via INDEPENDENT MARKOV CHAINS. The configs in a phase form a
   Markov chain (config n+1 = config n + decorrelation sweeps), so a
   single chain cannot be parallelized. Instead we launch several
   INDEPENDENT chains (different seeds), each producing a share of the
   configs, then pool. This is standard, statistically valid ensemble
   generation and scales ~linearly with cores.
   (Each worker pins OMP_NUM_THREADS=1, as in the Paper 1 sweep.)
3. PER-PHASE with independent config counts (100 confined / 20 deconf).
4. eigsh ROBUSTNESS: larger ncv, a retry with more Lanczos vectors and a
   looser tolerance, then a graceful skip (the deficit logic regenerates
   skipped configs on the next run) — no single bad config kills the run.
5. OPTIONAL SHIFT-INVERT speedup (off by default). D†D's smallest
   eigenvalues are ~0.41 (well-conditioned, far from 0), so shift-invert
   at sigma=0 via an inner CG solve converges far faster than 'SM'.
   VALIDATE it against a few 'SM' configs before trusting it (it changes
   the solver). Toggle SHIFT_INVERT below.

REQUIRES ga_su3_lattice_v2.py and ga_su3_dirac_v3.py in the same folder.
ENV: Python 3.11, NumPy + SciPy.  Not run by Claude -- run locally.

WINDOWS TIP (the actual cause of the last failure): before a long run,
Settings -> Windows Update -> set Active Hours / Pause updates; and
  powercfg /change standby-timeout-ac 0
  powercfg /change monitor-timeout-ac 0
so the machine won't sleep or reboot mid-run. Launching from a plain
cmd/PowerShell window (not the VS Code terminal) also means closing the
editor can't take the job down.
"""

import os, sys, time, glob, uuid, traceback
import multiprocessing as mp
import numpy as np
from scipy.sparse.linalg import eigsh, LinearOperator, cg

# ============================================================
# CONFIG — edit here
# ============================================================
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'eigenvalue_results')

# Phases to run this invocation. Comment one out to run a single phase.
PHASES = {
    'confined':   dict(beta=4.5, N=6, n_configs=100),
    'deconfined': dict(beta=6.0, N=6, n_configs=20),
}
RUN_PHASES = ['confined', 'deconfined']   # which of the above to run now

M0          = 0.05      # matched mass, both phases (mass-matched comparison)
K_EIGEN     = 50        # lowest eigenvalues per config
EIGEN_TOL   = 1e-8
N_THERM     = 300       # thermalization sweeps per chain
N_DECORR    = 10        # sweeps between configs within a chain

N_WORKERS   = 10        # set to your physical core count
SHIFT_INVERT = True    # True = faster solver (validate first!); False = 'SM'

# ============================================================
# FRAMEWORK LOADER
# ============================================================
def load_framework():
    g = {}
    for fname in ['ga_su3_lattice_v2.py', 'ga_su3_dirac_v3.py']:
        path = os.path.join(BASE_DIR, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f'{fname} not found in {BASE_DIR}')
        src = open(path, encoding='utf-8').read().replace(
            "if __name__ == '__main__':", "if False:")
        exec(compile(src, path, 'exec'), g)
    return g

def pin_single_thread():
    for v in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS',
              'MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
        os.environ[v] = '1'

# ============================================================
# EIGENVALUE COMPUTATION (matrix-free, robust)
# ============================================================
def compute_eigenvalues(D, lat, k=50, tol=1e-8):
    """k smallest Dirac eigenvalues lambda = sqrt(eig(D†D)), matrix-free.
    Tries a robust 'SM' solve with retry; optional shift-invert."""
    n = lat.N**lat.D * D.n_dof

    def DdagD(v):
        return D._apply_Ddag(D.apply(v))
    op = LinearOperator((n, n), matvec=DdagD, dtype=complex)

    def _run(ncv, t, sigma_mode):
        if sigma_mode:
            # shift-invert at sigma=0: OPinv applies (D†D)^{-1} via CG.
            def opinv_mv(b):
                x, _ = cg(op, b, rtol=1e-7, maxiter=2000)
                return x
            opinv = LinearOperator((n, n), matvec=opinv_mv, dtype=complex)
            ev = eigsh(op, k=k, sigma=0.0, which='LM', OPinv=opinv,
                       ncv=ncv, tol=t, return_eigenvectors=False)
        else:
            ev = eigsh(op, k=k, which='SM', ncv=ncv, tol=t,
                       maxiter=10*n, return_eigenvectors=False)
        ev = np.sort(np.abs(ev.real))
        return np.sqrt(np.maximum(ev, 0.0))

    ncv1 = min(n-1, max(2*k + 10, 80))
    for attempt, (ncv, t) in enumerate([(ncv1, tol),
                                        (min(n-1, 2*ncv1), tol*10)]):
        try:
            lam = _run(ncv, t, SHIFT_INVERT)
            if len(lam) == k:
                return lam
        except Exception as e:
            if attempt == 1:
                print(f'      eigsh failed after retry: {e}')
    return np.array([])

# ============================================================
# ONE CHAIN (one worker): thermalize, then produce its share
# ============================================================
def run_chain(task):
    pin_single_thread()
    beta = task['beta']; N = task['N']; produce = task['produce']
    seed = task['seed']; wid = task['wid']; save_dir = task['save_dir']
    try:
        g = load_framework()
        np.random.seed(seed)
        cl  = g['CliffordCl6']()
        su3 = g['SU3Algebra'](eps=0.3, g=1.0)
        lat = g['Lattice'](N=N, D=4)
        fields = g['Fields'](lat, cl, su3, mode='hot')
        metro  = g['Metropolis'](fields, beta=beta, eps=0.25)
        for _ in range(N_THERM):
            metro.sweep()

        done = 0
        for i in range(produce):
            for _ in range(N_DECORR):
                metro.sweep()
            D = g['DiracOperator'](fields, m0=M0, r=1.0)
            tc = time.time()
            evals = compute_eigenvalues(D, lat, k=K_EIGEN, tol=EIGEN_TOL)
            dt = time.time() - tc
            if len(evals) == K_EIGEN:
                # incremental save: one unique file per config
                fn = os.path.join(save_dir, f'cfg_{uuid.uuid4().hex[:12]}.npy')
                np.save(fn, evals)
                done += 1
                print(f'  [w{wid:02d} b{beta}] cfg {i+1}/{produce}  '
                      f'lambda_1={evals[0]:.6f}  {dt:5.1f}s  (saved)')
            else:
                print(f'  [w{wid:02d} b{beta}] cfg {i+1}/{produce}  '
                      f'eigsh skipped  {dt:5.1f}s')
        return dict(wid=wid, beta=beta, done=done, status='ok')
    except Exception as e:
        return dict(wid=wid, beta=beta, done=0, status='error',
                    error=f'{e}\n{traceback.format_exc()[:300]}')

# ============================================================
# RESUME / POOL helpers
# ============================================================
def phase_dir(beta, N):
    d = os.path.join(OUTPUT_DIR, f'eig_b{beta}_N{N}')
    os.makedirs(d, exist_ok=True)
    return d

def count_saved(save_dir):
    return len(glob.glob(os.path.join(save_dir, 'cfg_*.npy')))

def pool_phase(beta, N, label):
    save_dir = phase_dir(beta, N)
    files = sorted(glob.glob(os.path.join(save_dir, 'cfg_*.npy')))
    if not files:
        print(f'  [{label}] nothing to pool'); return None
    arr = np.array([np.load(f) for f in files])      # (n_configs, K_EIGEN)
    master = os.path.join(OUTPUT_DIR, f'eigenvalues_b{beta}_N{N}.npy')
    np.save(master, arr)
    per_cfg_mean = arr.mean(axis=1)                  # <lambda> per config
    grand = per_cfg_mean.mean()
    err   = per_cfg_mean.std(ddof=1)/np.sqrt(len(per_cfg_mean)) \
            if len(per_cfg_mean) > 1 else float('nan')
    print(f'  [{label}] pooled {arr.shape[0]} configs x {arr.shape[1]} '
          f'eigenvalues -> {master}')
    print(f'  [{label}] <lambda> = {grand:.5f} ± {err:.5f}  '
          f'(mean of {K_EIGEN} lowest, over configs)')
    return dict(beta=beta, N=N, label=label, n_configs=arr.shape[0],
                mean_lambda=grand, mean_lambda_err=err, master=master)

# ============================================================
# MAIN
# ============================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print('='*64)
    print('DIRAC EIGENVALUES — HARDENED PARALLEL RUNNER')
    print(f'  M0={M0}  K_EIGEN={K_EIGEN}  N_WORKERS={N_WORKERS}  '
          f'SHIFT_INVERT={SHIFT_INVERT}')
    print('='*64)

    summary = []
    for phase in RUN_PHASES:
        spec = PHASES[phase]; beta = spec['beta']; N = spec['N']
        target = spec['n_configs']
        save_dir = phase_dir(beta, N)
        have = count_saved(save_dir)
        deficit = max(0, target - have)
        print(f'\n--- {phase} (beta={beta}, N={N}) ---')
        print(f'  target={target}  already saved={have}  to produce={deficit}')

        if deficit > 0:
            nchains = min(N_WORKERS, deficit)
            base, extra = divmod(deficit, nchains)
            tasks = []
            for c in range(nchains):
                tasks.append(dict(
                    beta=beta, N=N, wid=c,
                    produce=base + (1 if c < extra else 0),
                    seed=2024 + c*7919 + int(beta*1000),
                    save_dir=save_dir))
            with mp.Pool(processes=nchains) as pool:
                for r in pool.imap_unordered(run_chain, tasks):
                    if r['status'] == 'error':
                        print(f'  [w{r["wid"]:02d}] ERROR: {r["error"]}')
                    else:
                        print(f'  [w{r["wid"]:02d}] finished, {r["done"]} configs')
        # Pool whatever exists (saved + new). Re-run later to top up deficit.
        res = pool_phase(beta, N, phase)
        if res:
            summary.append(res)

    print('\n' + '='*64)
    print('SUMMARY')
    for r in summary:
        print(f'  {r["label"]:<11} beta={r["beta"]}  n={r["n_configs"]:>3}  '
              f'<lambda>={r["mean_lambda"]:.5f} ± {r["mean_lambda_err"]:.5f}')
    if len(summary) == 2:
        a, b = summary
        d = a['mean_lambda'] - b['mean_lambda']
        de = np.sqrt(a['mean_lambda_err']**2 + b['mean_lambda_err']**2)
        print(f'  Delta<lambda> = {a["label"]} - {b["label"]} = '
              f'{d:+.5f} ± {de:.5f}   (STATISTICAL ONLY)')
    print('='*64)
    print('Re-run this script to top up any deficit (it resumes).')

if __name__ == '__main__':
    mp.freeze_support()
    main()
