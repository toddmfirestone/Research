"""
Cl(6) Lattice QCD — Paper 2, Block 5, Step 4b  ·  PRODUCTION RUN  (v2)
===============================================================================
Grade-projected pion correlators on thermalised SU(3) gauge ensembles.
"Experiment A"

CHANGELOG v1 -> v2:
  BUG FIX  The frozen-staple parallel sweep in v1 violates detailed balance:
           all V links in a direction were proposed simultaneously using a
           single pre-computed staple, so accepting link x did not update the
           staple used to accept link x'. This produces a permanently low
           plaquette (~5% at beta=4.5, ~2.4% at beta=6.0) that does NOT go
           away with more sweeps. Diagnosed by thermalization probe run.
  FIX      Replaced mc_sweep (frozen-staple) with mc_sweep_eo:
           even/odd checkerboard within each direction. Even-parity links are
           updated using staples that include the current (just-updated) odd
           links; odd-parity links are updated with the now-corrected even
           links. Each half-step satisfies detailed balance exactly.
           Verified: E-O + 500 sw -> <P>=0.340 at beta=4.5 (0.52% from 0.338).
                     E-O + 800 sw -> <P>=0.590 at beta=6.0 (0.63% from 0.593).
  UPDATED  N_therm: beta=4.5 -> 1000 sweeps, beta=6.0 -> 800 sweeps.
  ADDED    Sweep-by-sweep plaquette history saved for all thermalization sweeps.
  ADDED    Equilibration gate: extends in 200-sweep increments if <P> (last 100
           sweeps) differs from Paper 1 target by more than 1% (max 3 ext.).
  UPDATED  eps tuning: beta=4.5 -> 0.40 (~70% acc); beta=6.0 -> 0.55 (~50%).

PARAMETERS (matching Computation Brief Section 7 and Paper 2 spec):
  Lattice:   N = 6  (V = 6^4 = 1296 sites, periodic BC)
  beta:      4.5 (confined), 6.0 (deconfined)
  m0:        0.05  ->  kappa = 1/(2*(m0+4)) = 0.12346  (r=1 fixed)
  N_therm:   1000 sweeps (beta=4.5), 800 sweeps (beta=6.0)
  N_cfg:     20 configurations per beta
  N_decorr:  5 sweeps between configurations
  N_src:     24 exact point sources (all 8 spin x 3 color at origin)
  CG tol:    1e-8, max 500 iterations

OUTPUTS (written to ./cl6_block5_results.pkl):
  results[beta] dict keys:
    frac_mean, frac_err    -- jackknife grade fractions, shape (N_lat,) each
    frac_raw               -- per-config fractions, shape (N_cfg, N_lat)
    Ctot_mean              -- mean total correlator, shape (N_lat,)
    plaq_therm_hist        -- full sweep-by-sweep plaquette during thermalization
    acc_therm_hist         -- full sweep-by-sweep acceptance during thermalization
    plaq_cfg_hist          -- plaquette at each production config
    delta_max              -- max completeness deviation over all configs/t
    gate_ok                -- bool: equilibration gate passed
    n_extensions           -- number of 200-sweep extensions needed
    cg_iters               -- CG iteration counts per config
    + full parameter record (N_therm, N_cfg, kappa, m0, beta, N_lat, eps)

ESTIMATED WALL-CLOCK: ~20-25 min
  (800-1000 E-O sweeps x 35ms + 20cfg x 24src x 0.4s) x 2 beta

DEPENDENCIES: Python >= 3.11, NumPy >= 2.2  (no other dependencies)
"""

import numpy as np
import time
import pickle
import sys


# ===============================================================================
# 1.  ALGEBRA SETUP
# ===============================================================================

sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)
I3 = np.eye(3, dtype=complex)
I8 = np.eye(8, dtype=complex)

# Cl(6) basis vectors via Pauli tensor products (Paper 1 Eq. 2.2)
e = [None] * 7
e[1] = np.kron(np.kron(sx, I2), I2)
e[2] = np.kron(np.kron(sy, I2), I2)
e[3] = np.kron(np.kron(sz, sx), I2)
e[4] = np.kron(np.kron(sz, sy), I2)
e[5] = np.kron(np.kron(sz, sz), sx)
e[6] = np.kron(np.kron(sz, sz), sy)

g5 = e[1] @ e[2] @ e[3] @ e[4]                     # gamma_5 = e1 e2 e3 e4

# Wilson-Dirac spin projectors P+/-_mu = (1/2)(I8 +/- e_mu)
Pm_s = np.array([0.5 * (I8 - e[mu + 1]) for mu in range(4)])   # (4,8,8)
Pp_s = np.array([0.5 * (I8 + e[mu + 1]) for mu in range(4)])   # (4,8,8)

# SU(3) grade projectors via Cartan hypercharge H = B12 + B34 + B56
# H eigenvalues {-3/2,-1/2,+1/2,+3/2} -> reps {1a, 3bar, 3, 1b}
B12 = (1j / 4) * (e[1] @ e[2] - e[2] @ e[1])
B34 = (1j / 4) * (e[3] @ e[4] - e[4] @ e[3])
B56 = (1j / 4) * (e[5] @ e[6] - e[6] @ e[5])
H   = B12 + B34 + B56


def lagrange_projector(lam, others):
    """Lagrange interpolation projector onto H-eigenspace eigenvalue lam."""
    P = I8.copy(); d = 1.0
    for mu in others:
        P = P @ (H - mu * I8); d *= (lam - mu)
    return P / d


P_3   = lagrange_projector( 1/2, [-3/2, -1/2,  3/2])   # triplet       dim=3
P_3b  = lagrange_projector(-1/2, [-3/2,  1/2,  3/2])   # anti-triplet  dim=3
P_1a  = lagrange_projector(-3/2, [-1/2,  1/2,  3/2])   # singlet I6=-i dim=1
P_1b  = lagrange_projector( 3/2, [-3/2, -1/2,  1/2])   # singlet I6=+i dim=1

PROJS  = [P_3, P_3b, P_1a, P_1b]
PNAMES = ['3', '3bar', '1a', '1b']
PREP   = {'3': '3', '3bar': '3bar', '1a': '1a', '1b': '1b'}

# 24x24 spin-colour projectors and gamma5
g5_24 = np.kron(g5, I3)
P24   = [np.kron(P, I3) for P in PROJS]


# ===============================================================================
# 2.  LATTICE GEOMETRY
# ===============================================================================

N_LAT  = 6
V      = N_LAT ** 4                                 # 1296 sites

coords = np.array(np.unravel_index(np.arange(V), [N_LAT] * 4)).T   # (V,4)

nf = np.zeros((4, V), dtype=int)                    # forward neighbours
nb = np.zeros((4, V), dtype=int)                    # backward neighbours
for mu in range(4):
    cf = coords.copy(); cf[:, mu] = (cf[:, mu] + 1) % N_LAT
    nf[mu] = np.ravel_multi_index(cf.T, [N_LAT] * 4)
    cb = coords.copy(); cb[:, mu] = (cb[:, mu] - 1) % N_LAT
    nb[mu] = np.ravel_multi_index(cb.T, [N_LAT] * 4)

t_idx  = coords[:, 0]                               # time-coordinate of each site
SRC    = 0                                           # point source at origin (0,0,0,0)

# Even/odd site parity -- used by the corrected E-O MC sweep
PARITY = coords.sum(axis=1) % 2                     # 0=even, 1=odd
IDX    = [np.where(PARITY == p)[0] for p in [0, 1]] # IDX[p] = site indices of parity p


# ===============================================================================
# 3.  WILSON-DIRAC OPERATOR  (BLAS-optimised, multi-RHS)
# ===============================================================================

def dw_opt(psi, U, m0, r=1.0):
    """
    Apply D_W to psi.

    psi : (V, 8, K)  where K = 3 * n_src  (packed multi-RHS)
    U   : (V, 4, 3, 3)
    Returns D_W psi, same shape.

    Spin projection via BLAS: Pm @ psi.T(1,0,2).reshape(8,-1)
    Colour gauge via batched np.matmul on (V, 8*n_src, 3).
    """
    K   = psi.shape[2]
    res = (m0 + 4.0 * r) * psi.copy()
    for mu in range(4):
        pf = psi[nf[mu]]; pb = psi[nb[mu]]; Ub = U[nb[mu], mu]

        # Spin projection (BLAS matmul via reshape)
        sf = (Pm_s[mu] @ pf.transpose(1, 0, 2).reshape(8, -1)).reshape(8, V, K).transpose(1, 0, 2)
        sb = (Pp_s[mu] @ pb.transpose(1, 0, 2).reshape(8, -1)).reshape(8, V, K).transpose(1, 0, 2)

        n_src = K // 3

        # Forward colour: result[v,i,a] = sum_b sf[v,i,b] * U[v,mu,a,b]
        sf_r  = sf.reshape(V, 8, 3, n_src).transpose(0, 1, 3, 2).reshape(V, 8 * n_src, 3)
        res  -= 0.5 * (np.matmul(sf_r, U[:, mu].transpose(0, 2, 1))
                         .reshape(V, 8, n_src, 3).transpose(0, 1, 3, 2).reshape(V, 8, K))

        # Backward colour: result[v,i,a] = sum_b sb[v,i,b] * U*[v,mu,b,a]  (U-dag)
        sb_r  = sb.reshape(V, 8, 3, n_src).transpose(0, 1, 3, 2).reshape(V, 8 * n_src, 3)
        res  -= 0.5 * (np.matmul(sb_r, Ub.conj())
                         .reshape(V, 8, n_src, 3).transpose(0, 1, 3, 2).reshape(V, 8, K))
    return res


def g5_apply(psi):
    """Apply gamma5 to psi (V,8,K) along spin axis only."""
    K = psi.shape[2]
    return (g5 @ psi.transpose(1, 0, 2).reshape(8, -1)).reshape(8, V, K).transpose(1, 0, 2)


def ddag_opt(psi, U, m0):
    """D-dagger = gamma5 D_W gamma5."""
    return g5_apply(dw_opt(g5_apply(psi), U, m0))


def ddagd_opt(psi, U, m0):
    """D-dagger D."""
    return ddag_opt(dw_opt(psi, U, m0), U, m0)


# ===============================================================================
# 4.  MULTI-RHS CG SOLVER
# ===============================================================================

def cg_mrhs(B, U, m0, tol=1e-8, maxiter=500):
    """
    Multi-RHS CG: solve D-dagger D X = B for all 24 sources simultaneously.
    B : (V, 8, 72). Returns (X, n_iterations).
    """
    X  = np.zeros_like(B); R = B.copy(); P = R.copy()
    r2 = float(np.real(np.vdot(R.ravel(), R.ravel()))); b2 = r2
    for k in range(maxiter):
        AP    = ddagd_opt(P, U, m0)
        pAp   = float(np.real(np.vdot(P.ravel(), AP.ravel())))
        alpha = r2 / pAp
        X    += alpha * P; R -= alpha * AP
        r2n   = float(np.real(np.vdot(R.ravel(), R.ravel())))
        if r2n / b2 < tol ** 2:
            return X, k + 1
        P     = R + (r2n / r2) * P; r2 = r2n
    print(f"  [WARN] CG did not converge in {maxiter} iters "
          f"(residual {np.sqrt(r2 / b2):.2e})", flush=True)
    return X, maxiter


def point_to_all_propagator(U, m0, src=0):
    """
    Compute full point-to-all propagator G(x; src) for all 24 spin-colour sources.

    All 24 RHS packed into one (V, 8, 72) array; one multi-RHS CG call.
    Returns G : (V, 24, 24)  and  nit : int.
    """
    n_src = 24
    S     = np.zeros((V, 8, 3 * n_src), dtype=complex)
    for j in range(n_src):
        spin_j, col_j = divmod(j, 3)
        S[src, spin_j, col_j * n_src + j] = 1.0
    B      = ddag_opt(S, U, m0)
    X, nit = cg_mrhs(B, U, m0)
    G      = X.reshape(V, 8, 3, n_src).reshape(V, 24, n_src)
    return G, nit


# ===============================================================================
# 5.  GRADE-PROJECTED PION CORRELATOR
# ===============================================================================

def grade_correlators(G):
    """
    C_pi(t)     = sum_{x:t(x)=t} ||g5_24 G(x;0)||^2_F
    C_pi^r(t)   = sum_{x:t(x)=t} ||(Pr_24 g5_24) G(x;0)||^2_F

    Completeness is algebraic: sum_r C_pi^r = C_pi exactly
    (since sum_r Pr_24 = I_24). Nonzero delta ~ 1e-16 = IEEE rounding only.

    G : (V, 24, 24). Returns C_tot (N_LAT,), C_grade dict.
    """
    C_tot   = np.zeros(N_LAT)
    C_grade = {nm: np.zeros(N_LAT) for nm in PNAMES}
    for t in range(N_LAT):
        mask = (t_idx == t); Gt = G[mask]
        g5G  = g5_24 @ Gt
        C_tot[t] = np.real(np.sum(np.abs(g5G) ** 2))
        for Pr, nm in zip(P24, PNAMES):
            C_grade[nm][t] = np.real(np.sum(np.abs((Pr @ g5_24) @ Gt) ** 2))
    return C_tot, C_grade


# ===============================================================================
# 6.  GAUGE MONTE CARLO  --  EVEN/ODD CHECKERBOARD  (corrected from v1)
# ===============================================================================
#
# v1 BUG: mc_sweep computed staple_all(U, mu) once for all V sites, then
# accepted/rejected all V links simultaneously. Accepting U[x,mu] does not
# update the staple used for U[x',mu] even when x' is a staple-neighbour of x.
# This "frozen-staple" update does NOT satisfy detailed balance. The resulting
# plaquette deficit (~5% at beta=4.5, ~2.4% at beta=6.0) is permanent.
#
# v2 FIX: even/odd checkerboard (mc_sweep_eo).
# For each direction mu and parity p in {0, 1}:
#   - Compute staple for all parity-p sites using CURRENT U.
#   - Propose and accept parity-p links in direction mu.
# Key property: the staple of U[x,mu] with parity(x)=p contains only
# U[y,mu] with parity(y)=1-p (at sites x+/-nu_hat). These are NOT being
# updated in this half-step, so the staple is current and correct for all
# parity-p sites simultaneously. Detailed balance is satisfied exactly.

_SU2_PAIRS = [(0, 1), (0, 2), (1, 2)]


def _staple_for_idx(U, mu, idx):
    """
    Wilson staple for sites in index array idx, direction mu.

    Sum_{nu != mu} [ U[x,nu] U[x+nu,mu] U[x+mu,nu]^dagger     (upper)
                   + U[x-nu,nu]^dagger U[x-nu,mu] U[x-nu+mu,nu] ] (lower)

    Delta_S = -(beta/3) Re Tr[(U_new - U_old) @ S^dagger]
    where S^dagger equals the correct K_{x,mu} (verified analytically).

    Returns S of shape (len(idx), 3, 3).
    """
    S = np.zeros((len(idx), 3, 3), dtype=complex)
    for nu in range(4):
        if nu == mu:
            continue
        xpmu     = nf[mu][idx]
        xpnu     = nf[nu][idx]
        xmnu     = nb[nu][idx]
        xpmu_mnu = nb[nu][nf[mu][idx]]
        t1 = np.matmul(np.matmul(U[idx, nu],  U[xpnu, mu]),
                       U[xpmu, nu].conj().swapaxes(1, 2))
        t2 = np.matmul(np.matmul(U[xmnu, nu].conj().swapaxes(1, 2), U[xmnu, mu]),
                       U[xpmu_mnu, nu])
        S += t1 + t2
    return S


def mc_sweep_eo(U, beta, eps, rng):
    """
    One full E-O Metropolis sweep over all links.

    For each direction mu and parity p:
      1. Compute staple for parity-p sites (uses CURRENT U -- correct).
      2. Propose 3 Cabibbo-Marinari SU(2) updates for each parity-p link.
      3. Accept/reject each link independently with min(1, exp(-Delta_S)).

    Returns: acceptance rate (float in [0, 1]).
    """
    acc = tot = 0
    for mu in range(4):
        for par in [0, 1]:
            idx = IDX[par]
            n   = len(idx)
            S   = _staple_for_idx(U, mu, idx)       # uses current U

            for i, j in _SU2_PAIRS:
                th  = rng.standard_normal((n, 3))
                th /= np.linalg.norm(th, axis=1, keepdims=True)
                ang = eps * rng.standard_normal(n)
                sv, cv = np.sin(ang), np.cos(ang)

                Vb = np.zeros((n, 3, 3), dtype=complex)
                Vb[:, 0, 0] = 1.0; Vb[:, 1, 1] = 1.0; Vb[:, 2, 2] = 1.0
                Vb[:, i, i] =  cv + 1j * sv * th[:, 2]
                Vb[:, i, j] =  sv * (1j * th[:, 0] + th[:, 1])
                Vb[:, j, i] =  sv * (1j * th[:, 0] - th[:, 1])
                Vb[:, j, j] =  cv - 1j * sv * th[:, 2]

                U_new = np.matmul(Vb, U[idx, mu])

                dS = -(beta / 3) * np.real(
                    np.trace(
                        np.matmul(U_new - U[idx, mu], S.conj().swapaxes(1, 2)),
                        axis1=1, axis2=2
                    )
                )
                rand   = rng.random(n)
                accept = (dS <= 0) | (rand < np.exp(np.clip(-dS, -50, 0)))
                U[idx[accept], mu] = U_new[accept]
                acc += accept.sum(); tot += n

    return acc / tot


def plaquette(U):
    """Average plaquette <P> = (1/6V) sum_{x,mu<nu} Re Tr[U_{mu nu}(x)] / 3."""
    P = 0.0
    for mu in range(4):
        for nu in range(mu + 1, 4):
            loop = np.matmul(
                np.matmul(
                    np.matmul(U[:, mu], U[nf[mu], nu]),
                    U[nf[nu], mu].conj().swapaxes(1, 2)
                ),
                U[:, nu].conj().swapaxes(1, 2)
            )
            P += np.real(np.trace(loop, axis1=1, axis2=2)).mean() / 3
    return P / 6


def rand_su3(rng):
    """Haar-random SU(3) matrix for hot-start initialisation."""
    Z    = (rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))) / np.sqrt(2)
    Q, R = np.linalg.qr(Z)
    Q    = Q @ np.diag(np.sign(np.diag(R)))
    return Q / np.linalg.det(Q) ** (1.0 / 3)


# ===============================================================================
# 7.  JACKKNIFE STATISTICS
# ===============================================================================

def jackknife(data):
    """
    Single-elimination jackknife over first axis.
    data : (N, ...) -> mean (...), err (...)
    """
    N        = data.shape[0]
    mean     = data.mean(axis=0)
    jk_means = np.array(
        [(data[:i].sum(axis=0) + data[i + 1:].sum(axis=0)) / (N - 1)
         for i in range(N)]
    )
    err = np.sqrt((N - 1) / N * ((jk_means - mean) ** 2).sum(axis=0))
    return mean, err


# ===============================================================================
# 8.  THERMALISATION WITH EQUILIBRATION GATE
# ===============================================================================

def thermalise(beta, n_therm_base, target_P, eps, seed,
               n_gate=100, tol_P=0.01, max_extensions=3,
               ext_size=200, log_every=100):
    """
    Hot-start thermalisation with E-O sweep and plaquette gate.

    Runs n_therm_base E-O sweeps. If <P> over the last n_gate sweeps
    deviates from target_P by more than tol_P (fractional), extends by
    ext_size sweeps and re-checks (up to max_extensions times).

    Returns:
        U            -- thermalised (V,4,3,3) gauge field
        plaq_hist    -- sweep-by-sweep plaquette, full thermalization
        acc_hist     -- sweep-by-sweep acceptance rate
        gate_ok      -- bool: gate passed
        n_extensions -- number of 200-sweep extensions used
    """
    rng = np.random.default_rng(seed)
    U   = np.zeros((V, 4, 3, 3), dtype=complex)
    for v in range(V):
        for mu in range(4):
            U[v, mu] = rand_su3(rng)

    plaq_hist = []; acc_hist = []
    n_done = 0; n_target = n_therm_base

    print(f"  target <P>={target_P:.5f}  tol=+-{tol_P*100:.0f}%  "
          f"eps={eps:.2f}  N_therm_base={n_therm_base}", flush=True)

    while True:
        t0 = time.perf_counter()
        while n_done < n_target:
            acc = mc_sweep_eo(U, beta, eps, rng)
            P   = plaquette(U)
            plaq_hist.append(P); acc_hist.append(acc)
            n_done += 1
            if n_done % log_every == 0:
                w   = np.array(plaq_hist[-100:])
                dev = abs(w.mean() - target_P) / target_P * 100
                print(f"  sweep {n_done:5d}/{n_target}: "
                      f"<P>={P:.5f}  avg100={w.mean():.5f}+/-{w.std()/10:.5f}  "
                      f"dev={dev:.2f}%  acc={acc:.1%}  "
                      f"t={time.perf_counter()-t0:.0f}s", flush=True)

        # Equilibration gate
        window = np.array(plaq_hist[-n_gate:])
        P_gate = window.mean(); P_err = window.std() / np.sqrt(n_gate)
        dev    = abs(P_gate - target_P) / target_P
        n_ext  = (n_target - n_therm_base) // ext_size

        status = "PASS" if dev <= tol_P else "FAIL"
        print(f"\n  Gate (last {n_gate} sw): "
              f"<P>={P_gate:.5f}+/-{P_err:.5f}  "
              f"dev={dev*100:.2f}%  {status}", flush=True)

        if dev <= tol_P:
            return U, plaq_hist, acc_hist, True, n_ext
        if n_ext >= max_extensions:
            print(f"  Max extensions ({max_extensions}) reached -- proceeding "
                  f"with systematic warning.", flush=True)
            return U, plaq_hist, acc_hist, False, n_ext

        n_target += ext_size
        print(f"  Extending +{ext_size} sweeps -> total {n_target}\n", flush=True)


# ===============================================================================
# 9.  MAIN PRODUCTION LOOP
# ===============================================================================

def run_production():

    m0       = 0.05
    kappa    = 1.0 / (2.0 * (m0 + 4.0))            # 0.12346
    N_CFG    = 20
    N_DECORR = 5

    # Per-beta parameters based on convergence study
    BETAS = [
        dict(beta=6.0, phase='Deconfined', n_therm=800,  eps=0.55,
             target_P=0.59334, seed=600042),
        dict(beta=4.5, phase='Confined',   n_therm=1000, eps=0.40,
             target_P=0.33816, seed=450042),
    ]

    results  = {}
    T_GLOBAL = time.perf_counter()

    print("=" * 66)
    print("Cl(6) Lattice QCD -- Block 5, Step 4b -- PRODUCTION RUN (v2)")
    print("=" * 66)
    print(f"  N={N_LAT}, m0={m0}, kappa={kappa:.5f}, "
          f"N_cfg={N_CFG}, N_decorr={N_DECORR}")
    print(f"  MC: even/odd checkerboard (v2 -- corrected from v1 frozen-staple)")
    print(f"  24 exact point sources, multi-RHS CG, jackknife errors\n")
    sys.stdout.flush()

    for bp in BETAS:
        beta     = bp['beta'];    phase    = bp['phase']
        n_therm  = bp['n_therm']; eps      = bp['eps']
        target_P = bp['target_P']; seed    = bp['seed']

        print(f"{'='*66}")
        print(f"beta={beta}  ({phase})  N_therm={n_therm}  eps={eps}  "
              f"{time.strftime('%H:%M:%S')}")
        sys.stdout.flush()

        # Thermalisation
        U, plaq_therm, acc_therm, gate_ok, n_ext = thermalise(
            beta, n_therm, target_P, eps, seed)

        # Production loop
        rng        = np.random.default_rng(seed + 1)
        cfg_Ctot   = []
        cfg_Cgrade = {nm: [] for nm in PNAMES}
        cfg_plaq   = []
        delta_max  = 0.0
        iters_list = []

        print(f"\n  Generating {N_CFG} configurations:")
        for cfg in range(N_CFG):
            for _ in range(N_DECORR):
                mc_sweep_eo(U, beta, eps, rng)

            P = plaquette(U); cfg_plaq.append(P)

            t_prop  = time.perf_counter()
            G, nit  = point_to_all_propagator(U, m0, src=SRC)
            t_prop  = time.perf_counter() - t_prop
            iters_list.append(nit)

            C_tot, C_grade = grade_correlators(G)

            Csum  = sum(C_grade[nm][1] for nm in PNAMES)
            delta = abs(C_tot[1] - Csum) / max(abs(C_tot[1]), 1e-30)
            delta_max = max(delta_max, delta)

            cfg_Ctot.append(C_tot.copy())
            for nm in PNAMES:
                cfg_Cgrade[nm].append(C_grade[nm].copy())

            print(f"  cfg {cfg+1:2d}/{N_CFG}: <P>={P:.5f}  "
                  f"nit={nit:3d}  t_prop={t_prop:.1f}s  "
                  f"delta(t=1)={delta:.2e}", flush=True)

        # Jackknife
        cfg_Ctot_arr = np.array(cfg_Ctot)
        frac_raw     = {}
        frac_mean    = {}; frac_err = {}
        for nm in PNAMES:
            arr          = np.array(cfg_Cgrade[nm])
            frac_raw[nm] = arr / cfg_Ctot_arr
            frac_mean[nm], frac_err[nm] = jackknife(frac_raw[nm])

        plaq_mean = float(np.mean(cfg_plaq))
        plaq_err  = float(np.std(cfg_plaq) / np.sqrt(N_CFG))

        # Per-beta summary
        print(f"\n  -- beta={beta} SUMMARY -----------------------------------------------")
        print(f"  <P> production = {plaq_mean:.5f} +/- {plaq_err:.5f}  "
              f"(target {target_P:.5f})")
        print(f"  Gate: {'PASS' if gate_ok else 'FAIL (systematic warning)'}  "
              f"({n_ext} extension(s))")
        print(f"  CG iterations: mean={np.mean(iters_list):.1f}  "
              f"min={min(iters_list)}  max={max(iters_list)}")
        print(f"  delta_max (completeness): {delta_max:.2e}  "
              f"[<1e-10: {'PASS' if delta_max < 1e-10 else 'FAIL'}]")
        print(f"  Grade fractions at t=1 (jackknife):")
        for nm in PNAMES:
            print(f"    {PREP[nm]:5s}: {frac_mean[nm][1]:.6f} +/- {frac_err[nm][1]:.6f}")
        fsum = sum(frac_mean[nm][1] for nm in PNAMES)
        print(f"    Sum  : {fsum:.10f}  [must = 1.0000000000]")
        sys.stdout.flush()

        results[beta] = {
            'phase'           : phase,
            'frac_mean'       : frac_mean,
            'frac_err'        : frac_err,
            'frac_raw'        : frac_raw,
            'Ctot_mean'       : cfg_Ctot_arr.mean(axis=0),
            'plaq_therm_hist' : plaq_therm,
            'acc_therm_hist'  : acc_therm,
            'plaq_cfg_hist'   : cfg_plaq,
            'plaq_mean'       : plaq_mean,
            'plaq_err'        : plaq_err,
            'delta_max'       : delta_max,
            'gate_ok'         : gate_ok,
            'n_extensions'    : n_ext,
            'cg_iters'        : iters_list,
            'N_therm'         : n_therm,
            'N_cfg'           : N_CFG,
            'N_decorr'        : N_DECORR,
            'kappa'           : kappa,
            'm0'              : m0,
            'beta'            : beta,
            'N_lat'           : N_LAT,
            'eps'             : eps,
        }

    # ===========================================================================
    # 10.  TABLE 6.1 AND SAVE
    # ===========================================================================

    t_total = time.perf_counter() - T_GLOBAL

    print(f"\n{'='*66}")
    print("TABLE 6.1  --  Grade Fractions  C_pi^(r) / C_pi")
    print(f"  N={N_LAT}, m0={m0}, kappa={kappa:.5f}, N_cfg={N_CFG}/beta, jackknife")
    print(f"  NOTE: The four Delta values encode ONE independent asymmetry.")
    print(f"  Anti-correlation forces Delta(3)=-Delta(3bar), Delta(1a)=-Delta(1b).")
    print(f"  Report as: 'single channel-pair asymmetry at X-sigma', not 4 effects.")
    print(f"{'─'*66}")

    print(f"{'Grade':6s} {'Rep':5s} | {'beta=4.5 (Conf)':22s} | "
          f"{'beta=6.0 (Deconf)':22s} | {'Delta':8s} {'(sigma)':7s}")
    print(f"{'─'*6} {'─'*5}─+─{'─'*22}─+─{'─'*22}─+─{'─'*16}")

    for nm in PNAMES:
        rep = PREP[nm]
        f45 = results[4.5]['frac_mean'][nm][1]; s45 = results[4.5]['frac_err'][nm][1]
        f60 = results[6.0]['frac_mean'][nm][1]; s60 = results[6.0]['frac_err'][nm][1]
        d   = f45 - f60
        sig = np.sqrt(s45 ** 2 + s60 ** 2)
        ns  = abs(d) / sig if sig > 0 else 0.0
        print(f"{nm:6s} {rep:5s} | {f45:.6f} +/- {s45:.6f}  | "
              f"{f60:.6f} +/- {s60:.6f}  | {d:+.5f} ({ns:.1f}s)")

    print(f"{'─'*6} {'─'*5}─+─{'─'*22}─+─{'─'*22}─+─{'─'*16}")
    for t_show in [1, 2, 3]:
        s45 = sum(results[4.5]['frac_mean'][nm][t_show] for nm in PNAMES)
        s60 = sum(results[6.0]['frac_mean'][nm][t_show] for nm in PNAMES)
        print(f"{'Sum':6s} {'check':5s} | t={t_show}: {s45:.10f}  | "
              f"t={t_show}: {s60:.10f}  | [= 1 algebraic]")

    print(f"\n{'─'*66}  Ensemble quality:")
    for bp in BETAS:
        b = bp['beta']
        r = results[b]
        print(f"  beta={b}: <P>={r['plaq_mean']:.5f}+/-{r['plaq_err']:.5f}  "
              f"gate={'PASS' if r['gate_ok'] else 'FAIL'}  "
              f"delta_max={r['delta_max']:.2e}")

    print(f"\n  Wall time: {t_total:.0f} s  ({t_total/60:.1f} min)")

    out_path = 'cl6_block5_results.pkl'
    with open(out_path, 'wb') as fh:
        pickle.dump(results, fh)
    print(f"\n  Saved -> {out_path}")
    print(f"  Load:  import pickle; r = pickle.load(open('{out_path}','rb'))")
    print(f"  Keys:  r[4.5], r[6.0]  ->  frac_mean, frac_err, frac_raw,")
    print(f"         plaq_therm_hist, plaq_cfg_hist, delta_max, gate_ok, ...")
    print(f"{'='*66}")


# ===============================================================================
if __name__ == '__main__':
    run_production()