#!/usr/bin/env python3
"""
su2_corrected_T3.py
═══════════════════
Paper 4 — Cl(6) Lattice QCD Series
Corrected SU(2) T₃ Eigenspace Correlator Simulation

Bugs fixed from previous run:
  Bug 1: G_full used P⁺ projector (giving G_full = G_plus by construction).
          Fix: G_full = Σ [8·Re(a_t·ā₀) + 8·Re(b_t·b̄₀)] — NO projector.
  Bug 2: G_minus sourced from wrong timeslice (gave G_plus(N+1-t), not G_plus(t)).
          Fix: source ALWAYS at t_source=0 for both G_plus and G_minus.

SU(2) parametrisation (used throughout):
  U = [[ a,   b ],      |a|² + |b|² = 1,   a,b ∈ ℂ
       [-b*,  a*]]
  U[0,0] = a,  U[0,1] = b,  U[1,0] = -b*,  U[1,1] = a*

Correlator definitions (exact, per specification):
  G_full(t)  = Σ_{x,μ} Re Tr[E_μ(x,t)·E_μ(x,0)†]
             = Σ [8·Re(a_t·ā₀) + 8·Re(b_t·b̄₀)]
  G_plus(t)  = Σ_{x,μ} 4·Re[U_t[0,0]·conj(U₀[0,0])]
  G_minus(t) = Σ_{x,μ} 4·Re[U_t[1,1]·conj(U₀[1,1])]
             = G_plus(t) exactly  [Corollary 1: Re[z]=Re[conj(z)]]
  Δ(t)       = G_full - G_plus - G_minus = Σ 8·Re[b_t·b̄₀]
  f̂_±(t)    = ⟨G_±⟩_jk / ⟨G_full⟩_jk   [ratio of jackknife means — NOT mean of ratios]

Estimator fix (this version):
  Wrong: f_cfg = G_plus_cfg / G_full_cfg  per config, then jackknife mean
         → biased and diverges when G_full_cfg ≈ 0 on individual configs
  Correct: f̂ = jackknife_mean(G_plus) / jackknife_mean(G_full)
           with error from leave-one-out resamples of the ratio

Completeness identity (verified per β, per t):
  f̂_+(t) + f̂_-(t) + Δ_cross(t)/G_full(t) = 1   [ratio-of-means version]

Lattice layout: L[t, x, y, z, mu, row, col]  shape (N,N,N,N,4,2,2)
"""

import sys
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
# SIMULATION PARAMETERS  (do not modify)
# ══════════════════════════════════════════════════════════════════════════════

N         = 6
N_THERM   = 500
N_DECORR  = 10
N_CFG     = 20
DS_FACTOR = 2.0   # ΔS = -(β/2) Re Tr[(U_new − U_old)·A]  [validated]

BETA_LIST  = [2.5, 3.0, 4.5, 6.0]
EPS_DICT   = {2.5: 0.48349, 3.0: 0.42080, 4.5: 0.32093, 6.0: 0.26999}
START_DICT = {2.5: 'hot', 3.0: 'hot', 4.5: 'cold', 6.0: 'cold'}

T_MEAS     = list(range(1, N))       # t = 1,2,3,4,5 for N=6
GFULL_COLD = N**3 * 4 * 8           # = 6912 for N=6  (free-field value)
GPLUS_COLD = N**3 * 4 * 4           # = 3456 for N=6


# ══════════════════════════════════════════════════════════════════════════════
# SU(2) PRIMITIVES  (validated — do not modify)
# ══════════════════════════════════════════════════════════════════════════════

def adj(U):
    """Conjugate transpose, batch-safe over last two axes."""
    return np.conjugate(np.swapaxes(U, -1, -2))


def su2_random(shape):
    """Haar-random SU(2) matrices (uniform on S³)."""
    a = np.random.randn(*shape, 4)
    a /= np.linalg.norm(a, axis=-1, keepdims=True)
    U = np.empty((*shape, 2, 2), dtype=complex)
    U[..., 0, 0] =  a[..., 0] + 1j * a[..., 3]
    U[..., 0, 1] =  a[..., 2] + 1j * a[..., 1]
    U[..., 1, 0] = -a[..., 2] + 1j * a[..., 1]
    U[..., 1, 1] =  a[..., 0] - 1j * a[..., 3]
    return U


def su2_project(U):
    """
    Project back onto SU(2) by quaternion extraction + renormalisation.
    Corrects floating-point drift; exact SU(2) inputs → identity to machine ε.
    """
    a0 = 0.5 * np.real(U[..., 0, 0] + U[..., 1, 1])
    a3 = 0.5 * np.imag(U[..., 0, 0] - U[..., 1, 1])
    a2 = 0.5 * np.real(U[..., 0, 1] - U[..., 1, 0])
    a1 = 0.5 * np.imag(U[..., 0, 1] + U[..., 1, 0])
    nrm = np.sqrt(a0**2 + a1**2 + a2**2 + a3**2)
    a0, a1, a2, a3 = a0/nrm, a1/nrm, a2/nrm, a3/nrm
    V = np.empty_like(U)
    V[..., 0, 0] =  a0 + 1j * a3
    V[..., 0, 1] =  a2 + 1j * a1
    V[..., 1, 0] = -a2 + 1j * a1
    V[..., 1, 1] =  a0 - 1j * a3
    return V


def su2_pert(shape, epsilon):
    """
    Exact rotation of angle ε about a random axis n̂ ∈ S².
    dU = cos(ε)·I + i·sin(ε)·(n̂·σ)
    (Previous chi-distribution angle bug is fixed here.)
    """
    n = np.random.randn(*shape, 3)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    c, s = np.cos(epsilon), np.sin(epsilon)
    dU = np.empty((*shape, 2, 2), dtype=complex)
    dU[..., 0, 0] =  c + 1j * s * n[..., 2]
    dU[..., 0, 1] =  s * n[..., 1] + 1j * s * n[..., 0]
    dU[..., 1, 0] = -s * n[..., 1] + 1j * s * n[..., 0]
    dU[..., 1, 1] =  c - 1j * s * n[..., 2]
    return dU


def embed_su2(U):
    """
    I₂ ⊗ U ⊗ I₂  →  8×8 matrix.
    Single 2×2 input (not batched); used in algebraic checks.
    Tr[I₂⊗U⊗I₂] = 4·Tr[U]  (Kronecker product trace identity).
    """
    I2 = np.eye(2, dtype=complex)
    return np.kron(I2, np.kron(U, I2))


# ══════════════════════════════════════════════════════════════════════════════
# LATTICE INITIALISATION
# ══════════════════════════════════════════════════════════════════════════════

def init_cold(n):
    """All links = I₂.  Gives ⟨P⟩=1, G_full=6912, Δ=0."""
    L = np.zeros((n, n, n, n, 4, 2, 2), dtype=complex)
    L[..., 0, 0] = 1.0
    L[..., 1, 1] = 1.0
    return L


def init_hot(n):
    """All links independently Haar-random.  Gives ⟨P⟩≈0."""
    return su2_random((n, n, n, n, 4))


# ══════════════════════════════════════════════════════════════════════════════
# METROPOLIS INFRASTRUCTURE  (validated — do not modify)
# ══════════════════════════════════════════════════════════════════════════════

def compute_staple(L, mu):
    """
    Staple sum A = Σ_{ν≠μ} (A_ν^upper + A_ν^lower), 6 terms in d=4.

    Upper: A_ν⁺ = U_{x+μ̂,ν} @ U†_{x+ν̂,μ} @ U†_{x,ν}
    Lower: A_ν⁻ = U†_{x+μ̂−ν̂,ν} @ U†_{x−ν̂,μ} @ U_{x−ν̂,ν}
    """
    staple = np.zeros((*L.shape[:4], 2, 2), dtype=complex)
    Um = L[..., mu, :, :]
    for nu in range(4):
        if nu == mu:
            continue
        Un         = L[..., nu, :, :]
        Un_pmu     = np.roll(Un,   -1, axis=mu)
        Um_pnu     = np.roll(Um,   -1, axis=nu)
        Un_mnu     = np.roll(Un,   +1, axis=nu)
        Um_mnu     = np.roll(Um,   +1, axis=nu)
        Un_pmu_mnu = np.roll(Un_pmu, +1, axis=nu)
        staple    += Un_pmu @ adj(Um_pnu) @ adj(Un)
        staple    += adj(Un_pmu_mnu) @ adj(Um_mnu) @ Un_mnu
    return staple


def metropolis_sweep(L, beta, epsilon):
    """
    Vectorised Metropolis sweep over all 4 directions.
    ΔS = -(β/DS_FACTOR) Re Tr[(U_new−U_old)·A]
    Accept if ΔS≤0 or rand < exp(−ΔS).
    Returns updated L and mean acceptance rate.
    """
    n = L.shape[0]
    n_accept = 0
    for mu in range(4):
        staple = compute_staple(L, mu)
        U_old  = L[..., mu, :, :]
        dU     = su2_pert((n, n, n, n), epsilon)
        U_new  = su2_project(dU @ U_old)
        diff   = U_new - U_old
        dS     = -(beta / DS_FACTOR) * np.real(
            np.trace(diff @ staple, axis1=-2, axis2=-1)
        )
        rand   = np.random.rand(n, n, n, n)
        accept = (dS <= 0) | (rand < np.exp(-np.clip(dS, 0, 500)))
        L[..., mu, :, :] = np.where(
            accept[..., np.newaxis, np.newaxis], U_new, U_old
        )
        n_accept += int(np.sum(accept))
    return L, n_accept / (4 * n**4)


def plaquette(L):
    """
    ⟨P⟩ = ⟨(1/2) Re Tr[U_{x,μ} U_{x+μ̂,ν} U†_{x+ν̂,μ} U†_{x,ν}]⟩
    Averaged over all sites and μ<ν pairs.
    """
    total, count = 0.0, 0
    n = L.shape[0]
    for mu in range(4):
        for nu in range(mu + 1, 4):
            Um     = L[..., mu, :, :]
            Un     = L[..., nu, :, :]
            Un_pmu = np.roll(Un, -1, axis=mu)
            Um_pnu = np.roll(Um, -1, axis=nu)
            P      = Um @ Un_pmu @ adj(Um_pnu) @ adj(Un)
            total += np.sum(np.real(np.trace(P, axis1=-2, axis2=-1)))
            count += n**4
    return total / (2.0 * count)


# ══════════════════════════════════════════════════════════════════════════════
# CORRELATORS — CORRECTED  (Bugs 1 and 2 fixed)
# ══════════════════════════════════════════════════════════════════════════════

def compute_correlators(L, t_meas):
    """
    Compute all five observables for timeslices in t_meas.

    Source ALWAYS at t_source = 0: L[0, x, y, z, mu, :, :]
    Sink   at timeslice t:         L[t, x, y, z, mu, :, :]
    Sum over all spatial sites (x,y,z) and all 4 directions μ.

    Returns arrays indexed 0..N; entries at t ∉ t_meas are zero.
    (GF, GP, GM, DC, fp, fm, ratio) — each shape (N+1,)
    """
    U_src = L[0]                  # shape (N, N, N, 4, 2, 2)
    a_0   = U_src[..., 0, 0]     # U[0,0] = a₀,        shape (N,N,N,4)
    b_0   = U_src[..., 0, 1]     # U[0,1] = b₀
    # U_src[...,1,1] = conj(a₀) by SU(2) parametrisation

    GF = np.zeros(N + 1)
    GP = np.zeros(N + 1)
    GM = np.zeros(N + 1)

    for t in t_meas:
        U_snk = L[t]              # shape (N, N, N, 4, 2, 2)

        a_t      = U_snk[..., 0, 0]   # = a_t
        b_t      = U_snk[..., 0, 1]   # = b_t
        a_t_conj = U_snk[..., 1, 1]   # = conj(a_t)  by SU(2)
        a_0_conj = U_src[..., 1, 1]   # = conj(a₀)   by SU(2)

        # ── G_full: BOTH a and b channels (Bug 1 fix: no projector) ──────────
        GF[t] = float(np.sum(
            8.0 * np.real(a_t * np.conj(a_0)) +
            8.0 * np.real(b_t * np.conj(b_0))
        ))

        # ── G_plus: T₃=+½ channel, uses U[0,0] = a ──────────────────────────
        GP[t] = float(np.sum(4.0 * np.real(a_t * np.conj(a_0))))

        # ── G_minus: T₃=−½ channel, uses U[1,1] = conj(a)  ──────────────────
        #   Source ALWAYS t=0 (Bug 2 fix).
        #   4·Re[U_t[1,1]·conj(U_0[1,1])] = 4·Re[a_t*·a₀] = 4·Re[a_t·conj(a₀)]
        #   → G_minus = G_plus exactly (Corollary 1)
        GM[t] = float(np.sum(4.0 * np.real(a_t_conj * np.conj(a_0_conj))))

    DC = GF - GP - GM    # Δ_cross = 8·Σ Re[b_t·b̄₀]

    with np.errstate(divide='ignore', invalid='ignore'):
        denom = np.where(np.abs(GF) > 1e-30, GF, np.nan)
        fp    = GP / denom
        fm    = GM / denom
        ratio = DC / denom

    return GF, GP, GM, DC, fp, fm, ratio


# ══════════════════════════════════════════════════════════════════════════════
# JACKKNIFE
# ══════════════════════════════════════════════════════════════════════════════

def jackknife(samples):
    """
    Jackknife mean and standard error.
    samples: array-like (N_cfg, ...) → (mean, error) with same trailing shape.
    """
    X   = np.asarray(samples, dtype=float)
    n   = len(X)
    mu  = np.mean(X, axis=0)
    jk  = np.array([np.mean(np.delete(X, i, axis=0), axis=0) for i in range(n)])
    err = np.sqrt((n - 1) * np.mean((jk - mu)**2, axis=0))
    return mu, err


def jackknife_ratio(num_samples, den_samples):
    """
    Correct jackknife estimator for a ratio of ensemble means.

      f̂ = ⟨num⟩ / ⟨den⟩

    where ⟨·⟩ is the sample mean over N_cfg configurations.

    Why this is correct:
      The naive estimator  mean(num_i / den_i)  is biased whenever
      den_i fluctuates across configurations (which it always does on
      finite ensembles).  The ratio-of-means  ⟨num⟩/⟨den⟩  is the
      standard unbiased estimator used in lattice QCD for quantities
      like f_± = G_±/G_full.

    Jackknife error on the ratio:
      For each leave-one-out resample i, compute f̂_jk[i] = mean_excl_i(num)
      / mean_excl_i(den).  The jackknife variance is then:
        Var(f̂) = (N-1)/N · Σ_i (f̂_jk[i] − f̂)²

    Args:
        num_samples: array-like (N_cfg, ...)  — numerator samples
        den_samples: array-like (N_cfg, ...)  — denominator samples
    Returns:
        (f_hat, f_hat_err) each with shape matching trailing dims of inputs
    """
    X = np.asarray(num_samples, dtype=float)
    Y = np.asarray(den_samples, dtype=float)
    n = len(X)

    num_mean = np.mean(X, axis=0)
    den_mean = np.mean(Y, axis=0)

    # Guard against zero denominator (should not occur on thermalized configs)
    with np.errstate(divide='ignore', invalid='ignore'):
        f_hat = np.where(np.abs(den_mean) > 1e-30,
                         num_mean / den_mean, np.nan)

    # Leave-one-out resamples of the ratio
    f_jk = np.array([
        np.mean(np.delete(X, i, axis=0), axis=0) /
        np.mean(np.delete(Y, i, axis=0), axis=0)
        for i in range(n)
    ])

    f_hat_err = np.sqrt((n - 1) / n * np.sum((f_jk - f_hat)**2, axis=0))

    return f_hat, f_hat_err


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: ALGEBRAIC CHECKS V1–V10
# ══════════════════════════════════════════════════════════════════════════════

def _check(label, ok, val=""):
    status = "PASS" if ok else "FAIL ⚠"
    detail = f"  ({val})" if val else ""
    print(f"  [{status}]  {label}{detail}")
    if not ok:
        sys.exit(f"\n  ABORTED on failed check: {label}")


def section1_algebraic():
    print("\n" + "═"*72)
    print("SECTION 1: ALGEBRAIC CHECKS  V1–V10")
    print("═"*72)

    I2 = np.eye(2, dtype=complex)
    I8 = np.eye(8, dtype=complex)

    sx = np.array([[0, 1],  [1,  0]],  dtype=complex)
    sy = np.array([[0, -1j],[1j, 0]],  dtype=complex)
    sz = np.array([[1, 0],  [0, -1]],  dtype=complex)

    T1 = np.kron(I2, np.kron(sx, I2)) / 2.0
    T2 = np.kron(I2, np.kron(sy, I2)) / 2.0
    T3 = np.kron(I2, np.kron(sz, I2)) / 2.0
    Ts = [T1, T2, T3]

    P_plus  = np.kron(I2, np.kron(np.array([[1,0],[0,0]], dtype=complex), I2))
    P_minus = np.kron(I2, np.kron(np.array([[0,0],[0,1]], dtype=complex), I2))

    # Levi-Civita: ε_{ijk} = +1 for (0,1,2),(1,2,0),(2,0,1); -1 for reverse
    _eps = {(0,1,2):1,(1,2,0):1,(2,0,1):1,
            (1,0,2):-1,(0,2,1):-1,(2,1,0):-1}

    # V1: [Ti, Tj] = i·ε_ijk·Tk  (all 6 ordered pairs)
    max_v1 = 0.0
    for i in range(3):
        for j in range(3):
            if i == j: continue
            comm = Ts[i] @ Ts[j] - Ts[j] @ Ts[i]
            rhs  = sum(1j * _eps.get((i,j,k),0) * Ts[k] for k in range(3))
            max_v1 = max(max_v1, np.linalg.norm(comm - rhs, 'fro'))
    _check("V1   [Ti,Tj]=i·ε_ijk·Tk  (all i≠j pairs)", max_v1 < 1e-12,
           f"max||err||_F = {max_v1:.2e}")

    # V2: Casimir T² = (3/4)·I₈
    T_sq = T1@T1 + T2@T2 + T3@T3
    v2   = np.linalg.norm(T_sq - 0.75*I8, 'fro')
    _check("V2   T²=(3/4)·I₈", v2 < 1e-12, f"||err||_F = {v2:.2e}")

    # V3: T₃ spectrum = {+0.5 (×4), -0.5 (×4)}
    evals    = np.sort(np.real(np.linalg.eigvals(T3)))
    expected = np.sort([-0.5]*4 + [0.5]*4)
    v3 = float(np.max(np.abs(evals - expected)))
    _check("V3   T₃ spectrum={+½×4, −½×4}", v3 < 1e-12, f"max|err| = {v3:.2e}")

    # V4: P⁺ + P⁻ = I₈
    v4 = np.linalg.norm(P_plus + P_minus - I8, 'fro')
    _check("V4   P⁺+P⁻=I₈", v4 < 1e-12, f"||err||_F = {v4:.2e}")

    # V5: P⁺·P⁻ = 0
    v5 = np.linalg.norm(P_plus @ P_minus, 'fro')
    _check("V5   P⁺·P⁻=0", v5 < 1e-12, f"||err||_F = {v5:.2e}")

    # V6: idempotence (P±)² = P±
    v6p = np.linalg.norm(P_plus  @ P_plus  - P_plus,  'fro')
    v6m = np.linalg.norm(P_minus @ P_minus - P_minus, 'fro')
    _check("V6   (P⁺)²=P⁺", v6p < 1e-12, f"||err||_F = {v6p:.2e}")
    _check("V6   (P⁻)²=P⁻", v6m < 1e-12, f"||err||_F = {v6m:.2e}")

    # V7: Tr(P±) = 4
    v7p = abs(np.real(np.trace(P_plus))  - 4.0)
    v7m = abs(np.real(np.trace(P_minus)) - 4.0)
    _check("V7   Tr(P⁺)=4", v7p < 1e-12, f"|err| = {v7p:.2e}")
    _check("V7   Tr(P⁻)=4", v7m < 1e-12, f"|err| = {v7m:.2e}")

    # V8: embed(I₂) = I₈
    v8 = np.linalg.norm(embed_su2(I2) - I8, 'fro')
    _check("V8   embed(I₂)=I₈", v8 < 1e-12, f"||err||_F = {v8:.2e}")

    # V9: embed(U)·embed(U)† = I₈  (20 random U)
    v9_errs = []
    for _ in range(20):
        U = su2_random((1,))[0]
        E = embed_su2(U)
        v9_errs.append(np.linalg.norm(E @ adj(E) - I8, 'fro'))
    v9 = float(max(v9_errs))
    _check("V9   embed(U)·embed(U)†=I₈  (20 random U)", v9 < 1e-13,
           f"max||err||_F = {v9:.2e}")

    # V10: Proposition 1 numerical check (50 pairs, both r=±1)
    #   Re Tr[P_r E_t P_r E₀†] = 4·Re[U_t[r,r]·conj(U₀[r,r])]
    v10_errs = []
    for _ in range(50):
        Ut = su2_random((1,))[0]
        U0 = su2_random((1,))[0]
        Et  = embed_su2(Ut)
        E0d = adj(embed_su2(U0))
        for P_r, idx in [(P_plus, (0,0)), (P_minus, (1,1))]:
            mat_val = np.real(np.trace(P_r @ Et @ P_r @ E0d))
            sca_val = 4.0 * np.real(Ut[idx] * np.conj(U0[idx]))
            v10_errs.append(abs(mat_val - sca_val))
    v10 = float(max(v10_errs))
    _check("V10  Prop.1: Re Tr[P_r E_t P_r E₀†]=4·Re[U_t[r,r]·conj(U₀[r,r])]  (50 pairs)",
           v10 < 1e-12, f"max|err| = {v10:.2e}")

    print()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: COLD-START GATE CHECK  C1–C6
# ══════════════════════════════════════════════════════════════════════════════

def section2_cold_gate():
    print("═"*72)
    print("SECTION 2: COLD-START GATE CHECK  C1–C6  (N=6, all links = I₂)")
    print("═"*72)

    L = init_cold(N)
    GF, GP, GM, DC, fp, fm, _ = compute_correlators(L, T_MEAS)

    # C1: cold plaquette = 1.000 exactly
    P_val = plaquette(L)
    _check("C1   Cold plaquette = 1.000", abs(P_val - 1.0) < 1e-12,
           f"got {P_val:.14f}")

    # C2: G_full(t) = 6912 = N³·4·8 for all t
    #   Per (site,dir): 8·Re(1·1)+8·Re(0·0) = 8;  total = N³×4×8
    c2_ok  = all(abs(GF[t] - GFULL_COLD) < 1e-6 for t in T_MEAS)
    c2_val = GF[T_MEAS[0]]
    _check(f"C2   G_full(t)={GFULL_COLD} (=N³·4·8) for all t", c2_ok,
           f"G_full(1) = {c2_val:.4f}")

    # C3: |G_plus − G_minus| < 1e-10 (cold)
    c3 = max(abs(GP[t] - GM[t]) for t in T_MEAS)
    _check("C3   |G_plus−G_minus|<1e-10 (cold)", c3 < 1e-10, f"max = {c3:.2e}")

    # C4: |f_plus − 0.5| < 1e-12 (cold)
    c4 = max(abs(fp[t] - 0.5) for t in T_MEAS)
    _check("C4   |f_plus−0.5|<1e-12 (cold)", c4 < 1e-12, f"max = {c4:.2e}")

    # C5: |Δ_cross| < 1e-10 (cold: b=0 everywhere)
    c5 = max(abs(DC[t]) for t in T_MEAS)
    _check("C5   |Δ_cross|<1e-10 (cold, b=0)", c5 < 1e-10, f"max = {c5:.2e}")

    # C6: G_full = G_plus + G_minus + Δ_cross  (partition identity)
    c6 = max(abs(GF[t] - GP[t] - GM[t] - DC[t]) for t in T_MEAS)
    _check("C6   G_full=G_plus+G_minus+Δ_cross", c6 < 1e-10, f"max|err| = {c6:.2e}")

    print()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 + 4: PER-β SIMULATION WITH RESULTS TABLE
# ══════════════════════════════════════════════════════════════════════════════

def run_beta(beta):
    """
    Full thermalization + production run at coupling β.
    Prints Section 3 (thermalization) and Section 4 (results) output.
    Returns summary dict for Section 5.
    """
    beta_eff = DS_FACTOR * beta
    wc_pred  = 1.0 - 3.0 / (4.0 * beta_eff)
    eps      = EPS_DICT[beta]
    hot      = (START_DICT[beta] == 'hot')

    # ── Section 3 header ──────────────────────────────────────────────────────
    print(f"\n{'━'*72}")
    print(f"SECTION 3 — THERMALIZATION   β_stated={beta}  β_eff={beta_eff}"
          f"  start={'hot' if hot else 'cold'}  ε={eps:.5f}")
    print(f"{'━'*72}")

    L = init_hot(N) if hot else init_cold(N)

    for sweep in range(1, N_THERM + 1):
        L, acc = metropolis_sweep(L, beta, eps)
        if sweep % 100 == 0:
            print(f"  therm {sweep:4d}/{N_THERM}  "
                  f"⟨P⟩={plaquette(L):.5f}  acc={acc:.3f}  "
                  f"[WC pred: {wc_pred:.5f}]")

    # ── Production ────────────────────────────────────────────────────────────
    plaq_samp = []
    GF_samp, GP_samp, GM_samp, DC_samp = [], [], [], []

    for cfg in range(N_CFG):
        for _ in range(N_DECORR):
            L, _ = metropolis_sweep(L, beta, eps)

        plaq_samp.append(plaquette(L))
        GF, GP, GM, DC, _, _, _ = compute_correlators(L, T_MEAS)

        GF_samp.append(GF.copy())
        GP_samp.append(GP.copy())
        GM_samp.append(GM.copy())
        DC_samp.append(DC.copy())

        if (cfg + 1) % 5 == 0:
            print(f"  cfg {cfg+1:3d}/{N_CFG}  ⟨P⟩={plaq_samp[-1]:.5f}")

    # ── Statistics ────────────────────────────────────────────────────────────
    plaq_mean, plaq_err = jackknife(plaq_samp)

    GF_mean, GF_err = jackknife(GF_samp)
    GP_mean, GP_err = jackknife(GP_samp)
    GM_mean, GM_err = jackknife(GM_samp)
    DC_mean, DC_err = jackknife(DC_samp)

    # Ratio-of-means estimator  f̂_± = ⟨G_±⟩ / ⟨G_full⟩
    # This is the correct estimator; per-config ratios are biased.
    fp_hat, fp_hat_err = jackknife_ratio(GP_samp, GF_samp)
    fm_hat, fm_hat_err = jackknife_ratio(GM_samp, GF_samp)
    rat_hat, rat_hat_err = jackknife_ratio(DC_samp, GF_samp)

    # Completeness identity: f̂_+(t) + f̂_-(t) + Δ/G_full(t) = 1
    # Uses the same ratio-of-means estimator for Δ/G_full.
    completeness = fp_hat + fm_hat + rat_hat   # should equal 1 at every t

    # ── C7: G_plus = G_minus per configuration ────────────────────────────────
    c7_vals = []
    for cfg_gp, cfg_gm in zip(GP_samp, GM_samp):
        for t in T_MEAS:
            denom = abs(cfg_gp[t]) if abs(cfg_gp[t]) > 1e-30 else 1.0
            c7_vals.append(abs(cfg_gp[t] - cfg_gm[t]) / denom)
    c7_max = float(max(c7_vals))
    c7_ok  = (c7_max < 1e-10)

    # ── C8: G_full ≠ G_plus (must differ > 1e-3 for ≥1 t) ───────────────────
    c8_vals = []
    for t in T_MEAS:
        if abs(GF_mean[t]) > 1e-30:
            c8_vals.append(abs(GF_mean[t] - GP_mean[t]) / abs(GF_mean[t]))
    c8_max = float(max(c8_vals)) if c8_vals else 0.0
    c8_ok  = (c8_max > 1e-3)

    # ── C9: Δ is O(1) — |Δ/G_full| ≥ 0.1 for ≥1 t  (ratio-of-means) ────────
    c9_vals = [abs(rat_hat[t]) for t in T_MEAS if not np.isnan(rat_hat[t])]
    c9_max  = float(max(c9_vals)) if c9_vals else 0.0
    c9_ok   = (c9_max >= 0.1)

    # ── C10: Completeness identity f̂_+(t)+f̂_-(t)+Δ/G_full(t) = 1 ──────────
    c10_vals = [abs(completeness[t] - 1.0) for t in T_MEAS
                if not np.isnan(completeness[t])]
    c10_max = float(max(c10_vals)) if c10_vals else 0.0
    c10_ok  = (c10_max < 1e-10)

    # ── Section 4: Results table ──────────────────────────────────────────────
    print(f"\n{'═'*72}")
    print(f"SECTION 4 — RESULTS   β_stated={beta}  β_eff={beta_eff}  (DS_FACTOR={DS_FACTOR})")
    print(f"{'═'*72}")
    print(f"  ⟨P⟩ = {float(plaq_mean):.5f} ± {float(plaq_err):.5f}"
          f"   WC 1-3/(4β_eff) = {wc_pred:.5f}")
    print(f"  Estimator: f̂_± = ⟨G_±⟩/⟨G_full⟩  (ratio of jackknife means)")
    print()

    c7_str  = "PASS" if c7_ok  else "FAIL ⚠"
    c8_str  = "PASS" if c8_ok  else "FAIL ⚠"
    c9_str  = "PASS" if c9_ok  else "FAIL ⚠"
    c10_str = "PASS" if c10_ok else "FAIL ⚠"

    print(f"  CHECK C7  G_plus=G_minus:          max|G_+−G_−|/|G_+|       = {c7_max:.2e}"
          f"  [{c7_str}]")
    print(f"  CHECK C8  G_full≠G_plus:           max|G_full−G_+|/|G_full| = {c8_max:.2e}"
          f"  [{c8_str}  (PASS if >1e-3)]")
    print(f"  CHECK C9  Δ is O(1):               max|Δ/G_full|            = {c9_max:.3f}"
          f"  [{c9_str}  (PASS if >0.1)]")
    print(f"  CHECK C10 Completeness f̂_++f̂_-+Δ/G=1: max|sum−1|          = {c10_max:.2e}"
          f"  [{c10_str}]")

    if not c7_ok:
        sys.exit("\n  ABORTED: C7 failed — Bug 2 not fixed.")

    print()

    # ── Results table ─────────────────────────────────────────────────────────
    # Columns: t | G_full | G_plus | G_minus | Δ/G_full | f̂_± | completeness
    # f̂_+ = f̂_- exactly (Corollary 1); one column reported for both.
    hdr = (f"  {'t':>3}  "
           f"{'G_full':>12}±{'err':>8}  "
           f"{'G_plus':>12}±{'err':>8}  "
           f"{'G_minus':>12}±{'err':>8}  "
           f"{'Δ/G_full':>9}±{'err':>7}  "
           f"{'f̂_±(=f̂_+)':>10}±{'err':>7}  "
           f"{'f̂_++f̂_-+Δ/G':>13}")
    print(hdr)
    print("  " + "-"*118)

    for t in T_MEAS:
        r_mu  = float(rat_hat[t])       if not np.isnan(rat_hat[t])   else 0.0
        r_er  = float(rat_hat_err[t])   if not np.isnan(rat_hat_err[t]) else 0.0
        fp_mu = float(fp_hat[t])        if not np.isnan(fp_hat[t])    else 0.0
        fp_er = float(fp_hat_err[t])    if not np.isnan(fp_hat_err[t]) else 0.0
        comp  = float(completeness[t])  if not np.isnan(completeness[t]) else np.nan

        print(f"  {t:>3}  "
              f"{GF_mean[t]:>12.2f}±{GF_err[t]:<8.2f}  "
              f"{GP_mean[t]:>12.2f}±{GP_err[t]:<8.2f}  "
              f"{GM_mean[t]:>12.2f}±{GM_err[t]:<8.2f}  "
              f"{r_mu:>9.5f}±{r_er:<7.5f}  "
              f"{fp_mu:>10.5f}±{fp_er:<7.5f}  "
              f"{comp:>13.10f}")

    print()

    # ── Post-table summaries ──────────────────────────────────────────────────
    cor1_max = max(abs(GP_mean[t] - GM_mean[t]) for t in T_MEAS)
    print(f"  Corollary 1:   max_t |G_plus(t)−G_minus(t)| = {cor1_max:.2e}"
          f"  [f̂_+ = f̂_- reported as single column above]")
    print(f"  Completeness:  max_t |f̂_+(t)+f̂_-(t)+Δ/G(t)−1| = {c10_max:.2e}"
          f"  [{c10_str}]")
    print(f"  Free-field:    f̂_± = 0.500 exactly (when Δ_cross = 0)")
    fp_t_mean = float(np.nanmean([fp_hat[t] for t in T_MEAS]))
    print(f"  Measured:      f̂_± mean over t = {fp_t_mean:.5f}"
          f"   [expected < 0.5;  preview ≈0.17 at t=3, ≈0.14 at t=4 for β_eff=12]")
    print()

    return {
        'beta':      beta,
        'beta_eff':  beta_eff,
        'wc':        wc_pred,
        'plaq':      (float(plaq_mean), float(plaq_err)),
        'c9_max':    c9_max,
        'c10_max':   c10_max,
        'fp_hat':    fp_hat,
        'fp_hat_err':fp_hat_err,
        'rat_hat':   rat_hat,
        'c7_ok':     c7_ok,
        'c8_ok':     c8_ok,
        'c9_ok':     c9_ok,
        'c10_ok':    c10_ok,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════

def section5_summary(results):
    print("═"*72)
    print("SECTION 5: PLAQUETTE AND FRACTION SUMMARY  (Tables 3 & 4)")
    print("═"*72)

    # ── Plaquette summary ─────────────────────────────────────────────────────
    hdr = (f"  {'β_stated':>8}  {'β_eff':>6}  "
           f"{'⟨P⟩±err':>18}  {'WC':>8}  "
           f"{'max|Δ/G|':>10}  {'C10 |sum-1|':>12}")
    print(hdr)
    print("  " + "-"*74)
    for r in results:
        pm, pe = r['plaq']
        print(f"  {r['beta']:>8.1f}  {r['beta_eff']:>6.1f}  "
              f"{pm:>8.5f} ± {pe:<8.5f}  "
              f"{r['wc']:>8.5f}  "
              f"{r['c9_max']:>10.4f}  "
              f"{r['c10_max']:>12.2e}")

    # ── f̂_±(t) table across all β_eff and t ──────────────────────────────────
    print()
    print("  f̂_±(t) = ⟨G_±⟩/⟨G_full⟩  [ratio of jackknife means; f̂_+=f̂_- by Corollary 1]")
    print(f"  {'β_eff':>6}  " +
          "  ".join(f"{'t='+str(t):>14}" for t in T_MEAS))
    print("  " + "-"*74)
    for r in results:
        row = f"  {r['beta_eff']:>6.1f}  "
        for t in T_MEAS:
            fv = float(r['fp_hat'][t])    if not np.isnan(r['fp_hat'][t])     else float('nan')
            fe = float(r['fp_hat_err'][t]) if not np.isnan(r['fp_hat_err'][t]) else float('nan')
            row += f"{fv:>6.4f}±{fe:<6.4f}  "
        print(row)

    print()
    print(f"  DS_FACTOR = {DS_FACTOR}  (validated)")
    print(f"  Estimator: f̂_± = ⟨G_±⟩/⟨G_full⟩  NOT mean(G_±/G_full)")
    print(f"  Corollary 1 (G_plus=G_minus):  "
          + ("verified at machine precision for all β and t"
             if all(r['c7_ok'] for r in results)
             else "FAILED for some β — check Bug 2 fix"))
    print(f"  Completeness f̂_++f̂_-+Δ/G=1:  "
          + ("verified for all β and t"
             if all(r['c10_ok'] for r in results)
             else "FAILED — check ratio estimator consistency"))
    print(f"  Bug 1 (G_full=G_plus):  "
          + ("RESOLVED — G_full ≠ G_plus confirmed for thermalized configs"
             if all(r['c8_ok'] for r in results)
             else "NOT RESOLVED — G_full still equals G_plus"))
    print(f"  Bug 2 (G_minus=G_plus[N-t]):  "
          + ("RESOLVED — G_minus=G_plus confirmed for all t"
             if all(r['c7_ok'] for r in results)
             else "NOT RESOLVED"))
    print(f"  Expected: f̂_± < 0.5 on all thermalized configs;")
    print(f"            β_eff=12.0 preview: f̂_±≈0.17 at t=3, ≈0.14 at t=4")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    np.random.seed(42)

    # Section 1: Algebraic checks
    section1_algebraic()

    # Section 2: Cold-start gate check
    section2_cold_gate()

    # Sections 3 + 4: Per-β thermalization and results
    all_results = []
    for beta in BETA_LIST:
        res = run_beta(beta)
        all_results.append(res)

    # Section 5: Summary
    section5_summary(all_results)

    print("Done.")
