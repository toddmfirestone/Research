#!/usr/bin/env python3
"""
paper6_gauge_fix.py
═══════════════════════════════════════════════════════════════════════════════
Paper 6 — Landau Gauge-Fixing Infrastructure
Cl(6) Lattice QCD Series, Paper 6:
"Landau-Gauge W± Propagator from the b-Channel: Gauge Fixing, Gribov Copies,
 and the Cross-Projected Correlator in SU(2)"

Implements:
  F5  Steepest-descent (SD) Landau gauge-fixing (Cabibbo-Marinari local update)
  F6  Fourier-accelerated (FA) Landau gauge-fixing (FFT Laplacian preconditioning)
  F7  Gribov copy strategy: ≥5 independent random starts, select maximum F_L

Derivation sketch:
  Landau gauge condition: ∂_μ A_μ(x) = 0  (lattice version, equation 2.5)
  Implemented by maximising the gauge functional (equation 2.1):
    F_L[{g}] = Re Σ_{x,μ} Tr[U^g_μ(x)]
  Convergence criterion (equation 2.6):
    Θ = (1 / (d · N_c · V)) Σ_{x,a} [(∂_μ A_μ)^a(x)]²  < ε_gauge = 1e-14
  with d = 4 (FLAG-2: all 4 directions live), N_c = 2, V = N^4.

  SD algorithm (F5):
    At each site x, the optimal local gauge increment is h(x) = W†(x)/|W(x)|
    where W(x) = Σ_{μ=0}^{3} [U^g_μ(x) + (U^g_μ(x−μ̂))†]   (equation 2.10)
    Applied as checkerboard (even/odd parity) sweeps for vectorisation.
    Cumulative: g_cum(x) → h(x) · g_cum(x);  U^g_μ(x) → h(x) U^g_μ(x) h†(x+μ̂)
    Parity argument: even-parity updates do not share any link variable
    (forward links from even sites land at odd sites; backward links from even
    sites are forward links AT odd sites — no overlap within one parity pass).

  FA algorithm (F6):
    Instead of the local update, precondition the gradient with the lattice
    Laplacian in Fourier space (equation 2.16). Steps:
      1. Compute D^a(x) = (∂_μ A_μ)^a(x)
      2. D̃^a(p) = FFT_4D[D^a](p);  D̃^a(0,0,0,0) = 0 (zero-mode projection)
      3. ω̃^a(p) = D̃^a(p) / f(p);   f(p) = Σ_μ 4 sin²(π p_μ/N)
      4. ω^a(x) = IFFT_4D[ω̃^a](x)
      5. H(x) = (iα/2) ω^a(x) σ^a; h(x) = exp(H(x)) ∈ SU(2)
      6. g_cum(x) → h(x) · g_cum(x);  recompute U^g from U_raw and g_cum
    Convergence O(V^{1/2}) vs O(V) for SD. Preferred for N ≥ 12.

FLAG-2 compliance:
  All 4 link directions μ ∈ {0,1,2,3} are live. No temporal gauge pre-fixing.
  F_L^{max} = 2·d·V = 8V (d=4). Θ denominator = d · N_c · V = 8V.
  Documented in Appendix B, Table B.1.

Gauge field extraction (equation 2.4):
  A^a_μ(x) = (1/2i) Tr[T^a (U_μ(x) − U†_μ(x))],  T^a = σ^a/2
  For SU(2) with U = [[a, b], [−b*, a*]]:
    A^1_μ = Im(U[0,1])   (sin-component of b)
    A^2_μ = Re(U[0,1])   (cos-component of b)
    A^3_μ = Im(U[0,0])   (imaginary part of a)
  Derivation: Im(b) comes from Tr[σ¹(U−U†)]/4i; Re(b) from Tr[σ²(U−U†)]/4i;
              Im(a) from Tr[σ³(U−U†)]/4i (verified analytically in §2).

CR import discrepancies (documented, not silent):
  CR specifies: from paper4_simulation import load_ensemble, random_su2, su2_norm
    → None of these exist in paper4_simulation.py.
    → load_ensemble: implemented here from paper4/su2_corrected_T3 primitives.
    → random_su2: su2_corrected_T3.su2_random (correct Haar measure, S³ parametrisation).
    → su2_norm: implemented here (equation 2.13, sqrt(Tr[W†W]/2)).
  CR specifies: from su2_corrected_T3 import T_generators, project_su2
    → T_generators: not present; T^a = σ^a/2 encoded inline.
    → project_su2: su2_corrected_T3.su2_project (quaternion renormalisation).

Link array format (Paper 4/5 convention, preserved throughout):
  U shape: (N, N, N, N, 4, 2, 2)  — U[t, x, y, z, mu, row, col]
  g shape: (N, N, N, N, 2, 2)     — g[t, x, y, z, row, col]

Dependencies:
  numpy
  scipy.fft.fftn, ifftn  (F6 algorithm; N ≥ 12 only)
  su2_corrected_T3: su2_random, su2_project, adj, metropolis_sweep, plaquette

Seed isolation:
  §2 (gauge-fixing starts): seed = cfg_idx * 100 + start_idx + 222_000
  Ensemble generation: seed = 2000 + int(beta_eff * 100)
  Isolated from Paper 4 (seed 42), Paper 5 §5 (55), §7 (77), §8 (99/100).
"""

import numpy as np
from scipy.fft import fftn, ifftn

# ── SU(2) primitives ─────────────────────────────────────────────────────────

def adj(U):
    """Conjugate transpose, broadcasting over last two axes.
    For U shape (..., 2, 2): returns U†."""
    return np.conjugate(np.swapaxes(U, -1, -2))


def su2_norm(W):
    """
    SU(2)-norm |W| = sqrt(Tr[W†W] / 2)  (equation 2.13).
    For W = w₀ I₂ + i w⃗·σ⃗: |W| = sqrt(w₀² + |w|²).
    Input shape: (..., 2, 2). Output shape: (...).
    Returns scalar 1.0 for exact SU(2) input (|a|²+|b|²=1).
    Guard: clamp to ≥ 1e-300 to avoid division by zero in degenerate staples.
    """
    # Tr[W†W] = |W[0,0]|² + |W[0,1]|² + |W[1,0]|² + |W[1,1]|²
    norm_sq = (np.abs(W[..., 0, 0])**2 + np.abs(W[..., 0, 1])**2 +
               np.abs(W[..., 1, 0])**2 + np.abs(W[..., 1, 1])**2) / 2.0
    return np.sqrt(np.maximum(norm_sq, 1e-300))


def random_su2(shape, seed=None):
    """
    Haar-random SU(2) matrices, uniform on S³ (equation 2.7).
    Maps to: U = [[a₀+ia₃, a₂+ia₁], [−a₂+ia₁, a₀−ia₃]]
    where (a₀, a₁, a₂, a₃) is a unit 4-vector from N(0,1)⁴.
    Wraps su2_corrected_T3.su2_random with seeding support.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
        a = rng.standard_normal((*shape, 4))
    else:
        a = np.random.standard_normal((*shape, 4))
    a /= np.linalg.norm(a, axis=-1, keepdims=True)
    U = np.empty((*shape, 2, 2), dtype=complex)
    U[..., 0, 0] =  a[..., 0] + 1j * a[..., 3]
    U[..., 0, 1] =  a[..., 2] + 1j * a[..., 1]
    U[..., 1, 0] = -a[..., 2] + 1j * a[..., 1]
    U[..., 1, 1] =  a[..., 0] - 1j * a[..., 3]
    return U


def su2_project(U):
    """
    Project back onto SU(2) via quaternion renormalisation.
    Identical to su2_corrected_T3.su2_project — reproduced here
    for use without importing that module at runtime.
    """
    a0 = 0.5 * np.real(U[..., 0, 0] + U[..., 1, 1])
    a3 = 0.5 * np.imag(U[..., 0, 0] - U[..., 1, 1])
    a2 = 0.5 * np.real(U[..., 0, 1] - U[..., 1, 0])
    a1 = 0.5 * np.imag(U[..., 0, 1] + U[..., 1, 0])
    nrm = np.sqrt(a0**2 + a1**2 + a2**2 + a3**2)
    nrm = np.maximum(nrm, 1e-300)
    a0, a1, a2, a3 = a0/nrm, a1/nrm, a2/nrm, a3/nrm
    V = np.empty_like(U)
    V[..., 0, 0] =  a0 + 1j * a3
    V[..., 0, 1] =  a2 + 1j * a1
    V[..., 1, 0] = -a2 + 1j * a1
    V[..., 1, 1] =  a0 - 1j * a3
    return V


def matmul4(A, B):
    """Batched 2×2 matrix multiply over arbitrary leading dimensions.
    A, B shape (..., 2, 2) → (..., 2, 2)."""
    return np.einsum('...ij,...jk->...ik', A, B)


def su2_exp_vec(omega, alpha=1.0):
    """
    Vectorised SU(2) exponential for an array of algebra vectors.
    (equation 2.18: h(x) = exp(iα ω^a(x) T^a))

    H(x) = (iα/2) ω^a(x) σ^a = (iα/2)(ω¹σ¹ + ω²σ² + ω³σ³)
    exp(H) = cos(|h|/2) I₂ + sin(|h|/2)/|h| · i(h⃗·σ⃗)
    where h⃗ = α ω⃗, |h| = α|ω⃗|.

    Matrix form with i(h⃗·σ⃗) = [[ih³, ih¹+h², ih¹-h², -ih³]]:
      exp_H[0,0] = cos(θ) + i·sinc(|h|/2)·h³
      exp_H[0,1] = sinc(|h|/2)·(ih¹ + h²)
      exp_H[1,0] = sinc(|h|/2)·(ih¹ - h²)
      exp_H[1,1] = cos(θ) - i·sinc(|h|/2)·h³
    where θ = |h|/2, sinc(|h|/2) = sin(|h|/2)/|h| (→ 1/2 as |h|→0).
    Normalised to SU(2) by su2_project to remove floating-point drift.

    Input:  omega shape (..., 3), alpha scalar.
    Output: exp(H) shape (..., 2, 2).
    """
    h = alpha * omega                                    # (..., 3)
    h1, h2, h3 = h[..., 0], h[..., 1], h[..., 2]
    h_norm = np.sqrt(h1**2 + h2**2 + h3**2)             # (...)
    theta = h_norm / 2.0                                  # |h|/2

    cos_t = np.cos(theta)
    # sin(|h|/2)/|h|: Taylor for |h| < 1e-10 to avoid 0/0
    sin_over_norm = np.where(
        h_norm > 1e-10,
        np.sin(theta) / np.maximum(h_norm, 1e-300),
        0.5 - h_norm**2 / 48.0   # Taylor: sin(x/2)/x ≈ 1/2 - x²/48 + ...
    )

    exp_H = np.empty((*h.shape[:-1], 2, 2), dtype=complex)
    exp_H[..., 0, 0] = cos_t + 1j * sin_over_norm * h3
    exp_H[..., 0, 1] = sin_over_norm * (1j * h1 + h2)
    exp_H[..., 1, 0] = sin_over_norm * (1j * h1 - h2)
    exp_H[..., 1, 1] = cos_t - 1j * sin_over_norm * h3

    # Project to exact SU(2) to remove accumulated floating-point error
    return su2_project(exp_H)


# ── Gauge functional and convergence ─────────────────────────────────────────

def compute_FL(U):
    """
    Gauge functional F_L (equation 2.1):
      F_L[U^g] = Re Σ_{x,μ} Tr[U^g_μ(x)]
    Input: U shape (N, N, N, N, 4, 2, 2).
    Output: scalar. Maximum value = 2·d·V = 8V (d=4, FLAG-2).
    """
    # Tr[U_μ(x)] = U[..., 0, 0] + U[..., 1, 1]
    return float(np.sum(np.real(U[..., 0, 0] + U[..., 1, 1])))


def compute_gauge_field(U):
    """
    Gauge field A^a_μ(x) from link variables (equation 2.4):
      A^a_μ(x) = (1/2i) Tr[T^a (U_μ(x) − U†_μ(x))],  T^a = σ^a/2
    For SU(2) with U = [[a, b], [−b*, a*]]:
      A^1_μ = Im(U[0,1]) = Im(b)
      A^2_μ = Re(U[0,1]) = Re(b)
      A^3_μ = Im(U[0,0]) = Im(a)
    Input:  U shape (N, N, N, N, 4, 2, 2).
    Output: A shape (N, N, N, N, 4, 3).
    """
    A = np.empty((*U.shape[:5], 3))
    A[..., 0] = np.imag(U[..., 0, 1])   # A^1 = Im(b)
    A[..., 1] = np.real(U[..., 0, 1])   # A^2 = Re(b)
    A[..., 2] = np.imag(U[..., 0, 0])   # A^3 = Im(a)
    return A


def compute_divergence(A, N):
    """
    Lattice divergence (∂_μ A_μ)^a(x) (equation 2.5):
      (∂_μ A_μ)^a(x) = Σ_{μ=0}^{3} [A^a_μ(x) − A^a_μ(x−μ̂)]
    Uses backward finite difference; np.roll(F, +1, axis=μ)[x] = F[x−μ̂].
    Input:  A shape (N, N, N, N, 4, 3).
    Output: div_A shape (N, N, N, N, 3).
    FLAG-2: sum over μ = 0,1,2,3 (temporal included).
    """
    div_A = np.zeros((N, N, N, N, 3))
    for mu in range(4):
        A_mu = A[..., mu, :]                               # (N,N,N,N,3)
        div_A += A_mu - np.roll(A_mu, +1, axis=mu)        # backward diff
    return div_A


def compute_Theta(U):
    """
    Global convergence criterion Θ (equation 2.6):
      Θ = (1 / (d · N_c · V)) Σ_{x,a} [(∂_μ A_μ)^a(x)]²
    with d=4 (FLAG-2), N_c=2, V=N^4.
    Input:  U shape (N, N, N, N, 4, 2, 2).
    Output: scalar ≥ 0. Pass condition: Θ < 1e-14.
    """
    N = U.shape[0]
    V = N**4
    A = compute_gauge_field(U)
    div_A = compute_divergence(A, N)
    return float(np.sum(div_A**2)) / (4 * 2 * V)   # d=4, N_c=2


# ── Gauge transformation ─────────────────────────────────────────────────────

def gauge_transform(U_raw, g):
    """
    Apply gauge transformation g(x) to all links (equation 2.14):
      U^g_μ(x) = g(x) · U_μ(x) · g†(x+μ̂)
    Input:
      U_raw shape (N, N, N, N, 4, 2, 2) — raw (ungauged) links
      g shape     (N, N, N, N, 2, 2)    — gauge transformation
    Output: U_gf shape (N, N, N, N, 4, 2, 2).
    Uses np.roll to shift g†(x+μ̂) to position x for each μ.
    """
    N = U_raw.shape[0]
    U_gf = np.empty_like(U_raw)
    g_dag = adj(g)                               # g†(x), shape (N,N,N,N,2,2)
    for mu in range(4):
        # g†(x+μ̂): shift g† backward by 1 in axis mu → value at x is g†(x+μ̂)
        g_dag_shifted = np.roll(g_dag, -1, axis=mu)   # (N,N,N,N,2,2)
        # U^g_μ(x) = g(x) @ U_μ(x) @ g†(x+μ̂)
        U_gf[..., mu, :, :] = matmul4(g, matmul4(U_raw[..., mu, :, :], g_dag_shifted))
    return U_gf


# ── Parity mask ───────────────────────────────────────────────────────────────

def _parity_mask(N):
    """
    Boolean array shape (N,N,N,N): True at sites where (t+x+y+z) is even.
    Used for checkerboard decomposition in SD gauge fixing.
    """
    n0, n1, n2, n3 = np.mgrid[0:N, 0:N, 0:N, 0:N]
    return ((n0 + n1 + n2 + n3) % 2 == 0)


# ── SD algorithm (F5) ─────────────────────────────────────────────────────────

def _compute_staple_landau(U_gf):
    """
    Landau-gauge staple (equation 2.10) at all sites simultaneously:
      W(x) = Σ_{μ=0}^{3} [U^g_μ(x) + (U^g_μ(x−μ̂))†]
    Input:  U_gf shape (N, N, N, N, 4, 2, 2).
    Output: W    shape (N, N, N, N, 2, 2).
    Note: this is the Landau staple, NOT the Wilson plaquette staple.
    """
    W = np.zeros((*U_gf.shape[:4], 2, 2), dtype=complex)
    for mu in range(4):
        U_mu = U_gf[..., mu, :, :]                        # (N,N,N,N,2,2)
        U_mu_back = np.roll(U_mu, +1, axis=mu)            # U^g_μ(x−μ̂)
        W += U_mu + adj(U_mu_back)
    return W


def _sd_checkerboard_pass(U_gf, g_cum, parity_mask):
    """
    One checkerboard pass of the SD gauge-fixing sweep.
    Updates all sites of the given parity simultaneously (vectorised).

    Algorithm per site x (parity-p sites only):
      1. W(x)  = Σ_μ [U^g_μ(x) + (U^g_μ(x−μ̂))†]      (staple)
      2. h(x)  = W†(x) / |W(x)|                          (optimal increment)
      3. Forward:  U^g_μ(x) → h(x) @ U^g_μ(x)            (for all μ)
      4. Backward: U^g_μ(y) → U^g_μ(y) @ h†(y+μ̂)         (y = x−μ̂, all μ)
      5. g_cum(x) → h(x) @ g_cum(x)

    Parity argument (no link conflicts within one parity pass):
      Even-parity forward links land at odd sites (no self-conflict).
      Even-parity backward links are forward links AT odd sites (different from
      all forward links of even sites — no write conflict within the parity batch).
      Proof: if x (even) shares a backward link U_μ(x−μ̂) with another even site
      z, then x−μ̂ = z−ν̂ → x−z = μ̂−ν̂; but x−z has same parity as x (both even)
      while μ̂−ν̂ has even parity sum → this is consistent. However, U_μ(x−μ̂)
      and U_ν(z−ν̂) at the same odd site y are DIFFERENT link variables (μ ≠ ν in
      general), so even if y = x−μ̂ = z−ν̂, there is no write conflict.

    Input / Output: U_gf, g_cum updated in place (copies returned).
    parity_mask: bool array (N,N,N,N), True for sites being updated.
    """
    N = U_gf.shape[0]

    # Step 1-2: compute staple and optimal increment at ALL sites
    W = _compute_staple_landau(U_gf)                      # (N,N,N,N,2,2)
    norm_W = su2_norm(W)                                   # (N,N,N,N)
    h_all = adj(W) / norm_W[..., np.newaxis, np.newaxis]  # W†/|W|, all sites

    # Zero out h at off-parity sites (set to identity)
    mask4 = parity_mask[..., np.newaxis, np.newaxis]      # (N,N,N,N,1,1)
    I2 = np.eye(2, dtype=complex)
    h = np.where(mask4, h_all, I2)                        # (N,N,N,N,2,2)

    # Step 3: forward link update — U^g_μ(x) → h(x) @ U^g_μ(x)  for parity-p x
    for mu in range(4):
        U_gf[..., mu, :, :] = np.where(
            mask4,
            matmul4(h, U_gf[..., mu, :, :]),
            U_gf[..., mu, :, :]
        )

    # Step 4: backward link update — U^g_μ(y) → U^g_μ(y) @ h†(y+μ̂)  for odd y
    # h†(y+μ̂) at position y: roll h† by -1 in axis mu
    anti_mask4 = (~parity_mask)[..., np.newaxis, np.newaxis]
    h_dag = adj(h)
    for mu in range(4):
        h_dag_next = np.roll(h_dag, -1, axis=mu)          # h†(y+μ̂) at y
        U_gf[..., mu, :, :] = np.where(
            anti_mask4,
            matmul4(U_gf[..., mu, :, :], h_dag_next),
            U_gf[..., mu, :, :]
        )

    # Step 5: update cumulative gauge transformation
    g_cum[:] = np.where(mask4, matmul4(h, g_cum), g_cum)

    return U_gf, g_cum


# ── FA algorithm (F6) ─────────────────────────────────────────────────────────

def f_lattice(N):
    """
    Lattice Laplacian eigenvalues (equation 2.16):
      f(p) = Σ_{μ=0}^{3} 4 sin²(π p_μ / N),  p ∈ {0,...,N−1}^4.
    Shape: (N, N, N, N). The p=(0,0,0,0) mode is set to 1 to avoid
    division by zero; the numerator is zeroed separately (zero-mode projection).
    """
    p0, p1, p2, p3 = np.mgrid[0:N, 0:N, 0:N, 0:N]
    f = (4 * np.sin(np.pi * p0 / N)**2 + 4 * np.sin(np.pi * p1 / N)**2 +
         4 * np.sin(np.pi * p2 / N)**2 + 4 * np.sin(np.pi * p3 / N)**2)
    f[0, 0, 0, 0] = 1.0   # avoid division by zero; zero-mode zeroed in numerator
    return f


def get_alpha(beta_eff):
    """
    Step size α for FA gauge-fixing (equation 2.18).
    A value of 0.4 is standard and empirically robust for SU(2) Landau gauge.
    Can be tuned per β_eff if convergence is slow (typically not necessary for
    the β_eff ∈ {5,...,20} range of Paper 6).
    """
    return 0.4


def _fa_step(U_gf, g_cum, U_raw, alpha, N, f_arr):
    """
    One step of Fourier-accelerated gauge fixing (F6, equation 2.15–2.19).
    Returns updated (U_gf, g_cum).
    """
    # Step 1: gauge field and divergence
    A = compute_gauge_field(U_gf)                          # (N,N,N,N,4,3)
    div_A = compute_divergence(A, N)                       # (N,N,N,N,3)

    # Step 2: FFT each color component
    div_A_tilde = np.empty((N, N, N, N, 3), dtype=complex)
    for a in range(3):
        div_A_tilde[..., a] = fftn(div_A[..., a])

    # Zero-mode projection (equation 2.15): D̃^a(0,0,0,0) = 0
    div_A_tilde[0, 0, 0, 0, :] = 0.0

    # Step 3: Laplacian preconditioning (equation 2.16)
    omega_tilde = div_A_tilde / f_arr[..., np.newaxis]     # (N,N,N,N,3)

    # Step 4: IFFT to real space
    omega = np.empty((N, N, N, N, 3))
    for a in range(3):
        omega[..., a] = np.real(ifftn(omega_tilde[..., a]))

    # Step 5: form correction h(x) = exp(iα ω^a T^a)
    h = su2_exp_vec(omega, alpha=alpha)                    # (N,N,N,N,2,2)

    # Step 6: update cumulative transformation and links
    g_cum = matmul4(h, g_cum)
    U_gf = gauge_transform(U_raw, g_cum)

    return U_gf, g_cum


# ── Main gauge-fixing function (D19) ─────────────────────────────────────────

def gauge_fix_and_check_D19(U_raw, beta_eff, N,
                             algorithm='auto',
                             eps_gauge=1e-14,
                             k_max_SD=10000,
                             k_max_FA=2000,
                             random_seed=None):
    """
    Landau gauge-fixing with D19 convergence check.
    Implements F5 (SD) or F6 (FA) depending on N and algorithm flag.

    Derivation:
      F5 (SD): Local Cabibbo-Marinari-style update. Each checkerboard pass
        maximises F_L locally at all sites of one parity. Monotone convergence
        guaranteed (Lemma 4(i)). Rate: O(V) sweeps to Θ < ε_gauge.
      F6 (FA): FFT Laplacian preconditioning of the gradient. Each step
        is a global update in Fourier space, reducing low-momentum modes
        efficiently. Rate: O(V^{1/2}) steps.

    Parameters
    ----------
    U_raw        : ndarray (N,N,N,N,4,2,2), raw Monte Carlo configuration
    beta_eff     : float, β_eff for the ensemble
    N            : int, lattice side length
    algorithm    : 'SD', 'FA', or 'auto' (FA if N≥12, SD otherwise)
    eps_gauge    : float, convergence threshold Θ < eps_gauge (default 1e-14)
    k_max_SD     : int, max SD sweeps before declaring non-convergence
    k_max_FA     : int, max FA steps before declaring non-convergence
    random_seed  : int or None, seed for initial random gauge transformation

    Returns
    -------
    U_gf         : ndarray (N,N,N,N,4,2,2), gauge-fixed links
    g_fix        : ndarray (N,N,N,N,2,2),   cumulative gauge transformation
    FL_final     : float, F_L of gauge-fixed configuration
    Theta_final  : float, final Θ value
    k_converged  : int, number of sweeps/steps to convergence (k_max if not)
    converged    : bool, True if Θ < eps_gauge achieved

    D19 PASS condition: Theta_final < eps_gauge AND converged = True.
    Non-convergence at rate > 5% of configurations is anomalous (§2 CR).
    """
    if algorithm == 'auto':
        algorithm = 'FA' if N >= 12 else 'SD'

    # Initialise random gauge transformation g ~ Haar(SU(2))
    g_cum = random_su2((N, N, N, N), seed=random_seed)
    U_gf = gauge_transform(U_raw, g_cum)

    Theta = compute_Theta(U_gf)   # initial check (may already be small)

    if algorithm == 'SD':
        parity_even = _parity_mask(N)
        parity_odd  = ~parity_even

        for k in range(k_max_SD):
            # Even parity pass
            U_gf, g_cum = _sd_checkerboard_pass(U_gf, g_cum, parity_even)
            # Odd parity pass
            U_gf, g_cum = _sd_checkerboard_pass(U_gf, g_cum, parity_odd)

            # Periodically re-project links to SU(2) (every 50 sweeps)
            if (k + 1) % 50 == 0:
                for mu in range(4):
                    U_gf[..., mu, :, :] = su2_project(U_gf[..., mu, :, :])
                # Recompute U_gf from g_cum to avoid accumulated drift
                U_gf = gauge_transform(U_raw, g_cum)

            Theta = compute_Theta(U_gf)
            if Theta < eps_gauge:
                FL_final = compute_FL(U_gf)
                return U_gf, g_cum, FL_final, Theta, k + 1, True

        # Did not converge
        FL_final = compute_FL(U_gf)
        return U_gf, g_cum, FL_final, Theta, k_max_SD, False

    elif algorithm == 'FA':
        f_arr = f_lattice(N)
        alpha = get_alpha(beta_eff)

        for k in range(k_max_FA):
            Theta = compute_Theta(U_gf)
            if Theta < eps_gauge:
                FL_final = compute_FL(U_gf)
                return U_gf, g_cum, FL_final, Theta, k, True
            U_gf, g_cum = _fa_step(U_gf, g_cum, U_raw, alpha, N, f_arr)

        Theta = compute_Theta(U_gf)
        FL_final = compute_FL(U_gf)
        return U_gf, g_cum, FL_final, Theta, k_max_FA, False

    else:
        raise ValueError(f"Unknown algorithm '{algorithm}'. Use 'SD', 'FA', or 'auto'.")


# ── Check D20: per-site Landau condition ─────────────────────────────────────

def check_D20(U_gf, eps_gauge=1e-14):
    """
    D20 — per-site Landau condition check (Lemma 4, Definition 2.3).

    Derivation:
      From the global criterion Θ < ε_gauge and Θ = (1/(8V))Σ_{x,a} div²:
        Σ_{x,a} (∂_μ A_μ)^a(x)² < 8V · ε_gauge
      By the max-bound: max_{x,a} |(∂_μ A_μ)^a(x)| ≤ sqrt(8V · ε_gauge)
      A site that exceeds this bound in isolation signals localised failure
      (e.g., near a topological defect or Gribov horizon).

    Parameters
    ----------
    U_gf     : ndarray (N,N,N,N,4,2,2), gauge-fixed links
    eps_gauge: float, global convergence threshold

    Returns
    -------
    max_div  : float, max_{x,a} |(∂_μ A_μ)^a(x)|
    eps_site : float, derived per-site bound sqrt(8V · eps_gauge)
    passed   : bool, True if max_div < eps_site
    """
    N = U_gf.shape[0]
    V = N**4
    A = compute_gauge_field(U_gf)
    div_A = compute_divergence(A, N)
    max_div = float(np.max(np.abs(div_A)))
    eps_site = float(np.sqrt(8 * V * eps_gauge))
    return max_div, eps_site, bool(max_div < eps_site)


# ── Check D21: gauge-fixed link trace consistency ────────────────────────────

def check_D21(U_raw, U_gf, beta_eff, FL_other=None, V_other=None):
    """
    D21 — gauge-fixed link trace consistency (Lemma 4 upper bound).

    Three sub-checks:
      D21(a): F_L^{gf} > F_L^{initial}  (Lemma 4(i) sanity check)
      D21(b): R = F_L^{gf}/(8V) ≥ R_min(β_eff)  (physical plausibility)
              R_min values are empirical order-of-magnitude guards; see CR note.
              These are LOWER bounds only — values approaching 1 at high β are
              expected and correct.
      D21(c): Extensive scaling F_L(N=6)/F_L(N=12) ≈ V(6)/V(12) within ±5%
              (requires FL_other from a different-volume ensemble at same β_eff)

    R_min derivation:
      ⟨Re Tr[U_μ]⟩_Landau ≈ 2(1 − c/β_eff) with c = O(1); R ≈ 1 − c/β_eff.
      Values in R_MIN_TABLE use c ≈ 1.2–1.5 as order-of-magnitude estimates.
      EMPIRICAL ESTIMATES PENDING SIMULATION — must be updated from first run.

    Parameters
    ----------
    U_raw    : ndarray (N,N,N,N,4,2,2), raw (pre-fix) links
    U_gf     : ndarray (N,N,N,N,4,2,2), gauge-fixed links
    beta_eff : float
    FL_other : float or None — F_L from another-volume ensemble (for D21c)
    V_other  : int or None   — volume of that other ensemble (for D21c)

    Returns
    -------
    results : dict with keys 'D21a', 'D21b', 'D21c'; each a dict:
              {'value': ..., 'expected': ..., 'passed': bool, 'detail': str}
    """
    N = U_raw.shape[0]
    V = N**4

    FL_initial = compute_FL(U_raw)
    FL_gf      = compute_FL(U_gf)
    R          = FL_gf / (8.0 * V)

    # Empirical R_min guards (lower bounds; see docstring)
    R_MIN_TABLE = {
        2.0: 0.55, 2.5: 0.60, 3.0: 0.65, 3.5: 0.68,
        5.0: 0.65, 6.0: 0.70, 9.0: 0.80, 12.0: 0.88, 20.0: 0.93
    }
    R_min = R_MIN_TABLE.get(beta_eff, 0.60)

    # D21(a)
    d21a_val    = FL_gf - FL_initial
    d21a_passed = bool(d21a_val > 0)
    d21a = {'value': d21a_val, 'expected': '>0', 'passed': d21a_passed,
            'detail': f'F_L^gf − F_L^init = {d21a_val:.4f}'}

    # D21(b)
    d21b_passed = bool(R >= R_min)
    d21b = {'value': R, 'expected': f'>= {R_min:.2f}',
            'passed': d21b_passed,
            'detail': f'R = F_L^gf/(8V) = {R:.6f} (R_min={R_min:.2f})'}

    # D21(c)
    if FL_other is not None and V_other is not None:
        V_self  = float(V)
        ratio_measured = FL_gf / FL_other
        ratio_expected = V_self / float(V_other)
        rel_err = abs(ratio_measured - ratio_expected) / ratio_expected
        d21c_passed = bool(rel_err < 0.05)
        d21c = {'value': ratio_measured,
                'expected': f'{ratio_expected:.5f} ±5%',
                'passed': d21c_passed,
                'detail': f'F_L(N={N})/F_L(other) = {ratio_measured:.5f}, '
                          f'V ratio = {ratio_expected:.5f}, rel_err = {rel_err:.3f}'}
    else:
        d21c = {'value': None, 'expected': 'DEFERRED',
                'passed': None,  # None = not evaluated
                'detail': 'D21(c) requires multi-volume data; deferred to extended ensemble run.'}

    return {'D21a': d21a, 'D21b': d21b, 'D21c': d21c}


# ── F7: Gribov copy strategy ─────────────────────────────────────────────────

def gribov_copy_run(U_raw, beta_eff, N, n_starts=5,
                    eps_gauge=1e-14, k_max_SD=10000,
                    seed_offset=0, verbose=False):
    """
    F7: Gribov copy strategy (§3 of Paper 6).
    Run ≥5 independent gauge-fixing starts from random initial g.
    Select the copy achieving the maximum F_L value (best Gribov copy).
    Record F_L, Θ, k_converged for each start.

    Parameters
    ----------
    U_raw       : ndarray (N,N,N,N,4,2,2)
    beta_eff    : float
    N           : int
    n_starts    : int, number of independent Gribov starts (≥5 required)
    eps_gauge   : float
    k_max_SD    : int
    seed_offset : int, added to per-start seed for isolation
    verbose     : bool, print per-start summary

    Returns
    -------
    best        : dict with keys U_gf, g_fix, FL, Theta, k, converged,
                  start_idx (index of the best start)
    all_starts  : list of dicts (one per start), each with same keys
    """
    all_starts = []
    best = None

    for s in range(n_starts):
        seed = seed_offset + s * 7 + 13   # distinct seeds per start
        U_gf, g_fix, FL, Theta, k, conv = gauge_fix_and_check_D19(
            U_raw, beta_eff, N,
            algorithm='auto',
            eps_gauge=eps_gauge,
            k_max_SD=k_max_SD,
            random_seed=seed
        )
        record = {
            'U_gf': U_gf, 'g_fix': g_fix,
            'FL': FL, 'Theta': Theta, 'k': k,
            'converged': conv, 'start_idx': s
        }
        all_starts.append(record)

        if best is None or FL > best['FL']:
            best = record

        if verbose:
            status = 'CONV' if conv else 'FAIL'
            print(f"    start {s}: FL={FL:.4f}  Θ={Theta:.2e}  k={k:5d}  [{status}]")

    return best, all_starts


# ── Ensemble generation ───────────────────────────────────────────────────────

# Metropolis epsilon tuned for ~45% acceptance at each β_eff.
# Based on su2_corrected_T3.py EPS_DICT = {2.5:0.483, 3.0:0.421, 4.5:0.321, 6.0:0.270}
# and linear extrapolation / empirical tuning for β_eff ∈ {5,9,12,20}.
_EPS_DICT_P6 = {
    2.0: 0.53, 2.5: 0.48, 3.0: 0.42, 3.5: 0.38,
    5.0: 0.30, 6.0: 0.27, 9.0: 0.22, 12.0: 0.19, 20.0: 0.15
}

def generate_ensemble(beta_eff, N, N_cfg, N_therm=500, N_decorr=10, seed=None):
    """
    Generate a Monte Carlo ensemble of SU(2) pure-gauge configurations.
    Uses the Metropolis algorithm from su2_corrected_T3.py (DS_FACTOR=2).

    NOTE: No temporal gauge pre-fixing is applied (FLAG-2 compliance).
          All 4 link directions are live in the Monte Carlo.

    Parameters
    ----------
    beta_eff : float, lattice coupling (β = 4/g²)
    N        : int, lattice side length (volume V = N^4)
    N_cfg    : int, number of configurations to generate
    N_therm  : int, thermalisation sweeps (default 500)
    N_decorr : int, decorrelation sweeps between configurations (default 10)
    seed     : int or None

    Returns
    -------
    configs : ndarray (N_cfg, N, N, N, N, 4, 2, 2), one config per row
    plaq    : ndarray (N_cfg,), mean plaquette per config
    """
    # Import Metropolis infrastructure from su2_corrected_T3
    # (metropolis_sweep and plaquette are production-validated there)
    try:
        from su2_corrected_T3 import metropolis_sweep, plaquette
    except ImportError:
        raise ImportError(
            "su2_corrected_T3.py must be in the same directory as paper6_gauge_fix.py.\n"
            "Functions used: metropolis_sweep, plaquette."
        )

    eps = _EPS_DICT_P6.get(beta_eff, 0.25)
    if seed is not None:
        np.random.seed(seed)

    # Cold start: all links = I₂
    L = np.zeros((N, N, N, N, 4, 2, 2), dtype=complex)
    L[..., 0, 0] = 1.0
    L[..., 1, 1] = 1.0

    # Thermalisation
    for _ in range(N_therm):
        L, _ = metropolis_sweep(L, beta_eff, eps)

    # Production
    configs = np.empty((N_cfg, N, N, N, N, 4, 2, 2), dtype=complex)
    plaq    = np.empty(N_cfg)

    for cfg in range(N_cfg):
        for _ in range(N_decorr):
            L, _ = metropolis_sweep(L, beta_eff, eps)
        configs[cfg] = L.copy()
        plaq[cfg]    = plaquette(L)

    return configs, plaq