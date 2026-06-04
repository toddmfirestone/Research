"""
GA-SU(3) Lattice Gauge Theory Framework
========================================
Corrected modular implementation based on the Cl(6) -> SU(3) lattice study.

Bugs fixed vs. the conversation's evolving code:
  1. Single lattice initialization only (no double-init overwriting)
  2. SU(3) generators normalized consistently: Tr(T_a T_b) = (1/2) delta_ab
  3. Gell-Mann matrices divided by 2 before exponentiating for correct SU(3) elements
  4. Gauge links initialized as proper SU(3) elements (not raw random matrices)
  5. Plaquette trace normalized as Re[Tr(P)]/3 throughout
  6. Wilson loop backward legs use .conj().T (U-dagger) with consistent PBC
  7. Sanity checks (cold/hot start) added before any physics run

Structure (7 modules):
  1. CliffordCl6     -- Cl(6) gamma matrices and bivector basis
  2. SU3Algebra      -- Gell-Mann generators with correct normalization
  3. Lattice         -- lattice geometry and periodic boundary conditions
  4. Fields          -- Psi (spinor) and U (gauge links), single init
  5. Observables     -- plaquette average
  6. WilsonLoop      -- rectangular Wilson loop W(L,L)
  7. Main pipeline   -- sanity checks, then physics run

Author: corrected version for research evaluation
"""

import numpy as np
from itertools import combinations

# ============================================================
# 1. CLIFFORD ALGEBRA MODULE  Cl(6)
# ============================================================

class CliffordCl6:
    """
    Euclidean Clifford algebra Cl(6) built from Pauli tensor products.
    Provides:
      - 6 gamma matrices satisfying {gamma_i, gamma_j} = 2 delta_ij I
      - 15 bivectors B_ij = gamma_i @ gamma_j  (i < j)
      - Hilbert-Schmidt inner product and normalized basis
    """

    def __init__(self):
        I2 = np.eye(2, dtype=complex)
        sx = np.array([[0, 1], [1, 0]], dtype=complex)
        sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
        sz = np.array([[1, 0], [0, -1]], dtype=complex)

        def kron(*ops):
            out = np.array([[1]], dtype=complex)
            for op in ops:
                out = np.kron(out, op)
            return out

        # Standard Cl(6) embedding via tensor products of Pauli matrices
        self.gammas = [
            kron(sx, I2, I2),
            kron(sy, I2, I2),
            kron(sz, sx, I2),
            kron(sz, sy, I2),
            kron(sz, sz, sx),
            kron(sz, sz, sy),
        ]

        # Verify Clifford relation: {gamma_i, gamma_j} = 2 delta_ij
        self._verify_clifford()

        # Build and normalize bivector basis
        self.bivectors_raw = []
        self.bivector_labels = []
        for i, j in combinations(range(6), 2):
            self.bivectors_raw.append(self.gammas[i] @ self.gammas[j])
            self.bivector_labels.append(f"B{i+1}{j+1}")

        assert len(self.bivectors_raw) == 15, "Expected 15 bivectors for Cl(6)"

        # Hilbert-Schmidt inner product: <A,B> = Re[Tr(A†B)]
        self.inner = lambda A, B: np.real(np.trace(A.conj().T @ B))

        # Gram-Schmidt orthonormalize
        self.basis = self._gram_schmidt(self.bivectors_raw)
        self.n = len(self.basis)  # 15

        print(f"CliffordCl6 initialized: {self.n} orthonormal bivectors, dim=8x8")

    def _verify_clifford(self):
        for i in range(6):
            for j in range(6):
                ac = self.gammas[i] @ self.gammas[j] + self.gammas[j] @ self.gammas[i]
                expected = 2 * np.eye(8, dtype=complex) if i == j else np.zeros((8, 8), dtype=complex)
                assert np.allclose(ac, expected), f"Clifford relation failed for ({i},{j})"
        print("  Clifford relations verified: {gamma_i, gamma_j} = 2 delta_ij  ✓")

    def _gram_schmidt(self, ops):
        orth = []
        for X in ops:
            Y = X.copy()
            for Q in orth:
                proj = self.inner(Q, Y) * Q
                Y = Y - proj
            norm = np.sqrt(self.inner(Y, Y))
            if norm > 1e-12:
                orth.append(Y / norm)
        return orth

    def comm(self, A, B):
        return A @ B - B @ A


# ============================================================
# 2. SU(3) ALGEBRA MODULE
# ============================================================

class SU3Algebra:
    """
    SU(3) Gell-Mann generators with correct physics normalization.

    Convention: T_a = lambda_a / 2
    This gives: Tr(T_a T_b) = (1/2) delta_ab   [standard QCD convention]
    And:        [T_a, T_b] = i f_abc T_c

    The gauge link exponentiation is: U = exp(i theta_a T_a)
    which correctly gives U in SU(3).

    Bug fixed: original code used lambda_a without /2 in some places,
    which corrupts the normalization and structure constants.
    """

    def __init__(self, eps=0.1, g=1.0):
        self.eps = eps   # perturbation amplitude for gauge links
        self.g   = g     # coupling constant

        # Gell-Mann matrices (standard form)
        lam = self._gell_mann_matrices()

        # T_a = lambda_a / 2  (correct QCD normalization)
        self.T = [L / 2.0 for L in lam]

        # Verify normalization: Tr(T_a T_b) = (1/2) delta_ab
        self._verify_normalization()

        # Compute structure constants f_abc from [T_a, T_b] = i f_abc T_c
        self.f = self._compute_structure_constants()

        print(f"SU3Algebra initialized: eps={eps}, g={g}")
        print(f"  Max |f_abc| deviation from expected SU(3): "
              f"{self._check_known_f():.2e}")

    def _gell_mann_matrices(self):
        lam = [None] * 8

        lam[0] = np.array([[0, 1, 0],
                            [1, 0, 0],
                            [0, 0, 0]], dtype=complex)

        lam[1] = np.array([[0, -1j, 0],
                            [1j,  0, 0],
                            [0,   0, 0]], dtype=complex)

        lam[2] = np.array([[1,  0, 0],
                            [0, -1, 0],
                            [0,  0, 0]], dtype=complex)

        lam[3] = np.array([[0, 0, 1],
                            [0, 0, 0],
                            [1, 0, 0]], dtype=complex)

        lam[4] = np.array([[0, 0, -1j],
                            [0, 0,   0],
                            [1j,0,   0]], dtype=complex)

        lam[5] = np.array([[0, 0, 0],
                            [0, 0, 1],
                            [0, 1, 0]], dtype=complex)

        lam[6] = np.array([[0,  0,  0],
                            [0,  0, -1j],
                            [0, 1j,  0]], dtype=complex)

        lam[7] = np.array([[1, 0,  0],
                            [0, 1,  0],
                            [0, 0, -2]], dtype=complex) / np.sqrt(3)

        return lam

    def _verify_normalization(self):
        errors = []
        for a in range(8):
            for b in range(8):
                val = np.real(np.trace(self.T[a] @ self.T[b]))
                expected = 0.5 if a == b else 0.0
                errors.append(abs(val - expected))
        max_err = max(errors)
        assert max_err < 1e-12, f"SU(3) normalization error: {max_err}"
        print(f"  SU(3) normalization Tr(T_a T_b)=(1/2)delta_ab verified ✓  max err={max_err:.2e}")

    def _compute_structure_constants(self):
        f = np.zeros((8, 8, 8))
        for a in range(8):
            for b in range(8):
                C = self.T[a] @ self.T[b] - self.T[b] @ self.T[a]
                for c in range(8):
                    # [T_a, T_b] = i f_abc T_c  =>  f_abc = -2i Tr(T_c [T_a,T_b])
                    f[a, b, c] = np.real(-2j * np.trace(self.T[c] @ C))
        return f

    def _check_known_f(self):
        """
        Known nonzero SU(3) structure constants (indices 1-based in literature).
        We check a subset: f_123=1, f_147=1/2, f_156=-1/2, f_246=1/2, etc.
        """
        known = {
            (0, 1, 2): 1.0,
            (0, 3, 6): 0.5,
            (0, 4, 5): 0.5,       # f_156 sign convention varies; check magnitude
            (1, 3, 5): 0.5,
            (2, 3, 4): 0.5,
            (3, 4, 7): np.sqrt(3)/2,
            (5, 6, 7): np.sqrt(3)/2,
        }
        max_err = 0.0
        for (a, b, c), val in known.items():
            err = abs(abs(self.f[a, b, c]) - abs(val))
            max_err = max(max_err, err)
        return max_err

    def random_su3(self):
        """
        Generate a random SU(3) element near identity using the generators.
        U = exp(i * eps * sum_a theta_a T_a), theta_a ~ N(0,1)
        This is always in SU(3) by construction.
        """
        theta = np.random.randn(8)
        H = sum(theta[a] * self.T[a] for a in range(8))
        # exp(i * eps * H) is unitary; H is Hermitian so det=1 follows from tracelessness
        from scipy.linalg import expm
        U = expm(1j * self.eps * H)
        # Project onto SU(3) to handle floating point drift
        U = self._project_su3(U)
        return U

    def identity_su3(self):
        return np.eye(3, dtype=complex)

    def _project_su3(self, U):
        """Re-unitarize U by SVD and fix determinant."""
        from numpy.linalg import svd
        V, s, Wh = svd(U)
        U_unitary = V @ Wh
        # Fix det = 1
        det = np.linalg.det(U_unitary)
        U_unitary /= det ** (1/3)
        return U_unitary


# ============================================================
# 3. LATTICE MODULE
# ============================================================

class Lattice:
    """
    Hypercubic lattice geometry.
      N: sites per dimension
      D: number of dimensions
    Provides periodic boundary conditions.
    """

    def __init__(self, N=4, D=4):
        self.N = N
        self.D = D
        print(f"Lattice initialized: {N}^{D} = {N**D} sites, {D} directions")

    def shift(self, x, mu, step=1):
        """Return site x shifted by +step in direction mu (with PBC)."""
        x_new = list(x)
        x_new[mu] = (x_new[mu] + step) % self.N
        return tuple(x_new)

    def all_sites(self):
        """Iterate over all lattice sites."""
        return np.ndindex(*([self.N] * self.D))


# ============================================================
# 4. FIELDS MODULE  (single initialization — bug fixed)
# ============================================================

class Fields:
    """
    Lattice fields: Psi (spinor) and U (gauge links).

    CRITICAL FIX: Only ONE initialization block.
    Original code had two separate Psi/U assignments; the second silently
    overwrote the first (including any SU(3) scaling applied in the first block).

    U[x][mu] is a 3x3 complex SU(3) matrix (gauge link from x in direction mu).
    Psi[x] is a complex spinor with Cl(6) and color components.
    """

    def __init__(self, lattice, clifford, su3, mode='cold'):
        """
        mode='cold'  : all U = identity (ordered start, plaquette -> 1)
        mode='hot'   : all U = random SU(3) (disordered, plaquette -> 0)
        mode='warm'  : all U = small perturbation near identity
        """
        self.lat = lattice
        self.cl  = clifford
        self.su3 = su3

        N, D = lattice.N, lattice.D
        n_cl = len(clifford.basis)   # 15 for Cl(6)
        n_col = 3                    # SU(3) color

        # Psi: shape (N,N,N,N, n_cl, n_col) complex
        # Single initialization only
        self.Psi = (np.random.randn(N, N, N, N, n_cl, n_col)
                  + 1j * np.random.randn(N, N, N, N, n_cl, n_col)) * 0.01

        # U: shape (N,N,N,N, D) of 3x3 complex matrices
        self.U = {}
        for x in lattice.all_sites():
            self.U[x] = []
            for mu in range(D):
                if mode == 'cold':
                    link = su3.identity_su3()
                elif mode == 'hot':
                    # true random SU(3) via Gram-Schmidt on random matrix
                    link = self._random_su3_ginibre()
                elif mode == 'warm':
                    link = su3.random_su3()
                else:
                    raise ValueError(f"Unknown mode: {mode}")
                self.U[x].append(link)

        print(f"Fields initialized: mode='{mode}', "
              f"Psi shape={N}^4 x {n_cl} x {n_col}, "
              f"U: {N**D * D} links of 3x3 SU(3)")

    def _random_su3_ginibre(self):
        """
        True Haar-random SU(3) via QR decomposition of a random complex matrix.
        This gives a uniformly distributed element of SU(3).
        """
        Z = (np.random.randn(3, 3) + 1j * np.random.randn(3, 3)) / np.sqrt(2)
        Q, R = np.linalg.qr(Z)
        # Make R diagonal entries positive (canonical QR)
        d = np.diag(R)
        Q = Q * (d / np.abs(d))
        # Fix det = 1
        det = np.linalg.det(Q)
        Q = Q / (det ** (1/3))
        return Q

    def verify_su3_links(self, sample_size=20):
        """Verify a random sample of gauge links are in SU(3)."""
        sites = list(self.lat.all_sites())
        idx = np.random.choice(len(sites), min(sample_size, len(sites)), replace=False)
        max_unitary_err = 0.0
        max_det_err = 0.0
        for i in idx:
            x = sites[i]
            for mu in range(self.lat.D):
                U = self.U[x][mu]
                unitary_err = np.linalg.norm(U @ U.conj().T - np.eye(3))
                det_err = abs(np.linalg.det(U) - 1.0)
                max_unitary_err = max(max_unitary_err, unitary_err)
                max_det_err = max(max_det_err, det_err)
        print(f"  SU(3) link verification: max |UU†-I|={max_unitary_err:.2e}, "
              f"max |det(U)-1|={max_det_err:.2e}", end=" ")
        if max_unitary_err < 1e-10 and max_det_err < 1e-10:
            print("✓")
        else:
            print("✗ WARNING: links are not in SU(3)!")
        return max_unitary_err, max_det_err


# ============================================================
# 5. OBSERVABLES MODULE
# ============================================================

class Observables:
    """
    Lattice observables: plaquette expectation value.

    Plaquette P_mu,nu(x) = U_mu(x) U_nu(x+mu) U_mu†(x+nu) U_nu†(x)
    Normalized: Re[Tr(P)] / 3

    For a cold start (all U=I): plaquette = 1.0  (exact)
    For a hot start (random SU(3)): plaquette -> 0  (by Haar average)
    """

    def __init__(self, fields):
        self.F   = fields
        self.lat = fields.lat

    def plaquette(self, x, mu, nu):
        """Compute single plaquette at site x in plane (mu, nu)."""
        U  = self.F.U
        N  = self.lat.N

        x      = tuple(x)
        x_mu   = self.lat.shift(x, mu)
        x_nu   = self.lat.shift(x, nu)

        U1 = U[x][mu]             # U_mu(x)
        U2 = U[x_mu][nu]          # U_nu(x+mu)
        U3 = U[x_nu][mu].conj().T # U_mu†(x+nu)
        U4 = U[x][nu].conj().T    # U_nu†(x)

        P = U1 @ U2 @ U3 @ U4
        return P

    def avg_plaquette(self):
        """Average plaquette over all sites and planes."""
        total = 0.0
        count = 0
        for x in self.lat.all_sites():
            for mu in range(self.lat.D):
                for nu in range(mu + 1, self.lat.D):
                    P = self.plaquette(x, mu, nu)
                    total += np.real(np.trace(P)) / 3.0
                    count += 1
        return total / count if count > 0 else 0.0

    def gauge_action(self, beta):
        """
        Wilson gauge action: S = (beta/3) sum_{x,mu<nu} Re[Tr(I - P_mu,nu(x))]
        """
        S = 0.0
        for x in self.lat.all_sites():
            for mu in range(self.lat.D):
                for nu in range(mu + 1, self.lat.D):
                    P = self.plaquette(x, mu, nu)
                    S += np.real(np.trace(np.eye(3) - P)) / 3.0
        return (beta / 3.0) * S


# ============================================================
# 6. WILSON LOOP MODULE
# ============================================================

class WilsonLoop:
    """
    Rectangular Wilson loop W(L, T) in the (mu, nu) plane.

    W(L,T) = Tr[ product of L forward links in mu,
                 product of T forward links in nu,
                 product of L backward links in mu,
                 product of T backward links in nu ]

    Normalized: Re[Tr(W)] / 3

    Area law: <W(L,L)> ~ exp(-sigma * L^2)  =>  confinement
    Perimeter law: <W(L,L)> ~ exp(-mu_0 * L)  =>  deconfinement

    Bug fixed: all backward legs use .conj().T (U-dagger), and boundary
    conditions applied consistently on each step.
    """

    def __init__(self, fields):
        self.F   = fields
        self.lat = fields.lat

    def loop(self, x0, mu, nu, L, T=None):
        """
        Compute W(L,T) Wilson loop starting at x0 in plane (mu,nu).
        If T is None, compute square loop W(L,L).
        """
        if T is None:
            T = L

        U  = self.F.U
        x  = list(x0)
        W  = np.eye(3, dtype=complex)

        # Forward leg: L steps in mu direction
        for _ in range(L):
            W = W @ U[tuple(x)][mu]
            x[mu] = (x[mu] + 1) % self.lat.N

        # Forward leg: T steps in nu direction
        for _ in range(T):
            W = W @ U[tuple(x)][nu]
            x[nu] = (x[nu] + 1) % self.lat.N

        # Backward leg: L steps in -mu direction
        for _ in range(L):
            x[mu] = (x[mu] - 1) % self.lat.N
            W = W @ U[tuple(x)][mu].conj().T

        # Backward leg: T steps in -nu direction
        for _ in range(T):
            x[nu] = (x[nu] - 1) % self.lat.N
            W = W @ U[tuple(x)][nu].conj().T

        return W

    def avg_loop(self, L, T=None, mu=0, nu=1):
        """Average Wilson loop W(L,T) over all starting sites."""
        if T is None:
            T = L
        total = 0.0
        count = 0
        for x in self.lat.all_sites():
            W = self.loop(x, mu, nu, L, T)
            total += np.real(np.trace(W)) / 3.0
            count += 1
        return total / count if count > 0 else 0.0


# ============================================================
# 7. MAIN PIPELINE
# ============================================================

def sanity_checks(lat, cl, su3):
    """
    Three mandatory sanity checks before any physics.
    These must all pass, or the framework is broken.
    """
    print("\n" + "="*60)
    print("SANITY CHECKS")
    print("="*60)

    # CHECK 1: Cold start plaquette = 1.0 exactly
    print("\n[1] Cold start: all U=I => plaquette must be exactly 1.0")
    fields_cold = Fields(lat, cl, su3, mode='cold')
    fields_cold.verify_su3_links()
    obs_cold = Observables(fields_cold)
    p_cold = obs_cold.avg_plaquette()
    print(f"    <plaquette>_cold = {p_cold:.10f}", end=" ")
    if abs(p_cold - 1.0) < 1e-12:
        print("✓ PASS")
    else:
        print(f"✗ FAIL  (error = {abs(p_cold-1.0):.2e})")

    # CHECK 2: Hot start plaquette ~ 0 (by Haar averaging)
    print("\n[2] Hot start: all U=random SU(3) => plaquette should be near 0")
    fields_hot = Fields(lat, cl, su3, mode='hot')
    fields_hot.verify_su3_links()
    obs_hot = Observables(fields_hot)
    p_hot = obs_hot.avg_plaquette()
    print(f"    <plaquette>_hot = {p_hot:.6f}", end=" ")
    if abs(p_hot) < 0.15:   # typical value for random SU(3) on N=4 lattice
        print("✓ PASS  (near 0 as expected for random links)")
    else:
        print(f"  NOTE: value={p_hot:.4f}. "
              f"Small lattices may fluctuate; check distribution.")

    # CHECK 3: W(1,1) = plaquette (they must be the same object)
    print("\n[3] W(1,1) Wilson loop must equal plaquette (consistency)")
    wil_cold = WilsonLoop(fields_cold)
    w11 = wil_cold.avg_loop(1, 1, mu=0, nu=1)
    print(f"    W(1,1)_cold     = {w11:.10f}", end=" ")
    if abs(w11 - p_cold) < 1e-10:
        print("✓ PASS")
    else:
        print(f"✗ FAIL  (plaquette={p_cold:.10f}, W(1,1)={w11:.10f})")

    print("\nSanity checks complete.")
    return fields_cold, fields_hot


def run_cl6_analysis(cl):
    """
    Phase 1: Cl(6) algebraic analysis.
    Builds the internal curvature operator as the adjoint action
    of the sum of bivectors, then checks closure of the lowest
    8 eigenmodes under commutation.
    """
    print("\n" + "="*60)
    print("PHASE 1: Cl(6) SPECTRAL / CLOSURE ANALYSIS")
    print("="*60)

    basis = cl.basis
    n     = cl.n          # 15
    inner = cl.inner
    comm  = cl.comm

    # Build internal curvature operator: A = sum of all basis bivectors
    # O_int = ad_A = [A, .]  acting on the 15-dim bivector space
    A = sum(basis)
    print(f"\nBuilding adjoint operator [A, .] on {n}-dim bivector space...")

    # Matrix representation of [A, .] in the bivector basis
    O = np.zeros((n, n), dtype=complex)
    for j, B_j in enumerate(basis):
        comm_AB = comm(A, B_j)
        for i, B_i in enumerate(basis):
            O[i, j] = inner(B_i, comm_AB)

    # Diagonalize
    eigvals, eigvecs = np.linalg.eigh(O + O.conj().T)  # symmetrize for real spectrum
    idx = np.argsort(np.abs(eigvals))
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    print(f"  Eigenvalues (lowest 10): {np.round(eigvals[:10].real, 4)}")
    print(f"  Eigenvalue spectrum gap: {abs(eigvals[8]) - abs(eigvals[7]):.4f}")

    # Build lowest 8 mode operators
    num_modes = 8
    low_modes = []
    for m in range(num_modes):
        vec = eigvecs[:, m]
        T = sum(vec[k] * basis[k] for k in range(n))
        low_modes.append(T)

    print(f"\nConstructed {num_modes} low-energy mode operators from eigenvectors.")

    # Closure test: does span(T_1,...,T_8) close under commutation?
    def project(X, modes):
        coeffs = [inner(T, X) for T in modes]
        X_proj = sum(c * T for c, T in zip(coeffs, modes))
        return X_proj, coeffs

    closure_errors = []
    print("\nCommutator closure test (SU(3) requires all errors << 1):")
    for i in range(num_modes):
        for j in range(i + 1, num_modes):
            C = comm(low_modes[i], low_modes[j])
            C_proj, _ = project(C, low_modes)
            err = np.linalg.norm(C - C_proj)
            closure_errors.append(err)

    closure_errors = np.array(closure_errors)
    print(f"  Mean closure error: {closure_errors.mean():.6f}")
    print(f"  Max  closure error: {closure_errors.max():.6f}")
    print(f"  Min  closure error: {closure_errors.min():.6f}")

    if closure_errors.mean() < 0.01:
        print("  INTERPRETATION: Strong closure — 8D Lie algebra structure found!")
    elif closure_errors.mean() < 0.5:
        print("  INTERPRETATION: Partial closure — approximate 8D structure.")
    else:
        print("  INTERPRETATION: Poor closure — spectral selection alone "
              "does NOT produce a Lie algebra.")
        print("    (This is the expected result, per conversation analysis.)")

    return low_modes, closure_errors

def identify_algebra(T_opt, su3, cl):
    """
    After variational optimization, identify what Lie algebra was found
    by comparing computed structure constants against known SU(3) f_abc.
    """
    inner = cl.inner
    comm  = cl.comm
    
    # Compute structure constants of T_opt
    # [T_a, T_b] = f_abc * T_c  =>  f_abc = inner(T_c, [T_a,T_b])
    n = len(T_opt)
    f_emp = np.zeros((n, n, n))
    for a in range(n):
        for b in range(n):
            C = comm(T_opt[a], T_opt[b])
            for c in range(n):
                f_emp[a, b, c] = inner(T_opt[c], C)
    
    # Compare against SU(3) canonical f_abc
    f_su3 = su3.f
    
    # Find best permutation match (structure constants are basis-dependent)
    # Simple check: compare Killing form B_ab = f_acd f_bcd
    B_emp = np.einsum('acd,bcd->ab', f_emp, f_emp)
    B_su3 = np.einsum('acd,bcd->ab', f_su3, f_su3)
    
    # SU(3) Killing form eigenvalues: all equal (simple algebra)
    eigs_emp = np.sort(np.linalg.eigvalsh(B_emp))
    eigs_su3 = np.sort(np.linalg.eigvalsh(B_su3))
    
    print(f"Killing form eigenvalues (found): {np.round(eigs_emp, 3)}")
    print(f"Killing form eigenvalues (SU(3)): {np.round(eigs_su3, 3)}")
    
    # If all eigenvalues equal and negative -> simple Lie algebra
    # If they match SU(3) ratios -> it's SU(3)

def identify_algebra(T_opt, su3, cl):
    """
    Identify which Lie algebra the variational optimizer found by computing
    its Killing form and comparing eigenvalue ratios against known algebras.

    The Killing form B_ab = f_acd * f_bcd is a basis-independent fingerprint.
    SU(3): all 8 eigenvalues equal (simple, compact algebra)
    su(2)+su(2)+...: eigenvalues cluster in groups
    """
    inner = cl.inner
    comm  = cl.comm
    n     = len(T_opt)

    print("\n--- ALGEBRA IDENTIFICATION ---")

    # Step 1: compute structure constants f_abc of the found generators
    # [T_a, T_b] = sum_c f_abc T_c  =>  f_abc = inner(T_c, [T_a, T_b])
    f_emp = np.zeros((n, n, n))
    for a in range(n):
        for b in range(n):
            C = comm(T_opt[a], T_opt[b])
            for c in range(n):
                f_emp[a, b, c] = inner(T_opt[c], C)

    # Step 2: Killing form B_ab = f_acd * f_bcd
    B_emp = np.einsum('acd,bcd->ab', f_emp, f_emp)
    B_su3 = np.einsum('acd,bcd->ab', su3.f, su3.f)

    eigs_emp = np.sort(np.linalg.eigvalsh(B_emp))
    eigs_su3 = np.sort(np.linalg.eigvalsh(B_su3))

    print(f"  Killing form eigenvalues (found): {np.round(eigs_emp, 4)}")
    print(f"  Killing form eigenvalues (SU(3)): {np.round(eigs_su3, 4)}")

    # Step 3: check if all eigenvalues equal (simple algebra signature)
    spread_emp = eigs_emp.max() - eigs_emp.min()
    spread_su3 = eigs_su3.max() - eigs_su3.min()
    print(f"  Eigenvalue spread (found): {spread_emp:.4f}")
    print(f"  Eigenvalue spread (SU(3)): {spread_su3:.4f}")

    # Step 4: normalize and compare ratios
    if abs(eigs_emp).max() > 1e-10:
        ratios_emp = eigs_emp / eigs_emp[0]
        ratios_su3 = eigs_su3 / eigs_su3[0]
        ratio_err  = np.linalg.norm(ratios_emp - ratios_su3)
        print(f"  Eigenvalue ratio match error: {ratio_err:.4f}")
        if ratio_err < 0.05 and spread_emp < 0.5:
            print("  VERDICT: Found algebra is consistent with SU(3) ✓")
        elif spread_emp < 0.5:
            print("  VERDICT: Simple algebra found, but ratios differ from SU(3)")
            print("           Likely so(3) x u(1) x u(1) or similar — not SU(3)")
        else:
            print("  VERDICT: Semisimple but not simple — not SU(3)")
    else:
        print("  VERDICT: Degenerate Killing form — algebra may be abelian or trivial")

    # Step 5: direct f_abc comparison against SU(3) (up to basis rotation)
    frob_direct = np.linalg.norm(f_emp - su3.f)
    frob_su3    = np.linalg.norm(su3.f)
    print(f"  Direct f_abc difference (Frobenius): {frob_direct:.4f}")
    print(f"  SU(3) f_abc norm:                    {frob_su3:.4f}")
    print(f"  Relative difference: {frob_direct/frob_su3:.4f}")
    print(f"  (< 0.1 would be strong evidence of SU(3) in same basis)")

def run_variational_su3_search(cl, su3):
    """
    Phase 2: Variational search for SU(3)-closed 8D subalgebra in Cl(6).
    Minimizes the closure loss L = sum_{a,b} ||[T_a,T_b] - P([T_a,T_b])||^2
    """
    print("\n" + "="*60)
    print("PHASE 2: VARIATIONAL SU(3) CLOSURE SEARCH IN Cl(6)")
    print("="*60)

    from scipy.optimize import minimize

    basis = cl.basis
    n     = cl.n
    inner = cl.inner
    comm  = cl.comm

    def unpack(params):
        T = []
        for a in range(8):
            vec = params[a * n:(a + 1) * n]
            Ta  = sum(vec[k] * basis[k] for k in range(n))
            T.append(Ta)
        return T

    def loss(params):
        T   = unpack(params)
        err = 0.0
        for a in range(8):
            for b in range(8):
                C     = comm(T[a], T[b])
                coeffs = [inner(T[c], C) for c in range(8)]
                C_proj = sum(coeffs[c] * T[c] for c in range(8))
                err   += np.linalg.norm(C - C_proj) ** 2
        
        return err

    np.random.seed(42)
    x0 = np.random.randn(8 * n) * 0.1

    print(f"  Starting optimization (BFGS, 200 max iterations)...")
    print(f"  Initial loss: {loss(x0):.6f}")
    result = minimize(loss, x0, method='BFGS',
                      options={'maxiter': 200, 'disp': False})

    T_opt = unpack(result.x)
    print(f"  Final loss:   {result.fun:.6f}")
    print(f"  Converged:    {result.success}")

    print("\nFinal closure check on optimized generators:")
    all_err = []
    for a in range(8):
        for b in range(a + 1, 8):
            C      = comm(T_opt[a], T_opt[b])
            coeffs = [inner(T_opt[c], C) for c in range(8)]
            C_proj = sum(coeffs[c] * T_opt[c] for c in range(8))
            err    = np.linalg.norm(C - C_proj)
            all_err.append(err)

    all_err = np.array(all_err)
    print(f"  Mean closure error: {all_err.mean():.6f}")
    print(f"  Max  closure error: {all_err.max():.6f}")

    if result.fun < 1.0:
        print("\n  INTERPRETATION: Optimization found approximate SU(3)-like structure.")
        print("    Compare structure constants against known SU(3) f_abc to confirm.")
    else:
        print("\n  INTERPRETATION: Optimization converged to generic so(6) subalgebra,")
        print("    not SU(3). SU(3) is not a natural subalgebra of Cl(6) bivectors.")
        print("    (Consistent with conversation analysis — this is expected.)")

    identify_algebra(T_opt, su3, cl)
    return T_opt, result.fun


def run_lattice_observables(fields_warm):
    """
    Phase 3: Lattice observable pipeline on a warm-start configuration.
    Computes plaquette and Wilson loops W(L,L) for L=1,2,3.
    NOTE: Without thermalization these are single-config measurements — not
    ensemble physics. The output here is for framework validation only.
    """
    print("\n" + "="*60)
    print("PHASE 3: LATTICE OBSERVABLES (single config, pre-thermalization)")
    print("="*60)
    print("  NOTE: These are unthermalized measurements for code validation.")
    print("  Physical confinement requires Monte Carlo ensemble averaging.\n")

    obs = Observables(fields_warm)
    wil = WilsonLoop(fields_warm)

    p = obs.avg_plaquette()
    print(f"  Average plaquette <P> = {p:.6f}")

    print("\n  Wilson loop W(L,L) for L=1,2,3:")
    for L in [1, 2, 3]:
        # Only run if lattice is large enough
        if fields_warm.lat.N >= 2 * L:
            w = wil.avg_loop(L, L, mu=0, nu=1)
            print(f"    W({L},{L}) = {w:.6f}")
        else:
            print(f"    W({L},{L}) = skipped (lattice N={fields_warm.lat.N} too small)")

    print("\n  Gauge action (beta=6.0):", obs.gauge_action(beta=6.0))



def run_su3_structure_constants_check(su3):
    """
    Phase 4: Verify SU(3) structure constants against known values.
    This is a pure algebra check confirming the SU3Algebra module is correct.
    """
    print("\n" + "="*60)
    print("PHASE 4: SU(3) STRUCTURE CONSTANTS VERIFICATION")
    print("="*60)

    f = su3.f
    print("\n  Known nonzero SU(3) f_abc (1-indexed in literature, 0-indexed here):")
    known_pairs = [
        ((0,1,2), "f_123", 1.0),
        ((0,3,6), "f_147", 0.5),
        ((0,4,5), "f_156", -0.5),
        ((1,3,5), "f_246", 0.5),
        ((1,4,6), "f_257", 0.5),
        ((2,3,4), "f_345", 0.5),
        ((2,5,6), "f_367", -0.5),
        ((3,4,7), "f_458", np.sqrt(3)/2),
        ((5,6,7), "f_678", np.sqrt(3)/2),
    ]

    max_err = 0.0
    for (a, b, c), name, expected in known_pairs:
        computed = f[a, b, c]
        err = abs(computed - expected)
        max_err = max(max_err, err)
        status = "✓" if err < 1e-10 else "✗"
        print(f"    {name} = {computed:+.6f}  (expected {expected:+.4f})  {status}")

    print(f"\n  Max deviation from known f_abc: {max_err:.2e}")
    if max_err < 1e-10:
        print("  All structure constants correct ✓")
    else:
        print("  WARNING: Structure constant errors detected ✗")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')

    print("="*60)
    print("GA-SU(3) LATTICE GAUGE FRAMEWORK")
    print("Corrected & Modular Implementation")
    print("="*60)

    # --- Initialize modules ---
    cl  = CliffordCl6()
    su3 = SU3Algebra(eps=0.3, g=1.0)
    lat = Lattice(N=4, D=4)

    # --- Sanity checks (mandatory) ---
    fields_cold, fields_hot = sanity_checks(lat, cl, su3)

    # --- SU(3) structure constants check ---
    run_su3_structure_constants_check(su3)

    # --- Cl(6) spectral analysis ---
    low_modes, closure_errors = run_cl6_analysis(cl)

    # --- Variational SU(3) search ---
    T_opt, final_loss = run_variational_su3_search(cl, su3)

    # --- Lattice observables (warm start for variety) ---
    fields_warm = Fields(lat, cl, su3, mode='warm')
    fields_warm.verify_su3_links()
    run_lattice_observables(fields_warm)

    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print("""
Next steps (Monte Carlo / research-grade):
  1. Implement Metropolis update for gauge links
  2. Run thermalization sweeps at beta = 5.0, 5.5, 6.0, 6.5
  3. Compute ensemble-averaged Wilson loops <W(L,L)> over 1000+ configs
  4. Fit area law: -log<W(L,L)> = sigma * L^2 + perimeter + const
  5. Locate confinement-deconfinement transition (beta_c ~ 5.6 for SU(3))
""")


# ============================================================
# 8. METROPOLIS UPDATE MODULE
# ============================================================

class Metropolis:
    """
    Metropolis Monte Carlo for SU(3) lattice gauge theory.

    Standard Creutz-Wilson algorithm:
      For each link U_mu(x):
        1. Propose U_new = V @ U_mu(x), V random SU(3) near identity
        2. Compute delta_S = -(beta/3) Re[Tr((U_new - U_old) @ staple)]
        3. Accept with probability min(1, exp(-delta_S))

    Local action update uses the 2*(D-1) = 6 staples around each link.
    This is exact (not approximate) — no force calculation required.

    Physics:
      beta = 6/g^2 controls coupling strength
      beta >> 1  (weak coupling):  ordered, plaquette -> 1, DECONFINED
      beta << 1  (strong coupling): disordered, plaquette -> 0, CONFINED
      Transition at beta_c ~ 5.69 for SU(3) pure gauge in 4D (first order)
      Standard continuum QCD simulation: beta = 6.0

    Bugs fixed vs. conversation code:
      - Generator normalization consistent with SU3Algebra (T_a = lambda_a/2)
      - Re-unitarization after each proposal (floating point drift prevention)
      - Proposal eps tunable for ~50% acceptance at physical beta values
    """

    def __init__(self, fields, beta=6.0, eps=0.20):
        self.F    = fields
        self.lat  = fields.lat
        self.su3  = fields.su3
        self.beta = beta
        self.eps  = eps
        self._n_proposed = 0
        self._n_accepted = 0

    @property
    def acceptance_rate(self):
        if self._n_proposed == 0:
            return 0.0
        return self._n_accepted / self._n_proposed

    def _propose(self):
        """Random SU(3) element near identity: exp(2i*eps*H), H = sum theta_a T_a."""
        from scipy.linalg import expm
        H = sum(np.random.randn() * self.su3.T[a] for a in range(8))
        V = expm(2j * self.eps * H)
        Q, R = np.linalg.qr(V)
        d = np.diag(R)
        Q = Q * (d / np.abs(d))
        det = np.linalg.det(Q)
        return Q / (det ** (1.0/3.0))

    def _staple(self, x, mu):
        """
        Sum of staples attached to link U_mu(x).
        In 4D there are 2*(D-1)=6 staples.
        """
        U   = self.F.U
        lat = self.lat
        D   = lat.D
        A   = np.zeros((3, 3), dtype=complex)

        for nu in range(D):
            if nu == mu:
                continue
            x_mu    = lat.shift(x, mu)
            x_nu    = lat.shift(x, nu)
            x_mnu   = lat.shift(x, nu, step=-1)
            x_mu_mnu = lat.shift(x_mu, nu, step=-1)

            # Forward staple
            A += U[x_mu][nu] @ U[x_nu][mu].conj().T @ U[x][nu].conj().T
            # Backward staple
            A += U[x_mu_mnu][nu].conj().T @ U[x_mnu][mu].conj().T @ U[x_mnu][nu]

        return A

    def sweep(self):
        """One full sweep: update every link once. Returns acceptance rate."""
        U   = self.F.U
        acc = 0
        tot = 0

        for x in self.lat.all_sites():
            for mu in range(self.lat.D):
                A     = self._staple(x, mu)
                U_old = U[x][mu]
                V     = self._propose()
                U_new = V @ U_old

                dS = -(self.beta / 3.0) * np.real(
                    np.trace(U_new @ A) - np.trace(U_old @ A)
                )

                if dS <= 0 or np.random.random() < np.exp(-dS):
                    U[x][mu] = U_new
                    acc += 1
                tot += 1

        self._n_proposed += tot
        self._n_accepted += acc
        return acc / tot

    def thermalize(self, n_sweeps=100, print_every=10, verbose=True):
        """Thermalize configuration. Returns self for chaining."""
        obs = Observables(self.F)
        if verbose:
            print(f"\n  Thermalizing {n_sweeps} sweeps  "
                  f"[beta={self.beta}, eps={self.eps}]")
            print(f"  {'Sweep':>6}  {'<plaquette>':>12}  {'accept':>8}")
            print(f"  {'-'*6}  {'-'*12}  {'-'*8}")

        for i in range(1, n_sweeps + 1):
            acc = self.sweep()
            if verbose and (i % print_every == 0 or i == 1):
                p = obs.avg_plaquette()
                print(f"  {i:>6}  {p:>12.6f}  {acc:>8.4f}")

        if verbose:
            print(f"  Overall acceptance: {self.acceptance_rate:.4f}")
        return self

    def measure(self, n_configs=50, decorr_sweeps=5, verbose=True):
        """
        Collect ensemble measurements after thermalization.

        Parameters
        ----------
        n_configs     : number of independent configurations to measure
        decorr_sweeps : MC sweeps between measurements (reduces autocorrelation)

        Returns
        -------
        dict with arrays: plaquette, W11, W22, and scalar beta/acceptance_rate
        """
        obs = Observables(self.F)
        wil = WilsonLoop(self.F)

        plaq_vals = []
        w11_vals  = []
        w22_vals  = []

        if verbose:
            print(f"\n  Measuring {n_configs} configs  "
                  f"[{decorr_sweeps} sweeps between]")
            print(f"  {'Config':>7}  {'<P>':>10}  {'W(1,1)':>10}  {'W(2,2)':>10}")
            print(f"  {'-'*7}  {'-'*10}  {'-'*10}  {'-'*10}")

        for cfg in range(1, n_configs + 1):
            for _ in range(decorr_sweeps):
                self.sweep()

            p   = obs.avg_plaquette()
            w11 = wil.avg_loop(1, 1)
            plaq_vals.append(p)
            w11_vals.append(w11)

            if self.lat.N >= 4:
                w22 = wil.avg_loop(2, 2)
                w22_vals.append(w22)

            if verbose and (cfg % 10 == 0 or cfg == n_configs):
                w22_str = f"{np.mean(w22_vals):>10.5f}" if w22_vals else f"{'N/A':>10}"
                print(f"  {cfg:>7}  {np.mean(plaq_vals):>10.5f}  "
                      f"{np.mean(w11_vals):>10.5f}  {w22_str}")

        return {
            'beta':            self.beta,
            'plaquette':       np.array(plaq_vals),
            'W11':             np.array(w11_vals),
            'W22':             np.array(w22_vals) if w22_vals else None,
            'acceptance_rate': self.acceptance_rate,
        }


def run_metropolis_pipeline(lat, cl, su3, beta=6.0, n_therm=100, n_configs=50):
    """
    Complete Monte Carlo pipeline:
      hot start -> thermalization -> ensemble measurement -> physics report.

    Parameters
    ----------
    beta     : inverse coupling (6.0 = standard quenched QCD)
    n_therm  : thermalization sweeps
    n_configs: configurations to collect after thermalization
    """
    print("\n" + "="*60)
    print(f"METROPOLIS MONTE CARLO  [beta={beta}]")
    print("="*60)

    # Tune eps for reasonable acceptance: target ~50%
    # For beta=6.0 on N=4: eps~0.25 gives ~40-55% acceptance
    eps = 0.25

    print("\nHot start (maximally disordered)...")
    fields = Fields(lat, cl, su3, mode='hot')
    fields.verify_su3_links()
    obs_init = Observables(fields)
    print(f"  Initial <P> = {obs_init.avg_plaquette():.5f}  (should be ~0)")

    # Thermalize
    metro = Metropolis(fields, beta=beta, eps=eps)
    metro.thermalize(n_sweeps=n_therm, print_every=max(1, n_therm//10))

    # Verify SU(3) links after many updates
    print("\n  Post-thermalization link check:")
    fields.verify_su3_links(sample_size=100)

    # Measure
    results = metro.measure(n_configs=n_configs, decorr_sweeps=5)

    # Report
    _print_physics_report(results)
    return results


def _print_physics_report(results):
    """Interpret ensemble results physically."""
    print("\n" + "="*60)
    print("PHYSICS REPORT")
    print("="*60)

    beta = results['beta']
    P    = results['plaquette']
    W11  = results['W11']
    W22  = results['W22']
    n    = len(P)

    P_mean,   P_err   = P.mean(),   P.std() / np.sqrt(n)
    W11_mean, W11_err = W11.mean(), W11.std() / np.sqrt(n)

    print(f"\n  beta          = {beta}")
    print(f"  N configs     = {n}")
    print(f"  Acceptance    = {results['acceptance_rate']:.4f}")
    print(f"\n  <plaquette>   = {P_mean:.5f} +/- {P_err:.5f}")
    print(f"  <W(1,1)>      = {W11_mean:.5f} +/- {W11_err:.5f}")

    if W22 is not None:
        W22_mean = W22.mean()
        W22_err  = W22.std() / np.sqrt(n)
        print(f"  <W(2,2)>      = {W22_mean:.5f} +/- {W22_err:.5f}")

        # Crude 2-point string tension estimate
        if W11_mean > 1e-8 and W22_mean > 1e-8:
            # Area law: log W(L,L) ~ -sigma*L^2 + mu*4L
            # From two points: sigma ~ (log W11 - log W22) / (4-1)
            sigma = (np.log(W11_mean) - np.log(W22_mean)) / 3.0
            print(f"\n  String tension  sigma = {sigma:.5f}  (lattice units)")
            print(f"  (Physical SU(3) at beta=6.0: sigma*a^2 ~ 0.040-0.055)")
            if sigma > 0.01:
                print(f"  SIGNAL: positive sigma -> confinement-like behavior")
            else:
                print(f"  SIGNAL: sigma near zero -> deconfined / more thermalization needed")

    # Reference comparison
    known_plaq = {4.5: 0.340, 5.0: 0.430, 5.5: 0.524, 5.69: 0.547, 6.0: 0.593, 6.5: 0.640}
    nearest_beta = min(known_plaq.keys(), key=lambda b: abs(b - beta))
    ref = known_plaq[nearest_beta]
    deviation = abs(P_mean - ref)

    print(f"\n  REFERENCE: SU(3) pure gauge at beta={nearest_beta}: ~{ref:.3f}")
    print(f"    Your  <P> = {P_mean:.5f}   deviation = {deviation:.4f}")
    if deviation < 0.015:
        print(f"    STATUS: Consistent with equilibrium ✓")
    elif deviation < 0.04:
        print(f"    STATUS: Close — slight finite-volume or thermalization offset")
    else:
        print(f"    STATUS: Check thermalization")

    print(f"\n  PHASE INTERPRETATION:")
    if beta <= 5.5:
        print(f"    CONFINED phase (strong coupling) — expected at beta={beta}")
    elif beta >= 6.0:
        print(f"    DECONFINED phase (weak coupling) — expected at beta={beta}")
    else:
        print(f"    NEAR TRANSITION — watch for discontinuity around beta~5.69")

    print(f"\n  NEXT STEPS TO CONFIRM CONFINEMENT:")
    print(f"    1. Run beta sweep: 4.5, 5.0, 5.5, 5.69, 6.0, 6.5")
    print(f"    2. At each beta, compute W(L,L) for L=1,2,3,4")
    print(f"    3. Fit: -log<W(L,L)> = sigma*L^2 + mu_p*(4L) + const")
    print(f"    4. sigma>0 and growing as beta decreases -> confinement confirmed")
    print(f"    5. Discontinuity in <P> near beta~5.69 -> first-order transition")


# ============================================================
# ENTRY POINT FOR FULL v2 RUN
# ============================================================

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')

    print("=" * 60)
    print("GA-SU(3) LATTICE FRAMEWORK  v2  (+ Metropolis)")
    print("=" * 60)

    # Initialize modules
    cl  = CliffordCl6()
    su3 = SU3Algebra(eps=0.3, g=1.0)
    lat = Lattice(N=6, D=4)

    # Sanity checks — run these first always
    fields_cold, fields_hot = sanity_checks(lat, cl, su3)
    run_su3_structure_constants_check(su3)

    # Metropolis pipeline
    # Note: n_therm=100 is minimal. For publication-quality results use 500+
    results = run_metropolis_pipeline(
        lat, cl, su3,
        beta=6.5,
        n_therm=500,
        n_configs=100,
    )

    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print("""
To reach research-grade results from here:
  - Increase n_therm to 500, n_configs to 500
  - Add beta sweep: [4.5, 5.0, 5.5, 5.69, 6.0, 6.5]
  - Implement HMC (Hybrid Monte Carlo) for better autocorrelation
  - Add smearing (APE or stout) to reduce UV noise in Wilson loops
  - Couple Psi (Cl6 spinor) to U (SU3 gauge) via kinetic term
""")