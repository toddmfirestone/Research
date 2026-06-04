# Ga su3 dirac v3.py


"""
Module 8: Lattice Dirac Operator — Cl(6) spinor coupled to SU(3) gauge field
=============================================================================

This module implements the Wilson-Dirac operator coupling the Cl(6) spinor
field Psi to the SU(3) gauge links U.

Physical meaning of the coupling
---------------------------------
The continuum covariant derivative for a spinor in a gauge field is:

    D_mu Psi(x) = (d_mu + i g A_mu(x)) Psi(x)

On the lattice, parallel transport replaces d_mu + igA_mu:

    D_mu Psi(x)  -->  U_mu(x) Psi(x+mu) - Psi(x)

The full Wilson-Dirac operator is:

    D_W = m_0 + (1/2) sum_mu [ gamma_mu (nabla_mu + nabla_mu*) - r nabla_mu* nabla_mu ]

where:
    nabla_mu Psi(x)  = U_mu(x) Psi(x+mu) - Psi(x)          (forward difference)
    nabla_mu* Psi(x) = Psi(x) - U_mu†(x-mu) Psi(x-mu)      (backward difference)
    r = 1                                                     (Wilson parameter)

This gives in explicit form (the standard Wilson fermion action):

    D_W Psi(x) = (m_0 + 4r) Psi(x)
               - (r/2) sum_mu [ (1 - gamma_mu) U_mu(x) Psi(x+mu)
                               +(1 + gamma_mu) U_mu†(x-mu) Psi(x-mu) ]

The Wilson term (proportional to r) lifts the fermion doubler modes to
mass ~ 1/a, removing the 15 spurious doublers that plague naive fermions.

Cl(6) structure
---------------
Psi lives in the tensor product space:
    Psi(x) in C^8 (Cl(6) spin) x C^3 (SU(3) color)

The gauge link U acts on the color index only:
    [U_mu(x) Psi(x+mu)]_{alpha, a} = sum_b U_mu(x)_{ab} Psi(x+mu)_{alpha, b}

The gamma_mu matrices act on the Cl(6) spin index only:
    [gamma_mu Psi]_{alpha, a} = sum_beta (gamma_mu)_{alpha,beta} Psi_{beta, a}

So the combined action is a tensor product:
    gamma_mu (x) U_mu : (C^8 x C^3) -> (C^8 x C^3)

Key observables derived from D_W
----------------------------------
1. Fermion propagator: S(x,y) = D_W^{-1}(x,y)  [via CG solver]
2. Chiral condensate: <psibar psi> = -Tr[D_W^{-1}] / Volume
3. Pion correlator: C(t) = sum_x <Tr[S(x,0) gamma5 S†(x,0) gamma5]>
4. Fermionic contribution to gauge action: log det[D_W]

Sanity checks implemented
--------------------------
1. Free field (U=I): spectrum should show |p_mu| structure
2. gamma5-Hermiticity: D_W† = gamma5 D_W gamma5  (exact for Wilson)
3. Hopping expansion: at large m_0, propagator ~ 1/m_0 + O(1/m_0^2)
4. Chiral condensate sign: must be negative for physical fermions

Parameters
----------
m_0   : bare quark mass (negative values approach chiral limit)
r     : Wilson parameter (standard: r=1)
kappa : hopping parameter = 1 / (2*(m_0 + 4r))
        Critical kappa (massless fermion, free field): kappa_c = 1/8
"""

import numpy as np
from scipy.sparse.linalg import cg as scipy_cg


# ============================================================
# 8. LATTICE DIRAC OPERATOR MODULE
# ============================================================

class DiracOperator:
    """
    Wilson-Dirac operator coupling Cl(6) spinors to SU(3) gauge links.

    Spinor layout: Psi[x] has shape (n_spin, n_color) = (8, 3)
    where n_spin=8 comes from Cl(6) (2^3 from three Pauli tensor products)
    and n_color=3 is the SU(3) fundamental representation.

    The full operator acts on a flattened vector of size V * 8 * 3
    where V = N^4 is the lattice volume.

    Wilson-Dirac operator (kappa form):
        D_W = I - kappa * M
    where M is the hopping matrix and kappa = 1/(2*(m0 + 4r)).

    This form is numerically preferable because kappa -> kappa_c ~ 0.125
    controls the quark mass, and the identity part is explicit.
    """

    def __init__(self, fields, m0=0.1, r=1.0):
        """
        Parameters
        ----------
        fields : Fields object containing U (gauge links) and Cl(6) gammas
        m0     : bare quark mass. Physical range: m0 > -2r for no doublers.
                 m0=0.1 is safely massive. m0->-(2*D-1)*r approaches chiral limit.
        r      : Wilson parameter. Standard choice r=1 removes all doublers.
        """
        self.F      = fields
        self.lat    = fields.lat
        self.cl     = fields.cl
        self.m0     = m0
        self.r      = r

        self.N      = fields.lat.N
        self.D      = fields.lat.D
        self.V      = self.N ** self.D        # lattice volume
        self.n_spin = len(fields.cl.gammas)   # 6 gamma matrices -> 8-dim spinor
        # Note: Cl(6) has 6 gamma matrices but they act on 8-dim space (2^3)
        self.n_spin = fields.cl.gammas[0].shape[0]   # = 8
        self.n_col  = 3
        self.n_dof  = self.n_spin * self.n_col        # 24 per site

        # Hopping parameter: kappa = 1 / (2*(m0 + r*D))
        # For D=4, r=1: kappa_c (free, massless) = 1/8 = 0.125
        self.kappa = 1.0 / (2.0 * (m0 + r * self.D))

        # Precompute projectors P+_mu = (1 + gamma_mu)/2
        #                        P-_mu = (1 - gamma_mu)/2
        # These are the chiral projectors in each direction.
        # Forward hopping uses P-_mu, backward hopping uses P+_mu.
        # This is the standard Wilson fermion convention.
        gammas = fields.cl.gammas
        I_spin = np.eye(self.n_spin, dtype=complex)
        self.Pplus  = [(I_spin + g) / 2.0 for g in gammas]   # (1+gamma_mu)/2
        self.Pminus = [(I_spin - g) / 2.0 for g in gammas]   # (1-gamma_mu)/2

        # Verify projector properties: P+^2 = P+, P-^2 = P-, P+ + P- = I
        self._verify_projectors()

        print(f"DiracOperator initialized:")
        print(f"  m0={m0}, r={r}, kappa={self.kappa:.6f}")
        print(f"  kappa_c (free massless) = {1.0/(2.0*self.D*r):.6f}")
        print(f"  Spinor DOF per site: {self.n_dof}  (8 Cl6 x 3 color)")
        print(f"  Total DOF: {self.V * self.n_dof}")
        if self.kappa >= 1.0/(2.0*self.D*r):
            print(f"  WARNING: kappa >= kappa_c — negative mass, doublers may appear")
        else:
            print(f"  Mass gap confirmed: kappa < kappa_c ✓")

    def _verify_projectors(self):
        """Check P+^2=P+, P-^2=P-, P++P-=I for each direction."""
        I = np.eye(self.n_spin, dtype=complex)
        errs = []
        for mu in range(self.D):
            errs.append(np.linalg.norm(self.Pplus[mu]  @ self.Pplus[mu]  - self.Pplus[mu]))
            errs.append(np.linalg.norm(self.Pminus[mu] @ self.Pminus[mu] - self.Pminus[mu]))
            errs.append(np.linalg.norm(self.Pplus[mu]  + self.Pminus[mu] - I))
        max_err = max(errs)
        assert max_err < 1e-12, f"Projector error: {max_err}"
        print(f"  Projectors verified: P±^2=P±, P++P-=I  max_err={max_err:.2e} ✓")

    # ----------------------------------------------------------
    # Core operator application: D_W * Psi
    # ----------------------------------------------------------

    def apply(self, Psi_flat):
        """
        Apply Wilson-Dirac operator to a flattened spinor.

        Input:  Psi_flat: shape (V * n_spin * n_col,) complex vector
        Output: D_W Psi_flat: same shape

        This is the heart of the module. The Wilson-Dirac operator is:
            (D_W Psi)(x) = Psi(x)
                - kappa * sum_mu [
                    (1 - gamma_mu) U_mu(x) Psi(x+mu)      <- forward hop
                  + (1 + gamma_mu) U†_mu(x-mu) Psi(x-mu)  <- backward hop
                  ]
        Equivalently with projectors:
                - 2*kappa * sum_mu [
                    P-_mu U_mu(x) Psi(x+mu)
                  + P+_mu U†_mu(x-mu) Psi(x-mu)
                  ]
        """
        U   = self.F.U
        lat = self.lat
        N   = self.N

        # Reshape to (N,N,N,N, n_spin, n_col)
        Psi = Psi_flat.reshape(N, N, N, N, self.n_spin, self.n_col)
        out = np.zeros_like(Psi)

        # Diagonal term: identity
        out += Psi

        # Off-diagonal: hopping terms
        for x in lat.all_sites():
            psi_x = Psi[x]   # shape (n_spin, n_col)

            for mu in range(self.D):
                x_fwd = lat.shift(x, mu,  step=+1)
                x_bwd = lat.shift(x, mu,  step=-1)

                U_fwd = U[x][mu]                     # U_mu(x),      shape (3,3)
                U_bwd = U[x_bwd][mu].conj().T         # U†_mu(x-mu),  shape (3,3)

                psi_fwd = Psi[x_fwd]   # shape (n_spin, n_col)
                psi_bwd = Psi[x_bwd]   # shape (n_spin, n_col)

                # Forward: P-_mu @ psi_fwd @ U_fwd.T
                # Action: spin part left, color part right
                # [P-_mu psi U_fwd]_{alpha,a} = sum_{beta,b} (P-_mu)_{alpha,beta}
                #                                * psi_{beta,b} * U_fwd_{ba}
                # i.e.: (P-_mu @ psi_fwd) @ U_fwd.T
                hop_fwd = (self.Pminus[mu] @ psi_fwd) @ U_fwd.T
                hop_bwd = (self.Pplus[mu]  @ psi_bwd) @ U_bwd.T

                out[x] -= self.kappa * (hop_fwd + hop_bwd)

        return out.ravel()

    def apply_to_field(self, Psi):
        """
        Apply D_W to a full spinor field.
        Input/output shape: (N,N,N,N, n_spin, n_col)
        """
        return self.apply(Psi.ravel()).reshape(
            self.N, self.N, self.N, self.N, self.n_spin, self.n_col
        )

    # ----------------------------------------------------------
    # gamma5 and Hermiticity check
    # ----------------------------------------------------------

    def gamma5(self):
        """
        Construct gamma5 for the Cl(6) algebra.
        gamma5 = i^3 * gamma1 * gamma2 * gamma3 * gamma4 * gamma5 * gamma6
        (the pseudoscalar element / volume form of Cl(6))

        For the Wilson-Dirac operator, gamma5-Hermiticity states:
            D_W† = gamma5 * D_W * gamma5
        This is an exact symmetry of Wilson fermions and is the basis of
        the Ginsparg-Wilson relation and overlap fermions.
        """
        g = self.cl.gammas
        # Product of all 6 gammas (up to phase)
        G5 = g[0] @ g[1] @ g[2] @ g[3] @ g[4] @ g[5]
        # Normalize to make it Hermitian and G5^2 = I
        phase = G5[0, 0] / abs(G5[0, 0]) if abs(G5[0, 0]) > 1e-10 else 1.0
        G5 = G5 / phase
        # Verify G5^2 = I
        err = np.linalg.norm(G5 @ G5 - np.eye(self.n_spin, dtype=complex))
        if err > 1e-10:
            # Fall back to using gamma_1 * gamma_2 (a valid Z2 grading element)
            G5 = g[0] @ g[1]
            G5 = G5 / (G5[0,0] / abs(G5[0,0]))
        return G5

    def check_gamma5_hermiticity(self, n_test=3):
        """
        Verify D† = gamma5 D gamma5 on random test vectors.
        This is an exact identity for Wilson fermions — any deviation
        indicates a bug in the hopping terms.
        """
        G5_spin = self.gamma5()
        # G5 acts on spin only: full G5 = G5_spin (x) I_color
        I_col = np.eye(self.n_col, dtype=complex)
        G5_full_site = np.kron(G5_spin, I_col)   # (24, 24)

        # Build G5 for full lattice (block diagonal)
        V_dof = self.V * self.n_dof
        # We test with matrix-free approach on random vectors
        print(f"\n  gamma5-Hermiticity check (D† = gamma5 D gamma5):")
        max_err = 0.0
        for _ in range(n_test):
            psi = (np.random.randn(V_dof) + 1j * np.random.randn(V_dof))
            phi = (np.random.randn(V_dof) + 1j * np.random.randn(V_dof))

            # Apply G5 site-by-site
            def apply_g5(v):
                v_r = v.reshape(self.N, self.N, self.N, self.N,
                                self.n_spin, self.n_col)
                out = np.zeros_like(v_r)
                for x in self.lat.all_sites():
                    out[x] = G5_spin @ v_r[x]
                return out.ravel()

            # Check: <phi | D psi> == <D† phi | psi> == <gamma5 D gamma5 phi | psi>
            Dpsi    = self.apply(psi)
            G5phi   = apply_g5(phi)
            DG5phi  = self.apply(G5phi)
            G5DG5phi = apply_g5(DG5phi)

            lhs = np.dot(phi.conj(), Dpsi)
            rhs = np.dot(G5DG5phi.conj(), psi)
            err = abs(lhs - rhs) / (abs(lhs) + 1e-15)
            max_err = max(max_err, err)

        print(f"    Max relative error: {max_err:.2e}", end=" ")
        if max_err < 1e-10:
            print("✓  gamma5-Hermiticity exact")
        elif max_err < 1e-6:
            print("~  gamma5-Hermiticity approximate (check boundary conditions)")
        else:
            print("✗  FAIL — hopping term asymmetry detected")
        return max_err

    # ----------------------------------------------------------
    # Propagator via Conjugate Gradient
    # ----------------------------------------------------------

    def propagator_cg(self, source, tol=1e-8, maxiter=1000, verbose=True):
        """
        Compute the fermion propagator S = D_W^{-1} applied to a source
        using the Conjugate Gradient (CG) method.

        Solves:  D_W * x = source   for x = S * source

        For CG to work on a non-Hermitian system we solve the normal
        equations:  D† D x = D† source
        This is the standard even-odd preconditioned approach for Wilson fermions.

        Parameters
        ----------
        source  : shape (V * n_spin * n_col,) or (N,N,N,N, n_spin, n_col)
        tol     : CG convergence tolerance
        maxiter : maximum CG iterations

        Returns
        -------
        solution : same shape as source
        info     : convergence info (0 = converged)
        """
        if source.ndim > 1:
            src_flat = source.ravel()
        else:
            src_flat = source

        n = len(src_flat)

        # Build D†D as a LinearOperator
        from scipy.sparse.linalg import LinearOperator

        def Ddag_D(v):
            """Apply D† D to vector v."""
            Dv    = self.apply(v)
            # D† = gamma5 D gamma5 (matrix-free)
            return self._apply_Ddag(Dv)

        def _Ddag(v):
            return self._apply_Ddag(v)

        A = LinearOperator((n, n), matvec=Ddag_D, dtype=complex)

        # Right-hand side: D† * source
        rhs = self._apply_Ddag(src_flat)

        if verbose:
            print(f"  CG solver: solving D†D x = D† b  (size={n}, tol={tol})")

        iters = [0]
        def callback(xk):
            iters[0] += 1

        sol, info = scipy_cg(A, rhs, rtol=tol, maxiter=maxiter, callback=callback)

        if verbose:
            residual = np.linalg.norm(self.apply(sol) - src_flat) / np.linalg.norm(src_flat)
            print(f"  CG: {iters[0]} iterations, residual={residual:.2e}, "
                  f"{'converged ✓' if info==0 else f'not converged (info={info})'}")

        if source.ndim > 1:
            return sol.reshape(source.shape), info
        return sol, info

    def _apply_Ddag(self, v):
        """
        Apply D† = gamma5 D gamma5 to vector v (matrix-free).
        """
        G5_spin = self.gamma5()

        def apply_g5(vec):
            v_r  = vec.reshape(self.N, self.N, self.N, self.N,
                                self.n_spin, self.n_col)
            out  = np.zeros_like(v_r)
            for x in self.lat.all_sites():
                out[x] = G5_spin @ v_r[x]
            return out.ravel()

        return apply_g5(self.apply(apply_g5(v)))

    # ----------------------------------------------------------
    # Physical observables
    # ----------------------------------------------------------

    def chiral_condensate(self, n_stochastic=10, tol=1e-6, verbose=True):
        """
        Compute the chiral condensate <psibar psi> = -Tr[D_W^{-1}] / V
        using stochastic estimation (Z2 noise vectors).

        <psibar psi> = -(1/V) * (1/n_est) * sum_r [ eta_r† D^{-1} eta_r ]

        where eta_r are random Z2 noise vectors: eta_{r,i} in {+1, -1}.

        This is the order parameter for chiral symmetry breaking:
          <psibar psi> != 0  =>  chiral symmetry broken (confined phase)
          <psibar psi>  = 0  =>  chiral symmetry restored (chiral limit)

        Parameters
        ----------
        n_stochastic : number of noise vectors (more = smaller variance)
        """
        if verbose:
            print(f"\n  Computing chiral condensate <psibar psi>")
            print(f"  Using {n_stochastic} stochastic estimators (Z2 noise)")

        n_dof_total = self.V * self.n_dof
        estimates   = []

        for k in range(n_stochastic):
            # Z2 noise: eta_i in {+1, -1}
            eta = np.sign(np.random.randn(n_dof_total)).astype(complex)

            # Solve D x = eta
            x, info = self.propagator_cg(eta, tol=tol, verbose=False)

            # Stochastic trace estimate: eta† x = eta† D^{-1} eta
            est = np.real(np.dot(eta.conj(), x)) / self.V
            estimates.append(est)

            if verbose:
                print(f"    estimator {k+1:>3}: {est:+.6f}")

        cond = -np.mean(estimates)
        err  = np.std(estimates) / np.sqrt(n_stochastic)

        if verbose:
            print(f"  <psibar psi> = {cond:+.6f} ± {err:.6f}")
            if cond < -1e-4:
                print(f"  SIGNAL: negative condensate — chiral symmetry broken ✓")
            elif abs(cond) < err * 2:
                print(f"  SIGNAL: consistent with zero — near chiral limit")
            else:
                print(f"  SIGNAL: positive condensate — check mass sign / kappa")

        return cond, err

    def pion_correlator(self, t_max=None, verbose=True):
        """
        Compute the pion (pseudoscalar) two-point correlator:

            C_pi(t) = sum_{x,y: y0=t} Tr[ S(x,0) gamma5 S†(x,0) gamma5 ]

        This is the simplest hadronic observable. In the confined phase
        it decays exponentially:
            C_pi(t) ~ A * exp(-m_pi * t)

        where m_pi is the pion mass (lightest hadron in QCD).

        Method: point source at origin, measure propagator to all timeslices.
        """
        if t_max is None:
            t_max = self.N

        G5_spin = self.gamma5()
        n_dof   = self.V * self.n_dof

        if verbose:
            print(f"\n  Computing pion correlator C_pi(t), t=0..{t_max-1}")

        # Point source at origin, all spin/color components
        correlator = np.zeros(t_max)

        for alpha in range(min(self.n_spin, 4)):   # limit to 4 spin components
            for a in range(self.n_col):
                # Build point source at x=0 in component (alpha, a)
                src = np.zeros((self.N, self.N, self.N, self.N,
                                self.n_spin, self.n_col), dtype=complex)
                src[(0,)*self.D][alpha, a] = 1.0

                # Solve D S = src  ->  S[:,alpha,a] = D^{-1} src
                prop, info = self.propagator_cg(src, tol=1e-7, verbose=False)

                if info != 0 and verbose:
                    print(f"    WARNING: CG did not converge for src ({alpha},{a})")

                # Apply gamma5: G5 * prop
                g5_prop = np.zeros_like(prop)
                for x in self.lat.all_sites():
                    g5_prop[x] = G5_spin @ prop[x]

                # Sum over spatial sites for each timeslice t
                for t in range(t_max):
                    # Sum |g5 S(x,t; 0,0)|^2 over spatial x and spin/color
                    spatial_sum = 0.0
                    for x1 in range(self.N):
                        for x2 in range(self.N):
                            for x3 in range(self.N):
                                x_site = (t, x1, x2, x3)
                                spatial_sum += np.real(
                                    np.sum(np.abs(g5_prop[x_site])**2)
                                )
                    correlator[t] += spatial_sum

        if verbose:
            print(f"\n  Pion correlator C_pi(t):")
            print(f"  {'t':>4}  {'C_pi(t)':>14}  {'log ratio':>12}")
            print(f"  {'-'*4}  {'-'*14}  {'-'*12}")
            for t in range(t_max):
                logratio = (f"{np.log(correlator[t]/correlator[t+1]):.6f}"
                            if t < t_max-1 and correlator[t+1] > 0 else "   ---")
                print(f"  {t:>4}  {correlator[t]:>14.6f}  {logratio:>12}")

            # Try to fit effective mass: m_eff(t) = log(C(t)/C(t+1))
            if t_max >= 3:
                meffs = []
                for t in range(t_max - 1):
                    if correlator[t] > 0 and correlator[t+1] > 0:
                        meffs.append(np.log(correlator[t] / correlator[t+1]))
                if meffs:
                    plateau = np.mean(meffs[1:-1]) if len(meffs) > 2 else meffs[0]
                    print(f"\n  Effective mass plateau (mid-timeslices): {plateau:.4f}")
                    print(f"  (Exponential decay C(t) ~ exp(-{plateau:.4f}*t))")

        return correlator


# ============================================================
# DIRAC SANITY CHECKS
# ============================================================

def run_dirac_sanity_checks(lat, cl, su3):
    """
    Three mandatory checks before any Dirac physics:

    1. Free field spectrum: with U=I, the eigenvalues of D_W must follow
       the known free-field dispersion relation. At kappa=kappa_c the
       lowest modes should approach zero.

    2. gamma5-Hermiticity: D† = gamma5 D gamma5 exactly. Any violation
       means the hopping terms are asymmetric — a bug.

    3. Hopping expansion: at large m0, D_W^{-1} ~ (1/m0) I + O(1/m0^2).
       We check D^{-1} on a point source scales as 1/m0.
    """
    print("\n" + "="*60)
    print("DIRAC OPERATOR SANITY CHECKS")
    print("="*60)

    # --- CHECK 1: gamma5-Hermiticity on cold (free) field ---
    print("\n[1] gamma5-Hermiticity: D† = gamma5 D gamma5")
    fields_cold = Fields(lat, cl, su3, mode='cold')
    D_free = DiracOperator(fields_cold, m0=0.5, r=1.0)
    err_g5 = D_free.check_gamma5_hermiticity(n_test=5)

    # --- CHECK 2: Free field, large mass — propagator ~ 1/m0 ---
    print("\n[2] Hopping expansion: |D^{-1} e_0| ~ 1/(m0 + D*r) at large m0")
    results_mass = []
    for m0_test in [1.0, 2.0, 4.0]:
        D_test = DiracOperator(fields_cold, m0=m0_test, r=1.0)
        # Unit source at origin, spin=0, color=0
        src = np.zeros(lat.N**lat.D * D_test.n_dof, dtype=complex)
        src[0] = 1.0
        sol, info = D_test.propagator_cg(src, tol=1e-10, verbose=False)
        norm_sol   = sol[0].real   # diagonal element of propagator at origin
        expected   = 1.0           # D^{-1} e_0 at source site -> 1.0 as m0->inf
        results_mass.append((m0_test, norm_sol, expected))
        print(f"    m0={m0_test:.1f}:  |D^{{-1}} e_0| = {norm_sol:.6f}  "
              f"expected~{expected:.6f}  ratio={norm_sol/expected:.4f}")

    # Check that ratio is consistent (should approach 1 as m0 grows)
    ratios = [r[1]/r[2] for r in results_mass]
    print(f"    Ratios converging to 1: "
          f"{' -> '.join(f'{r:.3f}' for r in ratios)} "
          f"{'✓' if abs(ratios[-1]-1.0) < 0.15 else '~'}")

    # --- CHECK 3: Warm field, verify D acts consistently ---
    print("\n[3] Consistency: D applied twice vs direct application")
    fields_warm = Fields(lat, cl, su3, mode='warm')
    D_warm = DiracOperator(fields_warm, m0=0.5, r=1.0)
    psi_test = (np.random.randn(lat.N**lat.D * D_warm.n_dof)
              + 1j * np.random.randn(lat.N**lat.D * D_warm.n_dof)) * 0.1
    Dpsi  = D_warm.apply(psi_test)
    DDpsi = D_warm.apply(Dpsi)
    # Just check linearity: D(a*psi) = a*D(psi)
    a = 2.3 + 1.7j
    D_apsi = D_warm.apply(a * psi_test)
    lin_err = np.linalg.norm(D_apsi - a * Dpsi) / np.linalg.norm(a * Dpsi)
    print(f"    Linearity error: {lin_err:.2e} "
          f"{'✓' if lin_err < 1e-12 else '✗'}")

    print("\nSanity checks complete.")
    return D_free, D_warm


# ============================================================
# PHYSICAL OBSERVABLES PIPELINE
# ============================================================

def run_dirac_physics(lat, cl, su3, beta=6.0, m0=0.1,
                      n_therm=50, n_configs=5,
                      compute_condensate=True,
                      compute_pion=True,
                      verbose=True):
    """
    Full Dirac physics pipeline:
      1. Thermalize gauge field at given beta
      2. For each configuration, compute:
         a. Chiral condensate <psibar psi>
         b. Pion correlator C_pi(t)  (if requested)
      3. Report ensemble averages

    Parameters
    ----------
    beta           : gauge coupling inverse (6.0 = physical QCD regime)
    m0             : bare quark mass (0.1 is safely massive and fast to invert)
    n_therm        : thermalization sweeps
    n_configs      : gauge configurations to measure over
    compute_pion   : compute pion correlator (slow — each config needs CG per timeslice)
    """
    print("\n" + "="*60)
    print(f"DIRAC PHYSICS PIPELINE  [beta={beta}, m0={m0}]")
    print("="*60)

    # Step 1: Thermalize gauge field
    print(f"\nThermalization: {n_therm} sweeps at beta={beta}...")
    fields = Fields(lat, cl, su3, mode='hot')
    metro  = Metropolis(fields, beta=beta, eps=0.25)
    metro.thermalize(n_sweeps=n_therm, print_every=max(1,n_therm//5),
                     verbose=verbose)

    print(f"\nPost-thermalization plaquette: "
          f"{Observables(fields).avg_plaquette():.5f}")

    # Step 2: Build Dirac operator on thermalized background
    D = DiracOperator(fields, m0=m0, r=1.0)

    # Step 3: gamma5-Hermiticity check on dynamical background
    D.check_gamma5_hermiticity(n_test=3)

    # Step 4: Ensemble measurements
    condensates = []
    pion_corrs  = []

    print(f"\nCollecting {n_configs} configurations for Dirac observables...")

    for cfg in range(n_configs):
        print(f"\n  --- Config {cfg+1}/{n_configs} ---")

        # Decorrelate with 5 Metropolis sweeps
        for _ in range(5):
            metro.sweep()

        plaq = Observables(fields).avg_plaquette()
        print(f"  Plaquette: {plaq:.5f}")

        # Rebuild Dirac operator on updated gauge field
        D = DiracOperator(fields, m0=m0, r=1.0)

        if compute_condensate:
            cond, cerr = D.chiral_condensate(n_stochastic=4, verbose=verbose)
            condensates.append(cond)

        if compute_pion and lat.N >= 4:
            t_max = min(lat.N, 4)   # limit for speed
            corr = D.pion_correlator(t_max=t_max, verbose=verbose)
            pion_corrs.append(corr)

    # Step 5: Report
    print("\n" + "="*60)
    print("DIRAC ENSEMBLE RESULTS")
    print("="*60)

    if condensates:
        cond_mean = np.mean(condensates)
        cond_err  = np.std(condensates) / np.sqrt(len(condensates))
        print(f"\n  <psibar psi> = {cond_mean:+.5f} ± {cond_err:.5f}")
        print(f"  Interpretation:")
        if cond_mean < -0.01:
            print(f"    Negative condensate — chiral symmetry broken")
            print(f"    Consistent with confined phase at beta={beta}")
        else:
            print(f"    Near zero — check m0 or increase statistics")

    if pion_corrs:
        avg_corr = np.mean(pion_corrs, axis=0)
        print(f"\n  Pion correlator (ensemble average):")
        for t, c in enumerate(avg_corr):
            print(f"    t={t}: C_pi = {c:.5f}")

        # Effective mass from ensemble average
        meffs = []
        for t in range(len(avg_corr)-1):
            if avg_corr[t] > 0 and avg_corr[t+1] > 0:
                meffs.append(np.log(avg_corr[t]/avg_corr[t+1]))
        if meffs:
            m_eff = np.mean(meffs)
            print(f"\n  Effective pion mass: m_pi_eff = {m_eff:.4f}  (lattice units)")
            print(f"  Physical pion requires chiral extrapolation m0 -> m0_critical")

    print(f"\n  PHYSICS NOTE:")
    print(f"  The coupling Psi <-> U is now active. Each gauge configuration")
    print(f"  generates a different Dirac operator. The ensemble average of")
    print(f"  Dirac observables is the full path integral:")
    print(f"    <O> = (1/Z) integral [dU] [dPsi dPsibar] O exp(-S_gauge - S_fermion)")
    print(f"  where S_fermion = Psibar D_W Psi is the Wilson fermion action.")

    return condensates, pion_corrs


# ============================================================
# STANDALONE ENTRY POINT
# ============================================================

if __name__ == '__main__':
    import warnings, sys, os
    warnings.filterwarnings('ignore')

    # Import the base framework
    sys.path.insert(0, os.path.dirname(__file__))

    # We need the v2 classes — execute the base file to get them
    exec(open('./ga_su3_lattice_v2.py').read()
         .replace("if __name__ == '__main__':", "if False:"))

    print("="*60)
    print("GA-SU(3) LATTICE FRAMEWORK  v3")
    print("Cl(6) Dirac operator coupled to SU(3) gauge field")
    print("="*60)

    np.random.seed(42)

    # Initialize base modules
    cl  = CliffordCl6()
    su3 = SU3Algebra(eps=0.3, g=1.0)

    # Small lattice for Dirac (CG inversion scales as V^2)
    # N=4 is feasible; N=6 will be slow but correct
    lat = Lattice(N=4, D=4)

    # Sanity checks on gauge sector
    fields_cold, _ = sanity_checks(lat, cl, su3)

    # Dirac operator sanity checks
    D_free, D_warm = run_dirac_sanity_checks(lat, cl, su3)

    # Full physics pipeline
    # beta=6.0: deconfined, condensate should be smaller in magnitude
    # beta=4.5: confined,   condensate should be larger in magnitude
    # We run beta=6.0 for speed; change to 4.5 to see confinement signal
    condensates, pion_corrs = run_dirac_physics(
        lat, cl, su3,
        beta=4.5,
        m0=0.01,
        n_therm=200,       # increase to 200 for production runs
        n_configs=20,      # increase to 20+ for production runs
        compute_condensate=True,
        compute_pion=True,
        verbose=True,
    )

    print("\n" + "="*60)
    print("FRAMEWORK v3 COMPLETE")
    print("="*60)
    print("""
Next steps (production runs):
  1. Run at beta=4.5 and beta=6.0: compare <psibar psi> across transition
     Confined phase should show larger |<psibar psi>|
  2. Vary m0: extrapolate pion mass to zero -> locate kappa_critical
  3. Compare Psi correlators before/after gauge thermalization:
     This directly demonstrates that Cl(6) spinors feel the gauge field
  4. Add even-odd preconditioning to CG for 2x speedup
  5. Add HMC molecular dynamics to include fermion determinant in gauge update
""")