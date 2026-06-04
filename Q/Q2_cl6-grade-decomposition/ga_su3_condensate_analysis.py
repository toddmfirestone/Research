#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cl6_su3_centralizer_check.py
============================================================================
CONFIRMING CHECK for the virtual-photon / double-booking investigation.

Claim under test (premise (1) of the B-L reductio):

    The centralizer of su(3) inside the Cl(6) bivector Lie algebra
    so(6) = spin(6) is EXACTLY one-dimensional, and equals span{H},
    where H = B12 + B34 + B56 is the Cartan hypercharge
    (Paper 2, Eq. (cartan-H); H = (3/2) Q_{B-L}).

If confirmed: the only bivector commuting with all of color su(3) is H,
hence any color-neutral generator (e.g. a photon transverse-helicity
generator) would have to be proportional to H = (3/2)Q_{B-L} -- i.e. would
carry B-L charge. Photons carry no B-L. Therefore the transverse-helicity
generator cannot be color-neutral: the (d-2)=2 transverse polarization
modes are color-charged, and the polarization wall coincides with the
Coleman-Mandula spatial-rotation/color double-booking wall (C4 sec.6).

This script does NOT pin down which color-mover the helicity generator is
(that needs the emergent spatial directions, open in the C-series). It ONLY
confirms premise (1) in Todd's actual Cl(6) representation, and as a
by-product tests whether a genuine bivector su(3) octet closes in this rep
(the Paper2-prose-vs-code gap flagged in discussion).

CONSTRUCTION (principled, not hand-picked):
  - su(3) is DERIVED as the stabilizer of the preferred spinor |xi0> (the
    H = -3/2 state), i.e. the set of bivector combinations X with X|xi0> = 0.
  - Its closure under commutator is verified (genuine Lie subalgebra).
  - Its centralizer in the 15-dim bivector space is then solved as a linear
    system and checked to be span{H}.

Conventions match Paper 2:
  e_i Hermitian, e_i^2 = I8 ; B_ij = (i/4)[e_i, e_j] (Hermitian generators);
  H = B12 + B34 + B56 = (1/2) diag(-3,-1,-1,+1,-1,+1,+1,+3).

Environment: Python 3.11, NumPy. Pure linear algebra; no external data.
Output: printed to stdout AND teed to cl6_su3_centralizer_check_report.txt
in the script's own run directory.
============================================================================
"""

import sys
import os
import numpy as np
from itertools import combinations


# ----------------------------------------------------------------------------
# Tee: mirror all stdout to a local .txt report (per workflow convention)
# ----------------------------------------------------------------------------
class _Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
    def flush(self):
        for s in self.streams:
            s.flush()


def _open_report():
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "cl6_su3_centralizer_check_report.txt")
    fh = open(path, "w", encoding="utf-8")
    return fh, path


# ----------------------------------------------------------------------------
# Linear algebra helpers
# ----------------------------------------------------------------------------
def fnorm(M):
    return float(np.linalg.norm(M, "fro"))


def real_nullspace(A, tol=1e-10):
    """
    Real nullspace of a complex linear map A: R^n -> C^m applied to real
    coefficient vectors. We split into real/imag parts and use SVD.

    A is (m x n) complex; we want real c (length n) with A c = 0.
    Stack [Re A; Im A] (2m x n real) and take the SVD nullspace.
    Returns an (n x k) matrix whose columns are an orthonormal basis of the
    real nullspace; k = nullity.
    """
    Ar = np.vstack([A.real, A.imag])            # (2m x n) real
    # SVD
    U, s, Vt = np.linalg.svd(Ar, full_matrices=True)
    n = Ar.shape[1]
    # singular values beyond rank
    rank = int(np.sum(s > tol * max(1.0, s[0] if s.size else 1.0)))
    null_basis = Vt[rank:].T.conj()             # (n x (n-rank))
    return null_basis.real, rank


def commutator(A, B):
    return A @ B - B @ A


# ----------------------------------------------------------------------------
# 1. Build Cl(6) generators and bivectors
# ----------------------------------------------------------------------------
def build_clifford():
    I2 = np.eye(2, dtype=complex)
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)

    def kron(*ops):
        out = np.array([[1]], dtype=complex)
        for op in ops:
            out = np.kron(out, op)
        return out

    e = [
        kron(sx, I2, I2),   # e1
        kron(sy, I2, I2),   # e2
        kron(sz, sx, I2),   # e3
        kron(sz, sy, I2),   # e4
        kron(sz, sz, sx),   # e5
        kron(sz, sz, sy),   # e6
    ]
    return e


def build_bivectors(e):
    """B_ij = (i/4)[e_i, e_j], Hermitian generators of spin(6). 15 of them."""
    bivs = []
    labels = []
    for i, j in combinations(range(6), 2):
        B = (1j / 4.0) * (e[i] @ e[j] - e[j] @ e[i])
        bivs.append(B)
        labels.append(f"B{i+1}{j+1}")
    return bivs, labels


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    np.set_printoptions(precision=4, suppress=True, linewidth=120)

    print("=" * 76)
    print("  Cl(6) su(3) CENTRALIZER CHECK  --  confirming premise (1)")
    print("=" * 76)

    # --- build algebra ---
    e = build_clifford()

    # verify Clifford relations
    cl_err = 0.0
    for i in range(6):
        for j in range(6):
            ac = e[i] @ e[j] + e[j] @ e[i]
            exp = 2 * np.eye(8, dtype=complex) if i == j else np.zeros((8, 8), dtype=complex)
            cl_err = max(cl_err, fnorm(ac - exp))
    print(f"\n[A] Clifford relations {{e_i,e_j}}=2 delta_ij : max err = {cl_err:.2e}",
          "PASS" if cl_err < 1e-12 else "FAIL")

    bivs, labels = build_bivectors(e)
    print(f"[B] Built {len(bivs)} bivectors B_ij = (i/4)[e_i,e_j] (expect 15):",
          "PASS" if len(bivs) == 15 else "FAIL")

    # Hermiticity of bivectors
    herm_err = max(fnorm(B - B.conj().T) for B in bivs)
    print(f"[C] Bivectors Hermitian : max err = {herm_err:.2e}",
          "PASS" if herm_err < 1e-12 else "FAIL")

    # --- Cartan hypercharge H ---
    idx = {lab: k for k, lab in enumerate(labels)}
    H = bivs[idx["B12"]] + bivs[idx["B34"]] + bivs[idx["B56"]]
    Hdiag = np.diag(H).real
    expected_Hdiag = 0.5 * np.array([-3, -1, -1, +1, -1, +1, +1, +3])
    H_err = float(np.max(np.abs(Hdiag - expected_Hdiag)))
    print(f"[D] H = B12+B34+B56 diagonal matches Paper 2 (eq H-diag): max err = {H_err:.2e}",
          "PASS" if H_err < 1e-12 else "FAIL")
    print(f"      diag(H) = {Hdiag}")

    # --- preferred spinor |xi0> : the H = -3/2 eigenstate ---
    xi_idx = int(np.argmin(np.abs(Hdiag - (-1.5))))
    xi0 = np.zeros(8, dtype=complex)
    xi0[xi_idx] = 1.0
    print(f"[E] Preferred spinor |xi0> = e_{xi_idx} (H eigenvalue {Hdiag[xi_idx]:+.1f})")

    # --- derive su(3) = stabilizer of |xi0> within span(bivectors) ---
    # map M: R^15 -> C^8 ,  c -> (sum_a c_a B_a) |xi0>
    # columns = B_a |xi0>
    A = np.column_stack([B @ xi0 for B in bivs])           # (8 x 15) complex
    null_c, rank = real_nullspace(A, tol=1e-9)             # (15 x k)
    k = null_c.shape[1]
    print(f"\n[F] su(3) = {{ X in span(B) : X|xi0> = 0 }} ")
    print(f"      dim of stabilizer = {k}  (expect 8)",
          "PASS" if k == 8 else "FAIL")

    # materialize su(3) basis as 8x8 matrices
    su3 = []
    for col in range(k):
        c = null_c[:, col]
        X = sum(c[a] * bivs[a] for a in range(15))
        su3.append(X)

    # confirm they really annihilate xi0
    ann_err = max(np.linalg.norm(X @ xi0) for X in su3) if su3 else 0.0
    print(f"      max |X|xi0>| over basis = {ann_err:.2e}",
          "PASS" if ann_err < 1e-8 else "FAIL")

    # confirm H is NOT in the stabilizer (carries -3/2 on xi0)
    print(f"      H|xi0> norm = {np.linalg.norm(H @ xi0):.4f}  (must be nonzero: H excluded)")

    # --- verify su(3) closes as a Lie algebra (tests prose-vs-code gap) ---
    # project commutators back onto span(su3) and measure residual
    # Build orthonormal (Hilbert-Schmidt) basis of su3 for clean projection.
    def hs_inner(A_, B_):
        return np.trace(A_.conj().T @ B_)

    # Gram-Schmidt in HS inner product
    onb = []
    for X in su3:
        Y = X.copy().astype(complex)
        for Q in onb:
            Y = Y - hs_inner(Q, Y) * Q
        nrm = np.sqrt(abs(hs_inner(Y, Y)))
        if nrm > 1e-10:
            onb.append(Y / nrm)

    def project_onto(M, basis):
        P = sum(hs_inner(Q, M) * Q for Q in basis)
        return P

    closure_res = 0.0
    nonabelian = 0.0
    for a in range(k):
        for b in range(a + 1, k):
            C = commutator(su3[a], su3[b])
            nonabelian = max(nonabelian, fnorm(C))
            Cproj = project_onto(C, onb)
            closure_res = max(closure_res, fnorm(C - Cproj))
    print(f"\n[G] su(3) Lie-algebra closure:")
    print(f"      max ||[X_a,X_b] - proj_su3([X_a,X_b])|| = {closure_res:.2e}",
          "PASS (closes)" if closure_res < 1e-8 else "FAIL (does NOT close)")
    print(f"      max ||[X_a,X_b]||                       = {nonabelian:.4f}",
          "(nonabelian: genuine su(3))" if nonabelian > 1e-6 else "(ABELIAN -- not su(3)!)")

    # --- centralizer of su(3) in the 15-dim bivector algebra ---
    # Y = sum_a d_a B_a ;  require [Y, X_k] = 0 for all su(3) basis X_k.
    # Build linear map L: R^15 -> C^{(k*64)} , d -> stacked vec([B_a,X_k]).
    rows = []
    for X in su3:
        for B in bivs:
            rows.append(commutator(B, X).reshape(-1))   # 64-vector per (B,X)
    # We need, for fixed d: sum_a d_a [B_a, X_k] = 0 for each k.
    # Assemble big matrix G (m x 15): block per X_k stacks [B_a,X_k] columns.
    blocks = []
    for X in su3:
        Mk = np.column_stack([commutator(B, X).reshape(-1) for B in bivs])  # (64 x 15)
        blocks.append(Mk)
    G = np.vstack(blocks)                                  # (64k x 15) complex
    cent_basis, cent_rank = real_nullspace(G, tol=1e-9)
    cdim = cent_basis.shape[1]
    print(f"\n[H] Centralizer of su(3) in span(15 bivectors):")
    print(f"      dim = {cdim}  (claim: exactly 1)",
          "PASS" if cdim == 1 else "FAIL")

    # --- verify the centralizer is span{H} ---
    if cdim >= 1:
        # express H in bivector coefficients: H = B12+B34+B56
        hvec = np.zeros(15)
        hvec[idx["B12"]] = 1.0
        hvec[idx["B34"]] = 1.0
        hvec[idx["B56"]] = 1.0
        hvec = hvec / np.linalg.norm(hvec)
        # project H-direction onto centralizer subspace; check it lies in it
        # and that centralizer (if 1-dim) is parallel to H.
        Pc = cent_basis @ cent_basis.T          # projector onto centralizer (real)
        residual = np.linalg.norm(hvec - Pc @ hvec)
        print(f"      || H_dir - proj_centralizer(H_dir) || = {residual:.2e}",
              "PASS (H in centralizer)" if residual < 1e-8 else "FAIL")
        if cdim == 1:
            v = cent_basis[:, 0]
            v = v / np.linalg.norm(v)
            align = abs(float(v @ hvec))
            print(f"      |<centralizer_dir, H_dir>| = {align:.10f}  (1.0 => centralizer == span{{H}})",
                  "PASS" if abs(align - 1.0) < 1e-8 else "FAIL")
            # also report the materialized centralizer element vs H
            Y = sum(v[a] * bivs[a] for a in range(15))
            # scale Y to match H on xi-independent norm
            print(f"      Frobenius ||centralizer_elt|| = {fnorm(Y):.4f},  ||H|| = {fnorm(H):.4f}")

    # --- demonstration: a sample cross-plane bivector moves color ---
    sample = bivs[idx["B13"]]   # e1 e3 : connects color planes 1 and 2
    move = max(fnorm(commutator(sample, X)) for X in su3) if su3 else 0.0
    print(f"\n[I] Demonstration: sample cross-plane bivector B13")
    print(f"      max ||[B13, X_su3]|| = {move:.4f}",
          "(B13 MOVES color, as expected)" if move > 1e-6 else "(commutes -- unexpected)")

    print("\n" + "=" * 76)
    print("  SUMMARY")
    print("=" * 76)
    verdict_dim = (k == 8)
    verdict_close = (closure_res < 1e-8 and nonabelian > 1e-6)
    verdict_cent = (cdim == 1)
    print(f"  su(3) stabilizer is 8-dimensional ............ {'PASS' if verdict_dim else 'FAIL'}")
    print(f"  su(3) closes as a genuine Lie algebra ........ {'PASS' if verdict_close else 'FAIL'}")
    print(f"  centralizer in so(6) is 1-dimensional ........ {'PASS' if verdict_cent else 'FAIL'}")
    if cdim == 1:
        print(f"  centralizer == span{{H}} = span{{(3/2)Q_(B-L)}} . see [H] alignment above")
    print()
    if verdict_dim and verdict_close and verdict_cent:
        print("  >>> Premise (1) CONFIRMED in this representation.")
        print("  >>> Only color-neutral bivector is H = (3/2)Q_(B-L).")
        print("  >>> A color-neutral photon helicity generator would carry B-L charge,")
        print("  >>> which is forbidden. The transverse modes are color-charged.")
        print("  >>> Polarization wall == Coleman-Mandula double-booking wall.")
    else:
        print("  >>> One or more checks FAILED -- premise (1) NOT confirmed as stated.")
        print("  >>> Investigate before relying on the reductio.")
    print("=" * 76)


if __name__ == "__main__":
    report_fh, report_path = _open_report()
    old_stdout = sys.stdout
    sys.stdout = _Tee(old_stdout, report_fh)
    try:
        main()
    finally:
        sys.stdout = old_stdout
        report_fh.close()
        print(f"\n[report written to: {report_path}]")
