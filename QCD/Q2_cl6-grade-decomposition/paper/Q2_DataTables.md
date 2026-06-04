# Paper 2 — Production Data Tables

This file contains the numerical results from Block 1 (κ_c^ren determination)
and Experiment A (heavy-mass grade decomposition production run).  These
numbers are required to write §5.1, §6.5, and §6.6 of Paper 2.  They are
authoritative — do NOT recompute them; use as given.

═══════════════════════════════════════════════════════════════════════
1. BLOCK 1 — κ_c^ren DETERMINATION (PCAC, corrected)
═══════════════════════════════════════════════════════════════════════

Runs:    N=4 with N_cfg=5,10,20 (three runs); N=6 with N_cfg=20
Method:  PCAC mass zero crossing (Method A); Lagrange interpolation
         between bracketing κ values with SNR>0.8 on both endpoints
Status:  PARTIAL — 2 of 6 β values determined; sufficient for Paper 2 §6

Final values used in Paper 2:

| β    | κ_c^ren(β)        | Source                                     |
|------|-------------------|--------------------------------------------|
| 4.5  | 0.1468 ± 0.0050   | N=6, Method A (PCAC); systematic-dominated |
| 6.0  | 0.1513 ± 0.0006   | Weighted mean across runs; 0.9σ from       |
|      |                   | literature value 0.1518 (Paper 1 §5.5.1)   |
| 5.0  | PENDING           | Not determined — Paper 2 does not require  |
| 5.5  | PENDING           | Not determined — Paper 2 does not require  |
| 5.69 | PENDING           | Not determined — Paper 2 does not require  |
| 6.5  | PENDING           | Not determined — Paper 2 does not require  |

The β=4.5 value carries a ±0.005 systematic uncertainty derived from the
N=4/N=6 spread (0.127 → 0.147), reflecting genuine O(a) effects at strong
coupling that are not yet resolved.  This uncertainty must be propagated
explicitly in any chiral-regime calculation that uses this κ_c^ren.

Methodological notes for §3 / §5.1:
 - PCAC correlators C_AP(t), C_PP(t) computed via Z2 stochastic sources
 - m_PCAC(t) = [C_AP(t+1) − C_AP(t−1)] / (4·C_PP(t)) on plateau t∈[N/4, 3N/4]
 - Plateau averaged; sign-change bracketed between κ values
 - Linear interpolation between bracketing pair (the standard procedure)
 - Method B (mπ² extrapolation via cosh fit) attempted but unreliable on
   N=4 lattice — short plateau window forced fall-back to Method A only
 - Bracket selection requires SNR > 0.8 on both endpoints to filter out
   noise-dominated zero-crossings (early bug fix; see Paper 2 §5.1 notes)

═══════════════════════════════════════════════════════════════════════
2. ALGEBRAIC VERIFICATION — TABLE C.1
═══════════════════════════════════════════════════════════════════════

All six checks below must appear as Table C.1 of Paper 2 Appendix C.
Numerical environment: Python 3.11, NumPy 2.x, IEEE 754 double precision,
machine ε ≈ 2.22×10⁻¹⁶.

| Check | Identity                                  | Target    | Result    | Status |
|-------|-------------------------------------------|-----------|-----------|--------|
| C1    | Completeness:  max‖ Σ_r P_r − I₈ ‖_F      | < 10⁻¹⁴   | 0.00e+00  | ✓ PASS |
| C2    | Idempotency:   max‖ P_r² − P_r ‖_F        | < 10⁻¹⁴   | 0.00e+00  | ✓ PASS |
| C3    | Orthogonality: max‖ P_r P_s ‖_F (r≠s)     | < 10⁻¹⁴   | 0.00e+00  | ✓ PASS |
| C4    | Dimensions: Tr(P_3)=3, Tr(P_3̄)=3,         | exact     | exact     | ✓ PASS |
|       |             Tr(P_1a)=1, Tr(P_1b)=1        | integers  | integers  |        |
| C5    | SU(3) covariance: max‖ U P_r U† − P_r ‖_F | < 10⁻¹²   | 2.44e-15  | ✓ PASS |
|       | over 50 random U ∈ SU(3) and all r        |           |           |        |
| S1†   | Haar average: ⟨ρ(g)⟩ → P_1a + P_1b        | ~ N⁻¹/²   | 0.0050    | ✓ PASS |
|       | with N_samples=50000 random U ∈ SU(3)     | scaling   | (1/√N OK) |        |

† Stochastic check; see Paper 2 footnote — algebraic identity established
  by C1–C5; S1 verifies behavior under finite-sample group averaging.
  Expected error σ_max/√N ≈ 0.0026; observed 0.0050; ratio 1.9 consistent
  with √(2 ln 64) ≈ 2.9 max-over-element scaling.  1/√N scaling verified
  across N_samples = 100 to 200,000.

═══════════════════════════════════════════════════════════════════════
3. ADDITIONAL ALGEBRAIC IDENTITIES (for §6.4 / §6.7)
═══════════════════════════════════════════════════════════════════════

| Identity                              | Residual    | Status |
|---------------------------------------|-------------|--------|
| Charge conjugation: C P_3 C† = P_3̄    | 0.00e+00    | ✓      |
| Charge conjugation: C P_1a C† = P_1b  | 0.00e+00    | ✓      |
| γ5 commutativity:   [γ5, P_r] = 0     | 0.00e+00    | ✓      |
| H-grade labeling:   I₆ P_1a = −i P_1a | 0.00e+00    | ✓      |
| H-grade labeling:   I₆ P_1b = +i P_1b | 0.00e+00    | ✓      |

═══════════════════════════════════════════════════════════════════════
4. KINEMATIC LIMIT THEOREM — FREE-FIELD VERIFICATION
═══════════════════════════════════════════════════════════════════════

Configuration: N=4, U_μ = I₃ everywhere, m₀ = 0.05, single Z₂ source.

Algebraic completeness:  δ(t) = |C_π(t) − Σ_r C_π^(r)(t)| / |C_π(t)|

| t | δ(t)      |
|---|-----------|
| 0 | 0.00e+00  |
| 1 | 0.00e+00  |
| 2 | 0.00e+00  |
| 3 | 1.26e-16  |

max δ = 1.26e-16  (target < 10⁻¹⁰; exceeds by 6 orders of magnitude)

Grade fractions at t=1:

| Grade | f_r measured  | dim(r)/8 expected | residual |
|-------|---------------|-------------------|----------|
| 3     | 0.375000      | 0.375000          | 8.5e-15  |
| 3̄     | 0.375000      | 0.375000          | 8.5e-15  |
| 1a    | 0.125000      | 0.125000          | 8.4e-15  |
| 1b    | 0.125000      | 0.125000          | 8.4e-15  |
| Sum   | 1.000000      | 1.000000          | —        |

═══════════════════════════════════════════════════════════════════════
5. EXPERIMENT A — HEAVY-MASS PRODUCTION (corrected thermalization)
═══════════════════════════════════════════════════════════════════════

Lattice:                N = 6
Quark mass:             m₀ = 0.05     →  κ = 1/(2(0.05+4)) = 0.123457
Couplings:              β ∈ {4.5 (confined), 6.0 (deconfined)}
Configurations:         N_cfg = 20 per β
Thermalization:         N_therm = 1000 (β=4.5) / 800 (β=6.0)
Decorrelation:          N_decorr = 5 sweeps
Metropolis ε:           0.40 (β=4.5) / 0.55 (β=6.0)
Acceptance rate:        ~71% (β=4.5) / ~49% (β=6.0)
Source type:            24 exact point sources per config
CG tolerance:           rtol = 1e-8
CG iterations (mean):   18.9 (β=4.5), 21.9 (β=6.0)
CG failures:            0 (β=4.5), 0 (β=6.0)
Sweep algorithm:        Even-odd checkerboard Metropolis (corrected;
                        replaces the frozen-staple parallel sweep that
                        biased earlier runs by 4–10% at strong coupling)

Plaquette equilibration check (gate: deviation < 1% from Paper 1 N=6 values):

| β    | ⟨P⟩ measured        | Paper 1 N=6 ⟨P⟩ | Deviation | Gate    |
|------|---------------------|-----------------|-----------|---------|
| 4.5  | 0.34063 ± 0.00050   | 0.33816         | +0.73%    | ✓ PASS  |
| 6.0  | 0.59580 ± 0.00073   | 0.59334         | +0.41%    | ✓ PASS  |

Algebraic completeness check (gate: δ_max < 1e-10 per configuration):

  β = 4.5:  δ_max = 5.37e-16   (max over 20 configs × 6 timeslices)
  β = 6.0:  δ_max = 5.37e-16   (max over 20 configs × 6 timeslices)

═══════════════════════════════════════════════════════════════════════
6. TABLE 6.1 — GRADE FRACTIONS f_r AT t = 1
═══════════════════════════════════════════════════════════════════════

  Grade | Rep          | f_r(β=4.5)              | f_r(β=6.0)              |  Δ ≡ f_r(4.5)−f_r(6.0)  |  σ
  ──────┼──────────────┼─────────────────────────┼─────────────────────────┼─────────────────────────┼──────
  3     | Triplet      |  0.374918 ± 0.000079    |  0.374968 ± 0.000028    |  −0.000049 ± 0.000084   | 0.6
  3̄     | Anti-triplet |  0.375081 ± 0.000080    |  0.375032 ± 0.000028    |  +0.000049 ± 0.000085   | 0.6
  1a    | Singlet (−i) |  0.125082 ± 0.000079    |  0.125032 ± 0.000028    |  +0.000049 ± 0.000084   | 0.6
  1b    | Singlet (+i) |  0.124919 ± 0.000080    |  0.124968 ± 0.000028    |  −0.000049 ± 0.000085   | 0.6
  ──────┼──────────────┼─────────────────────────┼─────────────────────────┼─────────────────────────┼──────
  Sum   | Total        |  1.000000               |  1.000000               |   0.000000              | —

Free-field reference: f_r = dim(r)/8 = {0.375, 0.375, 0.125, 0.125}.
Maximum deviation from free-field at t=1: ~8.2×10⁻⁵ (β=4.5).

C-symmetry pairing observed exactly: Δf_3 = −Δf_3̄ and Δf_1a = −Δf_1b
to within statistical precision (forced by C P_r C† = P_(C·r) and Σf_r=1).

The four σ-values are equal to four decimal places because charge
conjugation and completeness collapse the four apparent observables to
ONE independent observable in the ensemble limit (see Paper 2 §6.7.4).

═══════════════════════════════════════════════════════════════════════
7. TABLE 6.1 EXTENSION — FRACTIONS AT OTHER TIMESLICES
═══════════════════════════════════════════════════════════════════════

For paper §6.6 t-dependence discussion.  Errors increase rapidly with t
because the pion correlator decays as e^(−m_π·t); at heavy mass m_π·t ≈ 3.4·t,
so by t=3 the signal has fallen by e⁻¹⁰ ≈ 5×10⁻⁵.

β = 4.5:
  t  |   f_3                   f_3̄                 f_1a                f_1b
  ───┼────────────────────────────────────────────────────────────────────────
  0  | 0.374998±0.000003   0.375002±0.000003   0.125002±0.000003   0.124998±0.000003
  1  | 0.374918±0.000079   0.375081±0.000080   0.125082±0.000079   0.124919±0.000080
  2  | 0.374872±0.000124   0.375129±0.000124   0.125128±0.000124   0.124871±0.000124
  3  | 0.374922±0.000251   0.375081±0.000251   0.125078±0.000251   0.124919±0.000251
  4  | 0.375130±0.000086   0.374872±0.000086   0.124870±0.000086   0.125128±0.000086
  5  | 0.375125±0.000054   0.374876±0.000054   0.124875±0.000054   0.125124±0.000054

β = 6.0:
  t  |   f_3                   f_3̄                 f_1a                f_1b
  ───┼────────────────────────────────────────────────────────────────────────
  0  | 0.374999±0.000001   0.375001±0.000001   0.125001±0.000001   0.124999±0.000001
  1  | 0.374968±0.000028   0.375032±0.000028   0.125032±0.000028   0.124968±0.000028
  2  | 0.375177±0.000090   0.374824±0.000091   0.124823±0.000090   0.125176±0.000091
  3  | 0.375679±0.000187   0.374346±0.000185   0.124321±0.000187   0.125654±0.000185
  4  | 0.375196±0.000090   0.374811±0.000090   0.124804±0.000090   0.125189±0.000090
  5  | 0.374953±0.000023   0.375051±0.000023   0.125047±0.000023   0.124949±0.000023

NOTE for §6.6: the apparent ~3.6σ deviation in f_3 at β=6.0, t=3 must
NOT be reported as a physical signal.  At m_π·t ≈ 10, the pion correlator
has fallen by e⁻¹⁰ ≈ 5×10⁻⁵ relative to t=0, and signal-to-noise is severely
degraded with N_cfg=20.  This is documented in the §6.7 Discussion.

═══════════════════════════════════════════════════════════════════════
8. METHODOLOGICAL CONTRIBUTION — CHECKERBOARD SWEEP CORRECTION
═══════════════════════════════════════════════════════════════════════

Initial production runs used a vectorized parallel Metropolis sweep that
computed staples once for all V=N⁴ sites simultaneously, then proposed
and accepted updates in parallel using those frozen staples.  This
violated detailed balance: when a proposal at site x was accepted, the
staples of x's neighbors changed, but those neighbors were evaluated
against pre-update staples within the same sweep.

The bias scaled with coupling strength: 5.1% plaquette deficit at β=4.5
and 2.4% at β=6.0, regardless of sweep count.

Resolution: even-odd checkerboard sweep.  For each direction μ and parity
p ∈ {0,1}: compute staples for all parity-p sites using current U
(including freshly-updated parity-(1−p) sites from the previous half-step),
then propose and accept at parity-p sites in parallel.  Within a parity
class no two sites share a same-direction same-parity staple, so detailed
balance holds exactly at each half-step.

Result: corrected sweep produces equilibrium plaquettes within 0.5–0.7%
of Paper 1 N=6 values (Table above).  This methodological detail belongs
in §3 of Paper 2 with Paper 1 §3 cited; it also retroactively suggests
that some of Paper 1's intermediate-β plaquette deviations (β=5.0 at 7%
low, β=5.5 at 5% low) may have the same origin and warrant re-checking.
