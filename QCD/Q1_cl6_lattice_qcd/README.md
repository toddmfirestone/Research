# Cl(6) Lattice QCD

A complete Wilson lattice QCD simulation with the Dirac operator and
SU(3) gauge field formulated inside the Clifford algebra Cl(6), with
SU(3) embedded as the Spin(6) ≅ SU(4) stabilizer in the bivector
subalgebra.

## What this is
A verified *reformulation*. Across 14 standard benchmarks (plaquette,
deconfinement transition, string tension, GOR linearity, critical
hopping parameter, chiral condensate, low-lying Dirac spectrum) the
Cl(6) formulation reproduces standard Wilson lattice QCD. The algebraic
framework is verified to machine precision.

## What this is NOT
- It does **not** derive SU(3) from Cl(6) — SU(3) is imposed by construction.
- It is **not** a faster method (24 vs 12 DOF/site overhead).
- It makes **no** claim of new physics.

## Requirements
Python 3.11.9 · NumPy 2.2.5 · SciPy ≥ 1.11

## Reproducing the results
- `code/ga_su3_sweep.py` — Gauge configuration generation via Metropolis Monte Carlo sampling
- `code/ga_su3_parallel_sweep.py` — Parallelized gauge sweeps for efficient batch generation
- `code/ga_su3_lattice_v2.py` — Gauge sector: plaquette, Wilson loops, deconfinement transition, string tension
- `code/ga_su3_dirac_v3.py` — Fermionic sector: GOR relation, critical hopping parameter, chiral condensate, Dirac spectrum
- `code/ga_su3_light_mass_sweep.py` — Quark mass sweep for GOR / GMOR slope and intercept analysis
- `code/ga_su3_light_mass_sweep_N10.py` — Extended light mass sweep at N=10 for chiral plateau analysis
- `code/ga_su3_condensate_analysis.py` — Chiral condensate extraction by multiple independent methods
- `code/ga_su3_eigenvalue.py` — Dirac eigenvalue spectrum: statistics, level spacing, chGUE universality
- `code/gor_gap_config_bootstrap.py` — GOR intercept gap (ΔA) bootstrap resampling and finite-volume extrapolation

## Generated data
- 'sweep_data'
- 'eigenvalue_results'
- 'reports'
- 'summaries'

## Paper
Draft: `paper/Q1_DRAFT_RevA.pdf` — Cl(6) Lattice QCD · Preprint v1 (June 2026) · comments welcome.

## License
MIT (see LICENSE).

## Methodology and AI assistance

This work was developed with substantial AI assistance, under a
human-verified pipeline with explicit scope control: every algebraic
claim is machine-checked, and all code and data are open (MIT). This
is stated up front by design. The work is intended to stand or fall on
the reproducible benchmarks — which is exactly how it should be judged.

---

*Repository maintained by Todd M. Firestone.*  
*Project contact: firestone.science@proton.me*
