import io

p = "CHANGELOG.md"
s = io.open(p, encoding="cp1252").read()
old = """All notable changes to the core library are documented here.

## [5.43.0] — 2026-09-09"""
new = """All notable changes to the core library are documented here.

## [Unreleased]

### Added -- MAGNETIC (permeability-tensor) anisotropy for the PURE (no-floor) staggered 2-D PMM

A layer of the no-floor 2-D engine may now carry a relative PERMEABILITY --
scalar or a Granet BLOCK-FORM tensor `[[m11, m12, 0], [m21, m22, 0],
[0, 0, m33]]` -- through `pmm_jones_2d_staggered(..., mu_cell=)` and
`PMM2DStackPure.add_layer(..., mu=/mu_cell=)`.

Granet 2023's equations are ALREADY the magnetic ones; the shipped solver
implemented their `chi_t = I`, `chi33 = 1` reduction.  With
`[chi_t] = [mu_t]^-1` taken POINTWISE per cell (exact for the piecewise-constant
cells this basis is built on) and `chi33 = 1/m33`, three operators gain weights
on the SAME `2 q^2` second-order pencil: `R = C[chi_t]C` (Eq. 24 / A39, whose
C-rotation swaps the transverse indices and adds two mixed blocks),
`K_tz = C[chi_t][d2; -d1]` (Eq. 21 / A43) and the `chi33`-weighted curl-curl
`S_tt` (Eq. 20 / A42).  `[eps_t]`, `Meps33` and `K_zt` are untouched, the
pencil keeps its dimension, and the cascade, the far field and the union grid
are unchanged.  `mu_cell = None` is a DISPATCH: every operator and every result
on the nonmagnetic path is bit-identical.

THE TRAP, handled explicitly: with `chi_t = I` the code used ONE object
(`-Rmat`) both as the pencil's right-hand matrix and as the block field Gram
that recovers the Eq.-25 H partners.  With `chi_t != I` these are different
operators -- Eq. 25 carries no `chi_t` -- so the plain Gram is now retained
separately and `_region_modes` projects with it.  Collapsing the two applies
`[chi_t]^-1` to every H partner: an error invisible to the eigenvalues and to
any renormalised energy check, measured at 2.1e-01 against the analytic oracle
where the correct separation reads 1.9e-14.

Verified (`docs/audits/BUILD_PMM2D_STAGGERED_MAGNETIC_2026_09_10.md`) against
two oracles that need no external engine: the ANALYTIC Airy formula with the
wave impedance `Z = sqrt(mu/eps)` (uniform slabs, lossless and lossy, normal
and oblique, R / T / Jones to 4.3e-14 at M=8, converging 7.9e+05x from M=5),
and electromagnetic DUALITY `(eps, mu) <-> (mu, eps)` with vacuum half-spaces,
which also routes a y-uniform MAGNETIC stripe onto `pmm_jones_1d` and
`rcwa_jones_1d` per order (7.6e-06 at M=8 against their own 8.7e-07 mutual
spread) -- no 1-D diffraction engine in the library takes a permeability
directly.  Lossless closure now means Hermitian eps AND Hermitian mu (a
gyrotropic `m12 = -m21 = i b` absorbs nothing), and the closure tripwire's
predicate was extended accordingly.

Out of scope, and raising: OUT-OF-PLANE mu, mu together with an out-of-plane
eps (the first-order generator has no permeability blocks), and MAGNETIC
HALF-SPACES (`mu_superstrate` / `mu_substrate` exist only to raise -- a
magnetic half-space changes the Rayleigh flux normalisation).  A uniform
magnetic layer takes its own region eig; `_homog_geom_cache` raises on one.

Cost at (3,3) M=8: assembly 0.32-0.39 s -> 0.42-0.44 s, the region eig
unchanged within the machine's own scatter (the pencil dimension does not
move), peak working set +4.2% (142.7 -> 148.7 MiB -- exactly the two retained
`q^2` Gram blocks).

## [5.43.0] — 2026-09-09"""
assert s.count(old) == 1
io.open(p, "w", encoding="cp1252", newline="").write(s.replace(old, new))

p = "docs/PMM_ROADMAP.md"
s = io.open(p, encoding="cp1252").read()
old = """*(2-D multi-region is already supported via the `eps_cell` grid — only needs a"""
new = """**MAGNETIC (permeability): SHIPPED 2026-09-10**
(`docs/audits/BUILD_PMM2D_STAGGERED_MAGNETIC_2026_09_10.md`). The one-line
follow-on the Stage-A plan predicted, made explicit: `R = C[χ_t]C` with
`χ_t = [μ_t]⁻¹` (Eq. 24 / A39), `K_tz = C[χ_t][∂₂; −∂₁]` (Eq. 21 / A43) and
the `χ₃₃`-weighted curl-curl `S_tt` (Eq. 20 / A42), on the SAME `2q²` pencil.
`mu_cell` (scalar or block-form) on `pmm_jones_2d_staggered`, `mu` / `mu_cell`
on `PMM2DStackPure.add_layer`; `mu_cell=None` stays bit-identical. **The trap:**
with `χ_t = I` the pencil's `−R` IS the block field Gram, so one object served
both roles; with `χ_t ≠ I` the Eq.-25 H recovery (which carries no `χ_t`) must
still project with the PLAIN Gram — collapsing them reads 2.1e-01 against the
analytic oracle vs 1.9e-14 correct. Oracles: the analytic Airy formula with the
wave impedance (4.3e-14 at M=8, 7.9e+05× convergence from M=5, lossy arms
included) and electromagnetic DUALITY `(ε,μ) ↔ (μ,ε)`, which also carries the
1-D check (no 1-D diffraction engine in the library takes a permeability).
Cost: assembly +~25%, the eig unchanged, peak +4.2%. Out of scope and raising:
out-of-plane `μ`, `μ` with an out-of-plane `ε`, and magnetic HALF-SPACES.

*(2-D multi-region is already supported via the `eps_cell` grid — only needs a"""
assert s.count(old) == 1
io.open(p, "w", encoding="cp1252", newline="").write(s.replace(old, new))
print("ok")
