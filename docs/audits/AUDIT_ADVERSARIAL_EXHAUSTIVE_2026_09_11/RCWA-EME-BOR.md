# RCWA-EME-BOR audit — `lumenairy/elements/rcwa`, `elements/eme`, `elements/bor`

Repo at `D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy`,
branch `main`, v5.45.1, HEAD `a1ff1e6e`. All probe scripts and raw output under
`…/scratchpad/RCWA-EME-BOR/` (`oracle1d.py`, `tmm.py`, `p01`–`pL`).

**Environment caveat that shaped every timing number below.** This box's
numpy/scipy-openblas (0.3.31, `DYNAMIC_ARCH NO_AFFINITY`, 24 threads) is
pathologically slow at default threading: a bare `np.linalg.inv` of a 163×163
complex matrix takes **2.29 s** unpinned vs **0.0057 s** with
`OPENBLAS_NUM_THREADS=1` (400×). Every performance number here was taken with
BLAS pinned to one thread. (Import of `lumenairy` also took 2.7 s – 275 s
depending on machine load from the other auditors, so a few probes were
truncated; those are flagged.)

## Scope read

Line-by-line: `rcwa/_core.py` (4415), `rcwa/oned.py` (1809), `rcwa/twod.py`
(2207), `rcwa/stack.py` (3263, skim of the builder/plot/field sections),
`rcwa/__init__.py`; `eme/eme_2d.py`, `eme/eme_2d_vector.py` (core + branch +
G-assembly), `eme/eme_diffraction.py`, `eme/_jax_modes.py`, `eme/_branch.py`;
`bor/coupled_radial_eigensolver.py`, `bor/zcascade.py`, `bor/radial_eigensolver.py`,
`bor/sem_radial.py` (header + mesh), `bor/farfield.py`, `bor/fiber_oracle.py`,
`bor/stepindex_oracle.py`, `bor/bor_solve.py` (header/guards), `bor/_orient.py`
(guards), `bor/_jax_bor.py` / `_jax_sem.py` (masks).
Context: `CONVENTIONS.md` §7/§7.1/§11, `git log --oneline -80`,
`AUDIT_BOR_PROPAGATING_CUTOFF_ENERGY_2026_07_13.md`,
`tests/unit/test_audit_bor_grazing_cutoff.py`, `tests/unit/test_coupled_eigensolver.py`.

Oracles I wrote for this audit (independent of the library):
* `oracle1d.py` — 1-D RCWA by **direct 4N boundary matching** (not an S-matrix),
  exp(−iωt), exact binary Fourier coefficients, Li-1996 inverse rule for TM,
  derived from Maxwell from scratch.
* `tmm.py` — multilayer TMM giving complex r/t in the Fresnel s/p convention and
  the lab-tangential (x,y) convention, validated against
  `coatings.coating_reflectance` (|r|² identical to 1e-12).
* Analytic slab-waveguide TE0 characteristic equation; analytic Airy slab;
  LP-mode characteristic equation; Bessel/Bessel-derivative zeros.

---

## Findings

### **[P1] `guided_modes` silently returns an EMPTY mode list for every weakly-guiding fiber (Δn ≲ 0.01)** — `lumenairy/elements/bor/coupled_radial_eigensolver.py:514-520`

```python
qlo, qhi = np.sqrt(eps_clad) * k0, np.sqrt(eps_core) * k0
q_margin = 5e-3 * k0                 # == 1e-2 at k0 = 2.0 (bit-exact)
...
if not (qlo + q_margin < q.real < qhi - q_margin and abs(q.imag) < imag_tol):
    continue
```

The guided window is `(n_core − n_clad)·k0` wide, but the guard band is
`5e-3·k0` on **each** side — a fraction of `k0`, not of the window. The window
therefore admits nothing at all unless `n_core − n_clad > 0.01`. Standard
telecom SMF has Δn ≈ 0.005; the classic textbook V = 2.4 fiber used in this
audit's brief (n_core = 1.45, n_clad = 1.44) has Δn = 0.01 exactly, so the
admissible interval collapses to the single point 1.445·k0 and the real mode at
1.4452932·k0 is rejected.

**Evidence.** λ = 1.55 µm, n₁ = 1.45, n₂ = 1.44, V = 2.4 ⇒ a = 3.482685 µm.

| | result |
|---|---|
| exact oracle (`fiber_oracle`, HE11) | n_eff = **1.445293173134** |
| `radial_coupled_modes(m=1, Rbig=6a, N=300, staggered)` | 1 clean mode in the window, n_eff = 1.445283244268, reldiv = **9.3e-14**, err −9.93e-6 |
| `radial_coupled_modes(m=1, Rbig=6a, N=600)` | n_eff = 1.445288209192, reldiv 5.4e-13, err −4.96e-6 |
| `radial_coupled_modes(m=1, Rbig=12a, N=600)` | n_eff = 1.445288205825, err −4.97e-6 |
| **`guided_modes(1, a, Rbig, N, n₁², n₂², k0)`** | **0 modes, at every (Rbig, N) tried: 4a/6a/8a/12a × 150/300/600** |

So the eigensolver is right and the *filter* throws the answer away. The failure
mode is exactly the one the function's own W6-B2 docstring says it fixed
("silently returned `[]` while the raw spectrum held the correct bound modes") —
that fix made the margin unit-invariant but left it **contrast**-dependent.

**Why CI is green:** all four `guided_modes` call sites in the suite
(`tests/unit/test_coupled_eigensolver.py:36,46,57`,
`tests/unit/test_niche_audit_w6_bor.py:341`) use `e1 = 6.0, e2 = 2.0`, i.e.
n₁ = 2.449 / n₂ = 1.414, **Δn = 1.04** — 100× the margin. No weakly-guiding
fixture exists.

**Impact.** Any fiber / waveguide / ring-resonator study at realistic index
contrast gets "no guided modes" with no warning, and the failure is
indistinguishable from a genuinely cut-off structure.

**Fix.** Scale the band to the window, e.g.
`q_margin = 1e-3 * (qhi - qlo)` (or `max(1e-6*k0, 1e-3*(qhi-qlo))`), and raise /
warn when `qhi - qlo <= 2*q_margin` instead of returning `[]`. Add a
weakly-guiding fixture (Δn = 0.005) to `test_coupled_eigensolver.py`.

---

### **[P1] The Rayleigh-anomaly wavelength nudge is silent, changes the answer by up to 4e-4 relative, and makes the result discontinuous in λ** — `lumenairy/elements/rcwa/_core.py:1507-1530` (`_grazing_safe_wavelength`)

When any diffracted order sits exactly at cutoff, the solver silently solves a
**different problem** (`wl *= 1 + 1e-7`, up to 8 iterations) and returns that
answer with no warning and no record of the substitution.

**Evidence** — the canonical Moharam-1995 mount Λ = λ = 1 µm, n_ridge = 2.04,
d = 1 µm, duty 0.5, n_I = n_II = 1, normal incidence, n_orders = 21:

| | library | my independent oracle at the *exact* λ | Δ |
|---|---|---|---|
| TM R₀ | 0.155908839054 | 0.155845785342 | **6.305e-05** (4.0e-4 rel) |
| TE R₀ | 0.040265175260 | 0.040264629480 | 5.458e-07 |

The library's value is *bit-identical* (3.1e-15) to its own λ·(1+1e-7) value, and
`_grazing_safe_wavelength(1e-6, …)` returns `1.000000100000000e-06` — the
substitution, confirmed directly.

It is **not** the continuous limit. Library R₀(TM) at λ(1+δ):

| δ | −3e-8 | **0** | +3e-8 | +1e-7 |
|---|---|---|---|---|
| R₀ | 0.155822748627 | **0.155908839054** | 0.155880240631 | 0.155908839054 |

The value at δ = 0 sits *above both neighbours* — a one-point spike, not a
limit (the oracle's 0.1558458 lies between them, where the limit belongs).

Everywhere else my oracle and the library agree to ≤ 2.0e-13 over 24
configurations (4 cells × TE/TM × 0°/25°/55°, including a 0.22+6.71i metal), so
this is the *only* disagreement and it is entirely the nudge.

The exact-λ problem is **not** ill-posed — the direct 4N boundary match solves it
cleanly and converges: R₀(TM) at the exact Wood point = 0.156055 / 0.155846 /
0.155819 / 0.155812 / 0.155808 / 0.155807 for M = 11/21/31/41/61/81, with
closure ≤ 1.5e-13 at every truncation, tracking the just-off-point solve
(λ(1−3e-8)) to ~2e-5 at matched M. So the singularity the nudge exists to avoid
is a property of *this* S-matrix formulation (`_inv_lam` floors λ at a grazing
region mode, which makes the `Vb` mode match ill-conditioned), not of the physics
— which is a legitimate engineering reason for the nudge, but not for its silence
or its asymmetry.

Measured silent (`warnings.catch_warnings(record=True)` → empty list), and
`RCWAStack` does the same (`stack.py:2896`): R at λ = 1.0 µm is bit-identical to
λ = 1.0000001 µm and differs from λ = 0.9999999 µm in the 4th digit.

**Impact.** Λ = λ is a standard design point (and the classic benchmark mount).
Users get a 4e-4-relative wrong number, no warning, and a λ-sweep with a
non-monotone one-point spike that will confuse any optimiser differencing
through it.

**Fix.** (a) emit a `UserWarning` naming the requested and the effective
wavelength whenever the nudge fires — one line, no behaviour change; and/or
(b) return the *symmetric* average of the ±δ solves, which is the continuous
limit to O(δ²); and/or (c) surface `wl_eff` on the result so a sweep can see it.

---

### **[P2] `rcwa_jones_2d(formulation='fff_nv')` breaks the cell's own C4/C∞ symmetry by up to 8e-3, and — unlike `rcwa_efficiency_2d` — carries no separability guard** — `twod.py:407-455` (`_li_convolutions_2d_tensor`), routed at `twod.py:1773-1787`

On `rcwa_jones_2d` the `fff_nv` token means the **Li-2003 successive
`L2·L1` factorization** (CONVENTIONS §11), which factorizes x first and y
second. That order is not x↔y symmetric, so a cell that *is* (verified
`np.array_equal(cell, cell.T)`) gets Jxx ≠ Jyy at normal incidence, where
symmetry makes them identical.

Measured, period 0.5 µm, depth 0.3 µm, λ = 0.633 µm, ε 6.25 in 2.25, θ = φ = 0,
96×96 cell, |Jxx − Jyy|:

| formulation | M=4 | 6 | 8 | 10 | 12 |
|---|---|---|---|---|---|
| square (C4), `laurent` | 4.1e-15 | 1.1e-14 | 1.3e-14 | 1.4e-13 | 1.1e-13 |
| square (C4), `li` | 4.2e-15 | 2.3e-15 | 1.5e-14 | 1.6e-14 | 2.2e-14 |
| square (C4), **`fff_nv`** | **3.09e-04** | 1.42e-04 | 8.02e-05 | 5.06e-05 | **3.42e-05** |
| disk, `laurent` | 1.9e-15 | 6.0e-15 | 4.1e-14 | 1.0e-13 | 3.0e-13 |
| disk, `li` | 1.9e-15 | 2.8e-15 | 1.0e-15 | 3.0e-15 | 1.4e-13 |
| disk, **`fff_nv`** | **8.00e-03** | 5.22e-03 | 3.70e-03 | 2.70e-03 | **2.05e-03** |

It converges (≈1/M), so it is a truncation artefact of the fixed order — the
docstring admits `L2 L1` vs `L1 L2` "differ in the truncated space" — but the
magnitude is not documented and, at the truncations people actually run (M = 6–12),
the disk shows **2–5e-3 of spurious form birefringence on a cell that has none**.
For the waveplate / polarization workflows `rcwa_jones_2d` exists to serve, that
is a systematic retardance and diattenuation bias. `fff_nv`'s Jxx also lands on
top of `li`'s (−0.18684458−0.11591788j vs −0.18684495−0.11577149j at M = 12), so
the asymmetry buys no accuracy here.

Second half of the finding: `rcwa_efficiency_2d(formulation='fff_nv')` **refuses**
this same disk —
`ValueError: NON-SEPARABLE geometry (… 17% of the boundary runs diagonal …)` —
while `rcwa_jones_2d(formulation='fff_nv')` accepts it silently.

**Fix.** Symmetrize: `ehat = (L2 L1 + L1 L2)/2`. Under the transpose T,
`T·L2L1(ε)·T = L1L2(εᵀ)`, so for a transpose-symmetric cell the average is
exactly symmetric, at 2× the (cheap, scalar-pivot) factorization cost. Failing
that, document the measured asymmetry scale in the `fff_nv` docstring and mirror
`_nv_nonseparable_guard`'s diagnostic onto `rcwa_jones_2d`.

---

### **[P2] `radial_spectrum` / the BOR eigensolvers form `M⁻¹A` explicitly and run a non-symmetric `eig` on a symmetric-definite pencil** — `bor/radial_eigensolver.py:162, 168`

```python
w, vec = np.linalg.eig(np.linalg.solve(Mk, Ak))      # return_modes
w = np.sort(np.linalg.eigvals(np.linalg.solve(Mk, Ak)).real)
```

`Ak` and `Mk` are real symmetric (M is SPD, A is SPD once the axis DOF is
dropped), so this is `scipy.linalg.eigh(Ak, Mk)` — which is ~3–5× faster, needs
no explicit `M⁻¹` (better conditioning on a graded mesh), returns sorted **real**
eigenvalues, and removes the silent `.real` truncation of a complex non-symmetric
spectrum. Accuracy is fine today (I measured 3.1e-13 / 1.8e-13 / 5.4e-14 relative
against `j_{m,n}` for m = 0/1/3 Dirichlet and 7.0e-13 / 1.8e-13 for m = 1/3
Neumann at degree 8, 12 elements), so this is a cost/robustness item, not a
correctness one.

---

### **[P3] The documented "`n_orders_y = 0` reproduces `rcwa_efficiency_1d` per order to ~5e-15" holds only in the infinite-pixel limit** — `_core.py:1544-1553` (`_validate_geometry` docstring, audit M8 claim)

The 1-D core builds **exact analytic** step coefficients (`_binary_step_coeffs`)
while the 2-D core FFTs a **rasterized** cell, so the two solve slightly
different structures. Measured with a grid-exact duty 0.5, period 0.5 µm,
n = 2/1 on n_sub = 1.5, d = 0.3 µm, λ = 0.633 µm, n_orders_x = 8, `li`:

| Sx | TE max\|ΔR\| | TE max\|ΔT\| | TM max\|ΔR\| | TM max\|ΔT\| |
|---|---|---|---|---|
| 64 | 1.46e-04 | 5.16e-04 | 1.86e-05 | 1.18e-04 |
| 256 | 9.14e-06 | 3.23e-05 | 1.14e-06 | 7.56e-06 |
| 1024 | 5.71e-07 | 2.02e-06 | 7.11e-08 | 4.73e-07 |
| 4096 | 3.57e-08 | 1.26e-07 | 4.44e-09 | 2.96e-08 |

Clean O(1/Sx²), but at the *minimum* sampling `_validate_cell_sampling` allows
(Sx ≥ 4·n_orders+1 = 33) the gap is ~1e-3, not 5e-15. The public
`rcwa_efficiency_2d` docstring's PIXEL CELL CONTRACT does warn about this; the
M8 claim in `_validate_geometry` does not carry the caveat. Restate it with its
rasterization scope.

(The same measurement confirms the *rule* reduction is real: on TM the 2-D
`'li'` matches 1-D `'li'` to 8.2e-6/5.3e-5 where 2-D `'laurent'` is
3.6e-3/9.1e-3 off — Li-1997 → Li-1996 works.)

---

### **[P3] `RCWAResult.per_order_amplitudes()` copies `kz` but aliases `Ex`/`Ey`/`kx`/`ky`** — `stack.py:694-702`

`kz` is explicitly copied (`np.array(..., copy=True)`) "so the public dict keeps
its writable-array contract", and I verified the module cache is genuinely safe:
`amp['kz'] *= 2` then re-solving gives `max|kz_after − kz_before| = 0.0`. But the
other entries are handed out by reference into the result's own modal dict:
`amp['Ex'][:] = 0` then calling `per_order_amplitudes()` again **on the same
result** returns `max|Ex| = 0.0`. Copy them too, or document the contract as
read-only for those keys.

---

### **[P3] `_inplane_ops` builds the Li-1997 operators twice and discards half of each** — `twod.py:1689-1691`

```python
Cxx = _li_convolutions_2d(exx, orders, n_orders_x, n_orders_y, xp)[0]
Cyy = _li_convolutions_2d(eyy, orders, n_orders_x, n_orders_y, xp)[1]
```

`_li_convolutions_2d` computes **both** `Cxx` and `Cyy` (Sy batched inversions of
(2Mx+1)³ *plus* Sx batched inversions of (2My+1)³) on each call. For the
overwhelmingly common isotropic cell `exx is eyy`, so one call suffices; guard on
identity (or array equality) and halve the Li-operator build.

---

### **[P3] `_check_energy`'s raise bar is `1.05 × n_states`** — `_core.py:955`

A passive structure can return `ΣR+ΣT` anywhere in (1, 1.05] with no signal,
because the tight lossless tripwire (`_core.py:990`) is disarmed as soon as any
permittivity is complex. The 5% headroom exists for the documented lossy-incidence
case (+2.3% at Im(n_sup)=0.1), but when the *incidence medium* is lossless
`R+T ≤ 1` is a theorem whatever the structure does — arm a tighter one-sided bar
in that case.

---

### **[P3] EME nits** — `eme/eme_2d.py:90, 439-441`

* `np.linalg.eigh(A.real if np.isrealobj(A) else A)` — `A` is constructed as
  `np.zeros(..., dtype=complex)`, so `np.isrealobj(A)` is always `False` and the
  `.real` arm is dead code.
* `strip_x_modes` deliberately uses `np.conj(ph)` for the Bloch wrap so the
  operator is *exactly* Hermitian (comment at :83-86), but its own FD oracle
  `ref_2d_modes` uses `1/px` / `1/py` at :439,441 — the inconsistency the comment
  exists to prevent, in the twin.

---

### **[P3] `threadpoolctl` is not a declared dependency, and the default is 400× slow on a many-core Windows box**

`set_blas_threads` / `rcwa_blas_threads` / the `@_with_blas_limit` decorator that
wraps every public entry point are **inert** without `threadpoolctl` (they warn
correctly — measured). On this workstation that means a 1-D TM solve at
`n_orders=81` takes **18.2 s** instead of **0.13 s**. `threadpoolctl` is tiny and
pure-Python; putting it in `requirements.txt` (rather than only in the optional
group) would make the library's own remedy work out of the box.

---

## Performance opportunities

Timings below are single-threaded BLAS, 1-D TM metallic grating
(Ag 0.135+3.99i, Λ = 0.5 µm, d = 0.2 µm, λ = 0.6328 µm, `formulation='li'`):

| n_orders | N | solve |
|---|---|---|
| 25 | 51 | 0.0046 s |
| 50 | 101 | 0.037 s |
| 100 | 201 | 0.198 s |
| 200 | 401 | **2.196 s** |

Component costs at N = 401 on the same machine: `eig` 0.326 s, `inv` 0.143 s,
`solve` 0.019 s. So the solve costs ≈ 6.7 eigensolves. The 1-D single-layer path
executes **6 explicit inverses + 4 solves + 1 eig**:

1. `inv(Toeplitz(1/ε))` in `_binary_grating_convolutions` (oned.py:131);
2. `inv(EPS)` for the P block (oned.py:652);
3–4. one `_guarded_inverse(a+b)` per `_interface_smatrix` (×2);
5–6. two `_guarded_inverse` in the final `_redheffer_star`
   (both `A22` and `B11` are non-zero there, so the cheap zero-block branch at
   `_core.py:2748-2756` does not fire for a single-layer stack).

Concrete wins, in order of payoff:

* **Toeplitz structure.** Items 1–2 invert genuine Toeplitz matrices with a
  general LU. A Levinson–Durbin / Gohberg–Semencul solve is O(N²) instead of
  O(N³) — ~0.29 s of the 2.20 s at N = 401, and it grows relatively as N grows.
  (Superfast Toeplitz solvers are O(N log²N) but Levinson is the pragmatic step.)
* **One star fewer.** For a *single-layer* stack the final
  `interface → propagation → interface` chain can be assembled directly (Moharam
  1995b enhanced-transmittance or a two-interface closed form) and skips both
  `_guarded_inverse` calls in the general star branch — ~0.29 s at N = 401.
* **`_inplane_ops` double work** (see P3 above) — halves the 2-D `li` operator
  build.
* **`eigh` for the BOR pencils** (see P2) — 3–5× on `radial_spectrum`,
  and the same applies to `_fast_geig`'s consumers.
* **Layer-eig reuse across ANGLES is impossible and correctly not attempted**:
  the layer eigenproblem is `P(Kx,Ky) @ Q(Kx,Ky)` and `Kx` carries `kx0`, so
  every angle needs a fresh eig. The *homogeneous* half-space modes are analytic
  and already cached (`_HOMOG_CACHE`, LRU 32, key includes wl/θ/φ/truncation/
  backend — verified complete). Across **wavelengths** the same holds. What *is*
  reusable and already exploited: identical layers within one stack
  (`_layer_eig_key`, content-hashed).
* **2-D matrix size**: the layer system is `2N × 2N` with `N = (2Mx+1)(2My+1)`;
  at Mx = My = 6 that is 338. `_li_convolutions_2d` adds `Sy` inversions of
  (2Mx+1)³ + `Sx` of (2My+1)³ — negligible next to the eig, as documented. The
  even-parity fold (`_symmetric_cascade_rt`) gives the advertised ~8× at normal
  incidence and is verified bit-equivalent to 1e-14; note it is gated to
  `formulation='laurent'` only (`twod.py:1703`), so `li`/`fff_nv` users never get
  it — a real, documented Amdahl gap worth closing by folding the tensor `(P,Q)`
  the way `_tensor_PQ` already allows.
* **`eig` vs `eigs`** in the EME FD oracle: `ref_2d_modes` already has a sparse
  shift-invert path (`k`/`sigma`) — good. `strip_x_modes` builds a *dense*
  `Nx×Nx` Hermitian matrix and takes the full spectrum; the lateral cascade needs
  all of them, so dense is right there.

## Alternative algorithms / methods

* **(a) R-matrix / hybrid for very deep gratings.** The current cascade is
  Redheffer-star S-matrices with a per-layer `exp(−λ k₀ L)` propagator, which is
  already unconditionally stable — I drove a 100 µm absorbing layer
  (n = 1.5+0.02i, n_orders = 15) and got finite, sane output
  (ΣR = 0.01797, ΣT = 1.53e-5, no NaN). The remaining deep-grating pain is
  *conditioning of the mode match*, not overflow, so the R-matrix
  (Li, *JOSA A* 13, 1024 (1996), "Multilayer-coated diffraction gratings:
  differential method of Chandezon et al. revisited" / the R-matrix propagation
  algorithm) buys little here. A better-targeted upgrade is **layer merging with
  interface-order reuse** or the **impedance/admittance (Riccati) formulation**,
  which avoids forming `(a+b)⁻¹` at all and is what `_guarded_inverse` is guarding
  against.
* **(b) Normal-vector method for arbitrary 2-D shapes (Schuster et al.,
  *JOSA A* 24, 2880 (2007); Götz et al. 2008).** Already implemented
  (`_nv_field_2d` / `_nv_convolutions_2d`) and, importantly, *gated* by
  `_nv_nonseparable_guard` to the separable regime where it was validated. The
  gap is that `rcwa_jones_2d`'s `fff_nv` is a *different* algorithm (Li-2003) and
  has no equivalent guard — see the P2 above. A worthwhile extension is
  **Weiss/Granet matched coordinates in 2-D** (Weiss et al., *Opt. Express* 17,
  8051 (2009)) for curved metallic walls, which converges faster than either NV
  or plain Li on circular pillars.
* **(c) FMM with ASR / matched coordinates (Granet, *JOSA A* 16, 2510 (1999)).**
  Implemented in 1-D and **it works as advertised** — measured below. The obvious
  extension is 2-D ASR (Granet & Guizal / Vallius & Honkanen, *Opt. Express* 10,
  24 (2002)) for crossed metallic gratings, and combining ASR with the Li rule in
  the anisotropic 1-D Jones path (currently ASR is scalar-only and
  normal-incidence-only).
* **(d) BOR: exact hybrid vs LP at V ≈ 2.4 — quantified.** λ = 1.55 µm,
  n₁ = 1.45, n₂ = 1.44, V = 2.4:
  LP01 (weakly-guiding characteristic equation) n_eff = **1.445308881546**;
  exact HE11 (`fiber_oracle`, the full 4×4 hybrid determinant)
  n_eff = **1.445293173134**. **Δ = 1.571e-05, relative 1.087e-05** (Δ_index
  = 0.00687). The library correctly uses the *exact* hybrid oracle, not LP —
  the right choice, since 1e-5 in n_eff is 4 decades above the solver's own FD
  floor.
* **(e) EME with PML vs perfectly matched Bloch modes.** The lateral cascade uses
  a genuine Bloch condition (`t = exp(i ky0 Ly)`) with no PML at all, which is
  the right choice for a periodic layer. For the *radial* BOR path both exist
  (`R_pml` nodal, closed Dirichlet staggered) and the staggered/div-conforming
  basis is correctly the default. The open research item flagged in the module
  docstring — the `u = r E_φ` (Copeland–Gopalakrishnan–Oh) exact-sequence
  formulation as the fallback if high-m spurious modes appear — is the right
  next step and is already pinned.

## Code organization observations

* `_core.py` is 4415 lines and genuinely mixes five concerns (BLAS thread
  control, conditioning guards, branch-cut helpers, Fourier factorization, the
  S-matrix algebra, and the shape/geometry validators). Its `__all__` lists
  **88 names, 83 of them private** (`_core.py:4326-4415`) — every submodule
  imports privates from it, so the "public API unchanged" split has not actually
  created a module boundary, only a file split. A `_branch.py`
  (sqrt/branch cuts), `_guards.py` (conditioning + energy), `_factorize.py`
  (Toeplitz + Li rules) and `_smatrix.py` would each be self-contained.
* **The sqrt/branch-cut duplication the commit log calls out is genuinely fixed
  on the RCWA side**: `_sqrt_decay` is now the one definition, and the EME side
  has its own consolidated `eme/_branch.py` (`cut_band` / `forward_decaying_root`)
  with the same shape. But there are now **three** independent bands with the same
  `1e-8` constant and different scales — `rcwa._core._CUT_BAND_REL`,
  `eme._branch._EME_CUT_BAND_REL`, and `pmm._core._forward_branch_flip`'s — plus
  `bor._orient.orient_band_scale`. They are documented as deliberately separate
  populations; a single `lumenairy/_branchcut.py` carrying the *shape* with a
  per-caller scale would make that structural rather than conventional.
* The forward/backward mode selector exists in at least four spellings:
  `_select_forward_flux` (rcwa, flux + deep-decay override),
  `_strip_split_forward` (eme vector), `forward_decaying_root` (eme scalar +
  diffraction), and `bor._orient.forward_orient`. All four implement "flux sign,
  falling back to decay sign, with a relative band". One helper parameterised by
  the flux functional would remove three.
* `bor/coupled_radial_eigensolver.py` carries ~100 lines of measurement prose as
  a module-level `#:` comment on a single constant (`STAGGERED_WALL_ANCHOR`).
  That is excellent provenance but it belongs in `docs/audits/`; the constant
  should carry a two-line summary and a doc link.
* Docstring-vs-behaviour: the M8 "~5e-15" claim (P3 above) and the
  `_layer_eigenmodes` claim that a uniform layer "falls through to the scalar
  path" (true for `RCWAStack` iso layers, `stack.py:2622`, but
  `_layer_eigenmodes_tensor` still has no analytic uniform branch — the
  `_step_coeffs` comment at `_core.py:3222-3228` documents the consequence).

## Unverified suspicions

* **BOR modal reciprocity is untested, not violated.** `BORStack.per_mode_amplitudes`
  pins a *display* gauge ("the dominant field sample is real-positive"), which is
  not the gauge in which reciprocity reads as matrix symmetry, so my measured
  `|S11 − S11ᵀ|/max = 0.45` and `|S21 − S21ᵀ|/max = 1.62` on an index-matched
  lossless ring grating prove nothing either way. (The unitarity question that
  motivated this is now settled in the affirmative — see "Checked and found
  correct".) A proper reciprocity gate would need the flux-metric
  `S21 = D S12ᵀ D⁻¹` with `D` the modal flux normalisation, and a mode ordering
  matched between the two half-space bases; worth building, since nothing in the
  suite currently gates BOR reciprocity.
* **EME `layer_modes` vs the 2-D FD oracle on a structured cell.** At Ny = 32 on
  a 2-strip cell (ε 4/1, k0 = 8, Nx = 16) the nine EME modes in (0, 4k0²) sit
  1.2e-3 – 1.4e-1 relative from their nearest FD eigenvalue, and one EME mode
  (qz² = 25.03) has no FD neighbour within 3.4. The module documents the FD as
  the *approximation* ("the EME is the Ny→∞ limit the 2-D-FD converges to"), so
  this needs an Ny-convergence study before it means anything. I did not run one.
* **Staggered BOR convergence order on a material interface.** The guided-mode
  error went 9.93e-6 → 4.96e-6 for N = 300 → 600 (first order), not the second
  order the `STAGGERED_WALL_ANCHOR` note measures for the *box* spectrum. Likely
  the harmonic-mean inverse rule at a wall that does not land on a face; worth
  confirming and documenting, since it caps the FD floor at ~1e-5 in n_eff.
* **`formulation='li'` on an anisotropic 2-D cell** applies the inverse rule to
  the diagonal blocks only and keeps the direct rule on `exy`/`eyx`
  (`twod.py:1689-1694`). For a rotated director that is not Li's rule (the
  `fff_nv` path's Schur composite is). It is commented as intentional; I did not
  construct an oracle to bound the resulting error.

## Checked and found correct

**RCWA 1-D formulation and normalisation**
* The TE/TM eigenproblems are exactly the Lalanne–Morris / Li-1996 ones. The
  planar fast path (`oned.py:642-659`) builds
  `M_TE = Kx² − [[ε]]` and `M_TM = −(I − Kx[[ε]]⁻¹Kx)·[[1/ε]]⁻¹`; the latter is
  similar (by `[[1/ε]]⁻¹`) to Li's Hy operator `[[1/ε]]⁻¹(Kx[[ε]]⁻¹Kx − I)`. I
  re-derived both from Maxwell. `_layer_Q_matrix` (`_core.py:2136`) puts
  `EPS_normal = [[1/ε]]⁻¹` on the wall-normal `Ex` row and the Laurent `[[ε]]`
  on the tangential `Ey` row — the placement that gives the fast TM rate; the
  `P`-block `E_z` elimination uses `inv([[ε]])` (direct rule then invert, Li 1997
  Eq. 27), which is correct for a z-invariant layer.
* Efficiency normalisation (`_project_efficiency`, `_core.py:1185`) is the
  full-field Poynting form `Re(kz_out/kz_inc)(|Ex|²+|Ey|²+|Ez|²)/|E_inc|²` with
  `Ez = −(kxEx+kyEy)/kz` and `einc_sq = sec²θ` for oblique TM — algebraically
  equivalent to Moharam's `Re(kz/n²)/(kz_inc/n_I²)` TM form, and it matched my
  oracle (which uses the textbook H-field TM normalisation) to 1e-14.
* **Independent-oracle agreement: ≤ 2.0e-13** on max|ΔR| and max|ΔT| over 24
  configurations at n_orders = 21 (4 cells incl. a 0.22+6.71i metal × TE/TM ×
  0°/25°/55°) — the one exception is the exact-Rayleigh case above.
* **Energy conservation**: `|ΣR+ΣT−1| ≤ 1.4e-13` for TE/TM × 0°/30°/60° ×
  n_sub ∈ {1, 1.5} at n_orders = 31.
* **Branch cut / lossy substrate**: a lossy exit substrate keeps
  `Re(kz_flux) ≥ 0` (`_forward_flux_kz` un-conjugates the region ε) so
  transmitted power is not zeroed; evanescent orders are masked by
  `Re(kz) > 0` and carry exactly 0. The round-3 `−r` (not `conj(r)`) on-cut flip
  in `_sqrt_decay` is the correct involution — I re-derived it: under `λ → −λ`
  the propagator pair swaps and `V = QW diag(1/λ)` changes sign, giving exactly
  the `(W, −V)` backward partner, so the assembled S-matrix is invariant;
  `conj(r) = −r + 2Re(r)` adds a backward-error-sized perturbation that is *not*.
* **Rayleigh-anomaly continuity in θ** (away from the exact cut):
  `ΣR+ΣT = 1.000000000000` at every θ across the −1-order cutoff
  (θ_c = −23.578178°, δθ = ±1e-4 … ±1e-7) and R₀ moves smoothly.
* **Thick-layer stability**: 1/10/100 µm absorbing layer — all finite, no
  overflow, T decaying 0.728 → 0.140 → 1.53e-5.
* **Reciprocity** is satisfied to truncation error and converges. Worst relative
  |T_fwd − T_rev| over m = ±1, ±2 at θ = 18° (Λ = 1.3 µm, n = 2.1/1 on 1.62):
  TE 1.57e-3 (M=11) → 1.04e-4 (21) → 7.5e-6 (41) → **5.0e-7** (81);
  TM 7.4e-3 → 1.1e-3 → 1.4e-4 → **1.9e-5**. (Note the reciprocal partner of
  transmitted order *m* is order *+m* from the substrate at −θ_m, not −m.)
* **Li-1996 metallic-TM convergence signature reproduced exactly** — Ag
  (0.135+3.99i), Λ = 0.5 µm, d = 0.2 µm, duty 0.5, λ = 0.6328 µm, normal, TM:

  | N | R₀ (`li`) | R₀ (`laurent`) | A (`li`) | A (`laurent`) |
  |---|---|---|---|---|
  | 11 | 0.506619 | 0.778438 | 6.88e-2 | 1.70e-1 |
  | 43 | 0.520008 | 0.543236 | 4.37e-2 | 7.99e-2 |
  | 123 | 0.521328 | 0.526067 | 4.26e-2 | 4.92e-2 |
  | 203 | 0.521841 | 0.517855 | 4.27e-2 | 4.88e-2 |
  | 303 | 0.522258 | 0.511455 | 4.25e-2 | 5.84e-2 |
  | 403 | 0.522461 | 0.519943 | 4.23e-2 | 5.56e-2 |

  The inverse rule is monotone and settles the absorptance by N ≈ 83; the direct
  rule **oscillates** and is still 2.5e-3 off in R₀ and 1.3e-2 off in
  absorptance at N = 403. This is Li's published behaviour, and the library's
  `'auto'` correctly selects `'li'` for TM or any metallic index (`oned.py:509`).
* **ASR (Granet matched coordinates) delivers its documented benefit and more.**
  Error against an N = 603 `li` reference on that same Ag case:

  | n_orders | uniform | ASR (best η) | gain |
  |---|---|---|---|
  | 6 | 5.17e-2 | 2.40e-3 (η=0.5) | 22× |
  | 12 | 6.13e-3 | 3.94e-4 (η=0.7) | 16× |
  | 24 | 4.19e-3 | 4.39e-5 (η=0.5) | 95× |
  | 48 | 1.38e-3 | 1.15e-4 (η=0.5) | 12× |

  The docstring's "~10× lower TM error at n_orders=12" is confirmed and
  conservative, and the documented non-monotonicity is visible (24 beats 48).
  The two non-obvious ingredients the section header calls load-bearing (the
  non-multiplied chain-rule factorization and the `G⁻¹` u↔x bridge) are both
  present as described.
* **Convergence acceleration cannot manufacture a false "converged" answer.**
  `rcwa_extrapolate(method='richardson')` on the same metallic case with samples
  at N = 41/81/161 returns 0.523201 against the N = 603 reference 0.522648 —
  error 5.53e-4, **no better** than the raw N = 161 sample's 5.55e-4, and it
  overshoots. That is precisely the `.. important::` caveat the docstring
  carries; nothing claims convergence that has not converged. `stabilize=`
  correctly treats a lossless-closure `_EnergyWarning` as a failed rung
  (`oned.py:474-477`).

**RCWA S-matrix / stack**
* **Unpatterned stacks reproduce TMM in amplitude AND phase.** `RCWAStack` with
  three uniform layers (n = 2.1/1.46/2.35, d = 120/200/80 nm on n_sub = 1.5,
  λ = 633 nm) against my independent TMM: `|Δr_xx| = 7.6e-16`,
  `|Δr_yy| = 3.8e-16`, `|Δt_xx| = 1.1e-15`, `|Δt_yy| = 9.0e-16` at θ = 0° and
  40°. `rcwa_jones_1d` on a single uniform film (including a lossy one) matches
  Fresnel to ≤ 3.3e-16 in amplitude and phase at 0° and 45°. The cascade order
  (layer[0] adjacent to the superstrate, cascaded downward) and both interface
  reference planes (r at z = 0, t at z = Σd) are therefore correct.
* **Jones sign / phase convention is self-consistent across the family.**
  `rcwa_jones_1d`'s `J_xx` is the p reflection in the **lab-x-component**
  convention, so `J_xx = J_yy` exactly at normal incidence (measured 5.6e-17) —
  the physically required identity — and its phase is *identical* to
  `coatings.coating_reflectance(..., polarization='p')`'s `phase_r`
  (−3.014972485 from both). `J_yy = r_s` exactly. Transmission
  `J_xx = t_p·cosθ_t/cosθ_i` to 2.2e-16 at 45°. This satisfies the CONVENTIONS §7
  "s == te, p == tm" bridge and the `polarization.py` retarder alignment.
* **Conical Jones basis rotation covariance (CONVENTIONS §7.1).** For a C4 cell
  at θ = 20°, `J(φ=90°) = R(90°)·J(φ=0°)·R(90°)ᵀ` to **8.56e-14**. (At φ = 30°
  the residual is 2.95e-2, as it must be — the square lattice is only C4, not
  C∞, so the rotated-Jones prediction does not apply there.)
* **2-D energy conservation at conical incidence** (disk cell, 0.5 µm pitch,
  n_orders 6×6): `|ΣR+ΣT−1| ≤ 1.2e-14` for `laurent` and `li` at
  (θ,φ) = (0,0), (25,35), (50,70), both polarizations.
* **Even-parity fold** (`symmetry=True` vs `False`): R₀₀ agrees to 1e-14
  (0.06818668486148 vs …49), closure 1.4e-15 both ways — exactly the "~1e-12,
  not bit-identical" contract.
* `_nv_nonseparable_guard` fires correctly and informatively on
  `rcwa_efficiency_2d(fff_nv)` for a curved cell.
* `_layer_eig_key` is a genuine content hash (kind, formulation, shape, dtype,
  bytes, plus the inverse-rule companion cells) and `_HOMOG_CACHE` is
  LRU-bounded (32), lock-guarded, and hands values out read-only. A caller's
  in-place `amp['kz'] *= 2` did **not** poison the next solve
  (`max|Δkz| = 0.0`).
* BLAS thread state is **thread-local**: three threads concurrently calling
  `set_blas_threads(1/2/4)` each saw their own value ({1:1, 2:2, 4:4}) with the
  main thread unchanged at `None`; `rcwa_blas_threads` restores on exit; and the
  missing-`threadpoolctl` case warns loudly rather than silently no-op'ing.
* `_validate_cell_sampling` correctly refuses an under-sampled cell
  (`(32,1)` for `n_orders_x=8` → needs `(33,1)`), preventing silent aliasing.
* **No complex64 anywhere** in the three packages: `_C = np.complex128` is used
  uniformly, the BOR/EME solvers build `dtype=complex` arrays throughout, and
  **every** JAX entry point hard-refuses without `jax_enable_x64` — the shared
  `_require_jax_x64` (`rcwa/_core.py:1084`, message naming the cond ~1e13
  reason) is called by `rcwa_efficiency_1d` (`oned.py:494`),
  `RCWAStack.solve` (`stack.py:2868`), both BOR twins
  (`bor/_jax_bor.py:153`, `bor/_jax_sem.py:319`) and both EME twins
  (`eme/_jax_modes.py:209`, `:229`) — verified at every call site, as the
  **first statement** of each. So the `jnp.complex128` annotations in the traced
  bodies cannot silently degrade to complex64 under a default JAX config.
  The same three entry points also `_reject_jax_unsupported` /
  `_concrete_bloch_phase` up front (`_jax_modes.py:210-211, 230-232`), so the
  tensor-eps, sparse-shift-invert, `return_vecs`, magnetic-`mu` and traced-Bloch-
  phase cases raise a named `NotImplementedError` rather than silently
  degrading.
* **Toeplitz construction is O(N²) but fully vectorised** — `_toeplitz_1d`
  (`_core.py:2118`) is a single fancy-index gather
  `coeffs[centre + (i[:,None] − i[None,:])]`, not a Python loop, and not
  `scipy.linalg.toeplitz` (which is NumPy-only; this version runs unchanged on
  CuPy/JAX and is differentiable). `_eps_convolution_2d` does the 2-D
  block-Toeplitz-Toeplitz equivalent the same way. Correct trade-off.
* The other two caches in the partition are also bounded and locked:
  `eme/_jax_modes._FROZEN_CACHE` (LRU 8, key = (Nx, Ny, Lx, Ly, kx0, ky0) — the
  geometry-only operators, complete) and the BORStack per-`k0` modal LRU keyed on
  the layer profile fingerprint (`bor_stack.add_layer`'s `key`, which is
  `("rings", period, duty, eps_r, eps_g)` / `("eps", value)` / the callable's
  identity — content-based for the value forms, identity-based for a callable,
  which is conservative in the right direction).

**EME**
* **`strip_x_modes` reproduces the analytic slab TE0** (n₁ = 1.50 core of width
  0.8λ in n₂ = 1.45, Lx = 12λ, kx0 = 0). Analytic β = 9.249604533741:

  | Nx | 240 | 480 | 960 | 1920 | 3840 |
  |---|---|---|---|---|---|
  | err | 2.19e-4 | 5.48e-5 | 1.37e-5 | 3.43e-6 | 8.59e-7 |

  Ratio 4.00 at every doubling — clean second order, as documented. (1e-8 in
  β needs Nx ≳ 4e4 with this FD; that is the discretisation's floor, not a bug.)
* The W6 routing fix is real: `Φᴴ Φ − I` ≤ 1.7e-15 and `Im λ` **exactly 0** at
  kx0 = 0 *and* kx0 = 0.37 for a real ε (the `eigh` branch), while a lossy ε
  correctly gives `max|Im λ| = 8.67e-2` through the `eig` branch.
* **`eme_diffraction.mode_match` reproduces the analytic Airy slab to 12
  digits**, including the stability fix:

  | case | R (FD) | R (analytic) | T (FD) | T (analytic) | energy |
  |---|---|---|---|---|---|
  | n=1.5, d=0.2 | 0.004594214440 | 0.004594214440 | 0.995405785560 | 0.995405785560 | 1.000000000000 |
  | n=1.5, d=2.0 | 0.147363631506 | 0.147363631506 | 0.852636368494 | 0.852636368494 | 1.000000000000 |
  | n=1.5, d=5.3 | 0.022448180626 | 0.022448180626 | 0.977551819374 | 0.977551819374 | 1.000000000000 |
  | n=1.5+0.2i, d=4 | 0.046104917703 | 0.046104917703 | 1.17309e-7 | 1.17309e-7 | 0.046105035013 |

  So the `c⁻` reference-plane move to `z = depth` (only decaying exponentials in
  the matrix) and the complex-`qz²` lossy path both hold.
* **Vector strip modes match the YEE DISCRETE dispersion exactly.** Uniform
  ε = 4 strip, Lx = 1, Nx = 24, k0 = 8: measured forward |ky| = 9.023 and 10.083
  against `sqrt(εk0² − ((2/h)sin(kx h/2))² − qz²)` = 9.028 and 10.08 for
  |m| = 3, 2 (the continuum values 9.965, 9.904 are the O(h²) targets). The
  forward-mode count is exactly 2Nx = 48 at qz² = 0 and 9, and the band-edge
  `LinAlgError` guard is present and correct.
* **The "overlap integral with the wrong normalisation" failure mode does not
  exist here.** Both EME cascades join sections by **tangential-field
  continuity** on the modal block matrices — scalar: `W = Φ`,
  `V = Φ diag(i ky)` (`eme_2d._wv`); vector: `W = [Ex; Ez]`, `V = [Hx; Hz]`
  (`strip_vector_modes`) — through the same `_interface`/`_star` algebra as
  RCWA. There is no `⟨ψ_a, ψ_b⟩` overlap matrix anywhere, so there is no
  `E·E*` vs `Re(E×H*)` choice to get wrong; the power bookkeeping enters only
  in `mode_match`'s `kz.real/kz0.real·|amp|²` flux weights, which are correct
  (verified against the analytic Airy slab above). Similarly, my partition
  contains **no longitudinal taper EME** — `eme_2d*` is a *lateral* cascade
  that computes a periodic layer's 2-D Bloch modes, and `eme_diffraction`
  matches those modes to plane waves in one step — so the "energy through a
  taper" probe has no target here.
* The vector `[W; −V]` backward mode is proven exact only for the
  block-anti-diagonal generator, and `_strips_have_eyz` / `_strip_modes_at`
  correctly **gate** the layer finder on `eyz`/`ezy` rather than returning wrong
  modes. `strip_vector_modes`'s `bc_ok` check detects the break and falls back to
  the full 4Nx `eig`.
* `_jax_modes` freezes only the eps-*independent* operators and caches them
  LRU-8 keyed on (Nx,Ny,Lx,Ly,kx0,ky0) — correct, and the tensor/sparse/`return_vecs`
  paths raise rather than silently degrading.

**BOR**
* **Both oracles audited and correct.** I re-derived the cylindrical transverse
  relations `E_φ = (i/g²)[(imq/r)E_z − k₀ ∂h_z/∂r]`,
  `h_φ = (i/g²)[(imq/r)h_z + k₀ε ∂E_z/∂r]` from `curl E = i k₀ h`,
  `curl h = −i k₀ ε E` and they match both oracles' docstrings and matrix rows
  term by term. `fiber_oracle`'s 4×4 determinant reduces algebraically to the
  textbook exact hybrid equation
  `[F_J+F_K][ε₁F_J + ε₂F_K] = (mq/k₀)²(1/U²+1/W²)²` with
  `F_J = J'_m/(U J_m)`, `F_K = K'_m/(W K_m)` — i.e. it is the *exact* HE/EH
  relation, not the LP approximation. `stepindex_oracle`'s PEC rows
  (`E_z(R) = 0`, `E_φ(R) = 0`) are the right 2 conditions. At V = 2.4 the fiber
  oracle returns exactly one m = 1 mode and **none** for m = 0 or m = 2 —
  correct single-mode behaviour below V = 2.405.
* **`radial_spectrum` (M1 SEM) vs Bessel zeros**: max relative error 3.11e-13
  (m=0 Dirichlet), 1.78e-13 (m=1 D), 5.42e-14 (m=3 D), 7.04e-13 (m=1 Neumann),
  1.83e-13 (m=3 N) at degree 8 / 12 elements — the documented ~1e-13. (A "1.0"
  for m=0 Neumann in my run is a probe artefact: the SEM spectrum contains the
  γ = 0 constant mode that `scipy.special.jnp_zeros` excludes.)
* **`AUDIT_BOR_PROPAGATING_CUTOFF_ENERGY_2026_07_13` re-measured at HEAD.** The
  doc's own reproducer (Rbig = 48 µm, N = 256, m = 1, n = 1.41, λ = 1 µm, one
  0.5 µm ring layer) gives **318 incident channels**, `max|R+T−1| = 6.786e-12`,
  fundamental (argmax Re q) **R = 0.142290**, **min q/k0 = 0.051165**. Those
  match the `STAGGERED_WALL_ANCHOR` note's documented post-anchor-flip values
  (318 / 0.142290 / 0.0512) to every published digit; the audit doc's original
  headline (319 / 1.2216561096e-11 / 0.146135) is correctly superseded in-tree by
  `tests/unit/test_audit_bor_grazing_cutoff.py`, whose "DELIBERATE UPDATE"
  comments carry the record. The classifier's real-axis floor is now `1e-6`
  (`_orient.py:72`), not the `0.05` angular cutoff the audit found, in all three
  twins (`bor_stack`, `bor_solve._physical_propagating`, `_jax_bor._mask`,
  `_jax_sem._mask` — all routed through the shared `channel_core`).
* **Unit-scale invariance is real and exact.** The same physical cell written at
  S = 1, 1e6 and 1e9 returns **bit-identical** `R_fund = 0.099828207830` and the
  same 192 channels, with closure 6.9e-12 / 5.9e-12 / 7.2e-12.
* **The staggered BOR modal basis is genuinely flux-ORTHONORMAL, and the
  propagating S-matrix is unitary.** On the same ring grating (Rbig = 12 µm,
  N = 96, m = 1, index-matched lossless, 71 propagating channels each side),
  restricting `S11`/`S21` to the solver's own `inc`/`out` index sets:

  | quantity | measured |
  |---|---|
  | library per-channel `max|R+T−1|` | 4.869e-13 |
  | restricted per-column closure | 4.872e-13 |
  | Gram `max|offdiag|` of `[S11; S21]` | **1.269e-12** |
  | **superposition** closure, 300 random unit-norm multi-channel inputs | **2.263e-13** |
  | `|UᴴU − I|` for the full 142×142 propagating S | **1.269e-12** |

  So energy is conserved for *arbitrary* excitation, not just single-mode —
  which is the real statement the per-channel `energy` array cannot make on its
  own, and which nothing in the suite currently gates. (An earlier reading of
  0.86 was my own probe bug: `res["inc"]`/`res["out"]` are **index arrays**
  from `np.where(...)[0]` (`bor_stack.py:912-913`), not boolean masks.)
* `zcascade._layer_modes_staggered` weights the z-flux with the **correct
  two-grid quadrature** (`Er·conj(hphi)·r_face·h` on faces minus
  `Ephi·conj(hr)·r_node·h` on nodes) — the audit-P3-14 half-cell error is
  genuinely avoided, and `r_face`/`wq_face` are exported so consumers cannot
  re-introduce it.
* `farfield.fourier_bessel`'s Parseval norms `N_n = R²J_{m+1}(α_n)²/2` are
  correct, the SEM non-uniform-grid path takes the native `wq` (already carrying
  `r dr`), and the Nyquist aliasing warning is present. `far_field_angles` takes
  `Re sqrt(ε)` so a lossy half-space no longer silently lexicographically
  compares complex numbers.
* `_check_wall` / `layer_modes(staggered=True)` **reject** the nodal-only
  `R_pml` and `wall='natural'` rather than ignoring them — the right call, since
  they were measured bit-identical before.
* The nodal-basis passivity refusal (`bor_solve._BOR_NODAL_SUPERUNITY_BAR = 1e-3`
  with a `1e-6` warn edge) is backed by a 132-solve two-sided census with a
  6.8-decade gap; the switch `BOR_NODAL_PASSIVITY_GUARD` restores the old
  behaviour bit for bit. This is the right shape for a guard.
