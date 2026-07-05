<!-- Literature search deliverable (2026-07-05): analytic differential ray
tracing to replace the GBD finite-difference per-surface Jacobian. Generated
from a 3-angle web-search workflow; citations web-verified where noted. -->

# Literature Synthesis: Analytic per-surface ray-transfer Jacobian to replace 9-ray finite differences

All three angle reports converge on one fact: **your per-surface 9-ray central-difference 4x4 Jacobian `[[A,B],[C,D]]` on `(x,y,ux,uy)` is a well-known object with two independent closed-form routes** -- (a) the analytic tensor-Q / generalized-Coddington surface matrices, and (b) forward-mode autodiff of the exact intersection+Snell map you already trace. They produce the same blocks; (b) is the drop-in, (a) is the cross-check and the source of the sign/obliquity bookkeeping.

---

## 1. Ranked, de-duplicated reading list

### Tier 1 -- directly usable (equation- or code-level; port these)

1. **Kochkina, PhD thesis, Leibniz Univ. Hannover (2013)** -- "Stigmatic and astigmatic Gaussian beams... impact of beam model choice on interferometric pathlength." Full PDF: https://repo.uni-hannover.de/handle/123456789/8158
   *Take:* Eqs (5.50)-(5.65) -- the **corrected** closed-form 2x2 complex-curvature-tensor `Q` transform at any tilted/curved surface (refraction & reflection), in pure vector form (`n*?` obliquity, surface Hessian `C_s`, tangent->beam projection `K_l`). This is *the* analytic per-surface update; highest-confidence source (full text retrieved, corrects Greco-Giusfredi typos).

2. **Kochkina, Wanner, Schmelzer, Tr?bs, Heinzel, Appl. Opt. 52(24), 6030-6040 (2013).** DOI:10.1364/AO.52.006030 -- https://opg.optica.org/ao/abstract.cfm?uri=ao-52-24-6030
   *Take:* Citable journal version of the same per-surface `Q` transform + real spot-size/curvature extraction.

3. **Ashcraft & Douglas, "Open-Source Gaussian Beamlet Decomposition Tool," arXiv:2106.09162 / Proc. SPIE 11450 (2021).** https://arxiv.org/pdf/2106.09162
   *Take:* Closest public peer to your codebase. Eq.(10) tensor-ABCD bilinear law `Q2^-^1=(C+D Q1^-^1)(A+B Q1^-^1)^-^1`; Eqs (8)-(9) the 4x4 skew-ray ABCD with 2x2 blocks; Eq.(11) explicit element/distance 4x4 matrices. States outright that your 4x4 Jacobian *is* the Q-propagator.

4. **Volatier, Menduina-Fernandez, Erhard, J. Opt. Soc. Am. A 34(7), 1146-1151 (2017).** DOI:10.1364/JOSAA.34.001146 -- https://opg.optica.org/josaa/abstract.cfm?uri=josaa-34-7-1146
   *Take:* Proof that AD of the ray-trace computational graph == generalized (Stone-Forbes) differential ray tracing for **arbitrary** surfaces. The theoretical license to replace 9-ray FD with `jacfwd`.

5. **DeepLens engine, `deeplens/geometric_surface/base.py`** (X. Yang et al.). https://github.com/singer-yang/DeepLens/blob/main/deeplens/geometric_surface/base.py
   *Take:* Verbatim, autodiff-tested per-surface primitives to port -- `newtons_method()` (stop-grad iterate + one grad step) and `refract()`/`reflect()` with the exact vector-Snell form and TIR mask.

6. **Wang, Chen, Heidrich, "dO," IEEE Trans. Comput. Imaging 8, 905-916 (2022).** DOI:10.1109/TCI.2022.3212837 -- code https://github.com/vccimaging/DiffOptics
   *Take:* The implicit-function-theorem trick: solve the intersection under `no_grad`, re-attach gradient only at the converged point (~6x memory cut, exact derivative). The single most important engineering point.

7. **Greco & Giusfredi, Appl. Opt. 46(4), 513-521 (2007).** DOI:10.1364/AO.46.000513 -- https://opg.optica.org/ao/abstract.cfm?uri=ao-46-4-513
   *Take:* Original derivation of the tilted-surface general-astigmatic curvature transform (tangential/sagittal powers + oblique factors). **Use Kochkina's corrected equations, not these directly.**

8. **Campbell, "Generalized Coddington equations found via an operator method," J. Opt. Soc. Am. A 23(7), 1691-1698 (2006).** DOI:10.1364/JOSAA.23.001691 -- https://opg.optica.org/josaa/abstract.cfm?uri=josaa-23-7-1691
   *Take:* The rotated-frame matrix Coddington: `n_r R_r C_r R_r^-^1 - n_i R_i C_i R_i^-^1 = (n_r costheta_r - n_i costheta_i) C_surf`. Gives the **non-diagonal** C-block (cross-xy power) at skew/off-axis incidence -- the high-NA astigmatism your FD captures.

9. **Kneisly II, "Local Curvature of Wavefronts in an Optical System," J. Opt. Soc. Am. 54(2), 229-235 (1964).** DOI:10.1364/JOSA.54.000229 -- https://opg.optica.org/josa/abstract.cfm?uri=josa-54-2-229
   *Take:* Transfer / refraction / **reflection** / grating curvature laws in one place, oblique incidence -- lets you handle mirrors and gratings without a separate derivation.

10. **Coddington equations (scalar tangential/sagittal single surface).** Review: Optics & Photonics News, "Who Discovered Coddington's Equations?" https://www.optica-opn.org/home/articles/volume_5/issue_8/features/who_discovered_coddington_s_equations/
    *Take:* The two scalar equations and the common RHS oblique power `K = c(n'costheta' - n costheta)` -- the principal-axis special case / numeric sanity check.

11. **Stone & Forbes, "Characterization of first-order optical properties for asymmetric systems," J. Opt. Soc. Am. A 9(3), 478-489 (1992).** DOI:10.1364/JOSAA.9.000478 -- https://opg.optica.org/josaa/abstract.cfm?uri=josaa-9-3-478
    *Take:* Authoritative statement that the 4x4 derivative matrix IS the complete first-order descriptor and composes by matrix multiply. (Full equations paywalled.)

12. **Yang, Fu, Heidrich, "Curriculum learning for ab initio deep learned refractive optics," Nature Communications 15, 6572 (2024).** DOI:10.1038/s41467-024-50835-7 -- https://www.nature.com/articles/s41467-024-50835-7
    *Take:* Demonstrates the same differentiable per-surface primitive scales to full prescriptions and drives gradient-descent design (curvature/thickness/index/asphere) -- validates the downstream use.

### Tier 2 -- adaptable

13. **Arnaud & Kogelnik, "Gaussian light beams with general astigmatism," Appl. Opt. 8(8), 1687-1693 (1969).** DOI:10.1364/AO.8.001687 -- https://opg.optica.org/ao/abstract.cfm?uri=ao-8-8-1687 -- foundational complex tensor `Q` + bilinear ABCD law; the physics your blocks feed.
14. **Stone & Forbes, "Differential ray tracing in inhomogeneous media," J. Opt. Soc. Am. A 14(10), 2824-2836 (1997).** DOI:10.1364/JOSAA.14.002824 -- https://opg.optica.org/josaa/abstract.cfm?uri=josaa-14-10-2824 -- tangent-linear (variational) Runge-Kutta for the transfer/B-block in **GRIN** segments. Only needed if you add GRIN.
15. **Alda, Wang, Bernabeu, "Analytical expression for the complex radius of curvature tensor Q...," Opt. Commun. 80(5-6), 350-358 (1991).** DOI:10.1016/0030-4018(91)90421-9 -- https://ui.adsabs.harvard.edu/abs/1991OptCo..80..350A/abstract -- *(This resolves the "unverified authorship" citation in angle 1: same PII `0030401891904219`.)* Parametric `Q` forms to seed/interpret propagation.
16. **Esser et al., "Derivation of the refraction equations for higher-order aberrations of local wavefronts at oblique incidence," J. Opt. Soc. Am. A 27(2), 218-237 (2010).** DOI:10.1364/JOSAA.27.000218 -- https://opg.optica.org/josaa/abstract.cfm?uri=josaa-27-2-218 -- 1st-order term = generalized Coddington; higher orders = optional beyond-ABCD accuracy.
17. **Tovar & Casperson, "Generalized beam matrices: ... misaligned complex optical systems," J. Opt. Soc. Am. A 12(7), 1522-1533 (1995).** DOI:10.1364/JOSAA.12.001522 -- https://opg.optica.org/josaa/abstract.cfm?uri=josaa-12-7-1522 -- augmented (affine) matrices if you ever need whole-element tilt/decenter.
18. **Sun, Wang, Fu, Dun, Heidrich, "End-to-end complex lens design with differentiable ray tracing," ACM Trans. Graph. 40(4), 71 (2021).** DOI:10.1145/3450626.3459674 -- https://dl.acm.org/doi/10.1145/3450626.3459674 -- multi-surface end-to-end pattern.
19. **Yang, Fu, Peng, Heidrich, "End-to-End Hybrid Refractive-Diffractive Lens Design with Differentiable Ray-Wave Model," SIGGRAPH Asia 2024 / arXiv:2406.00834.** https://arxiv.org/abs/2406.00834 -- coherent OPL/phase accumulation (`ray.opl += n*t`) -- for the **amplitude/complex-Q side** of GBD, not the geometric Jacobian.

### Tier 3 -- background / context

20. Stone & Forbes, J. Opt. Soc. Am. A 9(1), 96-109 (1992), DOI:10.1364/JOSAA.9.000096 -- Hamiltonian/eikonal grounding; use for **symplecticity check** on the analytic Jacobian.
21. Alda, "Laser and Gaussian Beam Propagation and Transformation," Encyclopedia of Optical Eng. 999-1013 (2003) -- https://sites.unimi.it/aqm/wp-content/uploads/JAlda-2003.pdf -- single-source convention-pinning review.
22. FORMIDABLE (JEOS Rapid Publ. 2024, jeos20230041) -- reference NURBS differential-RT architecture.
23. Balasubramanian & Campbell, "Sequential differential ray tracing ... using automatic differentiation" (Semantic Scholar; venue/year **unverified**) -- GRIN AD diff-RT.
24. "Gradient-descent freeform optics ... differentiable non-sequential ray tracing," Optim. Eng. (Springer 2023), DOI:10.1007/s11081-023-09841-9 (authors unverified) -- AD non-sequential.

---

## 2. Concrete per-surface analytic recipe -> your `(x,y,ux,uy)` Jacobian

System Jacobian is the ordered block-matrix product `J = J_N * T_{N-1} * ... * T_1 * J_1`, each 4x4 as 2x2 blocks `[[A,B],[C,D]]` acting on state `(x, y, ux, uy)`.

### (a) Free-space transfer between surfaces (homogeneous)

In a local frame with the base ray along `z` over **geometric** path `d`:

```
A = I2      B = d*Pperp      C = 02      D = I2
```

`Pperp = I - t t?` projects out the base-ray direction so the differential ray stays on the base wavefront (angle 1). For paraxial/axial rays `B = d*I2`. In reduced (optical) coordinates the standard GBD form is `B = (d/n)*I2` (Ashcraft-Douglas Eq.11). **For GRIN, replace this closed form with the Stone-Forbes 1997 tangent-linear RK** (integrate the ray and its 4x4 derivative together).

### (b) Refraction at a curved surface, oblique incidence (the core)

**Vector form (Kochkina 5.65 -- recommended, sign-robust).** At the intersection you already have: surface normal `n`; incident/refracted unit directions `?_i, ?_t` (from vector Snell); surface curvature Hessian `C_s` (2x2; `=(1/R)I` for a sphere); indices `n1,n2`. Build the 2x2 tangent->beam projection matrices `K_i, K_t` (rows = chosen transverse axes `x_l,?_l` dotted into the surface tangent axes; for a meridionally-tilted surface `K ~= diag(costheta, 1)`). Then the refracted curvature tensor:

```
Q_t = (n1/n2)*(K_t?)^-^1 [ K_i? Q_i K_i  -  C_s ( n*?_i - (n2/n1) n*?_t ) ] K_t^-^1
```

Reflection is the same with `n2 -> -n1`, `?_t -> ?_r`.

**Identifying the 2x2 ABCD blocks** (thin surface, so `B=0`):

```
A = K_t^-^1 K_i                                  (anamorphic tilt re-projection; = I2 untilted)
B = 02
C = K_t^-? * C_s * ( n1 costheta_i - n2 costheta_t ) * K_i^-^1    (surface POWER tensor P)
D = (n1/n2)-scaled projection                  (= (n1/n2) I2 untilted; index folds into ux if reduced)
```

with `costheta_i = -n*?_i`, `costheta_t = n*?_t`. Plugging `J=[[A,0],[C,D]]` into the bilinear law reproduces Kochkina (5.64)/(5.65) exactly.

**Principal-axis reduction (Coddington / Campbell) -- for validation.** Because of the `K^-?(...)K^-^1` sandwich with `K=diag(costheta,1)`, the meridional (tangential) entry picks up an extra `1/cos^2theta` and the sagittal entry none, so `P` diagonalizes (in the s,t frame) to:

- sagittal power `P_s = c(n2costheta_t - n1costheta_i)` -> `n2/s' - n1/s = K`
- tangential power `P_t = P_s/(costheta_i costheta_t)` -> `n2cos^2theta_t/t' - n1cos^2theta_i/t = K`

with `K = c(n2costheta_t - n1costheta_i)` the common oblique surface power. **Off-diagonal `C` (cross-xy power) appears whenever the surface astigmatism axes are rotated w.r.t. the beam frame** -- that is exactly the high-NA off-axis astigmatism (~theta^2) your FD version captures; the Campbell rotated-operator form (`R C R^-^1`) generates it automatically.

### (c) How tensor-Q falls out

The Gaussian beamlet's complex curvature tensor propagates by the **same** blocks (Arnaud-Kogelnik; Ashcraft-Douglas Eq.10):

```
Q_out^-^1 = (C + D Q_in^-^1)(A + B Q_in^-^1)^-^1        (curvature/inverse form, GBD-standard)
Q_out   = (A Q_in + B)(C Q_in + D)^-^1            (Q form, textbook)
```

So the analytic per-surface `[[A,B],[C,D]]` fully drives GBD -- **no separate curvature integrator needed.** Real wavefront curvature uses the identical map with real `C_wf`.

---

## 3. Implementation plan for `ray_transfer_jacobian_analytic`

**Recommended primary route: forward-mode AD of the existing trace** (Volatier 2017 license; dO/DeepLens engineering). Keep the closed-form matrices of ?2 as the analytic oracle.

1. **Factor the per-surface map** `P_k: (x,y,ux,uy)_in ? (x,y,ux,uy)_out` as `intersect ? refract`, in your chosen reduced coordinates, reusing the *exact same* intersection + vector-Snell code the tracer already runs:
   - Intersection: Newton on `F(t)=sag(x+tL, y+tM) - (z+tN)=0`, `F'=sag_x*L+sag_y*M-N`.
   - Refract: `cos_i=-(n*d)`, `k=1-?^2(1-cos_i^2)`, `d'=? d+(? cos_i-?k) n`, `?=n1/n2`; reflect `d'=d-2(n*d)n`.
2. **Differentiable intersection (load-bearing):** run `maxiter-1` Newton steps under `stop_gradient`/`no_grad`, then **one** Newton step with grad enabled. That final step equals the implicit-function-theorem correction, so AD through it yields exact `dt/d(inputs)` without unrolling (dO/DeepLens). Don't backprop the whole loop.
3. **Jacobian:** `J_k = jacfwd(P_k)` over the 4 inputs (forward-mode optimal for 4 inputs). Compose `J = ?_k J_k` with free-space `[[I, L*I],[0,I]]` gaps. Reverse-mode is reserved for the *design-parameter* gradients (`c, thickness, n`).
4. **Validation gates (bidirectional -- refute successes AND prove failures):**
   - **Byte-vs-FD:** on a tilted single spherical surface + a full prescription, `J_analytic` vs the existing 9-ray central-difference primitive; agreement should be at FD-truncation level (analytic will be *better* at high NA / small curvature).
   - **Coddington scalar check:** single surface, principal frame -- confirm `C`-block eigenvalues equal `P_s, P_t` above; at normal incidence `C -> -(n2-n1)c = -Power`.
   - **Symplecticity:** `J? ? J = ?` (Stone-Forbes 1992 JOSA A 9(1)) -- will only hold in the correct (reduced/optical-momentum) coordinates.
   - **Vector-form oracle:** compare AD `J` against the Kochkina (5.65) closed form assembled from `n, ?_i, ?_t, C_s`.
   - **Q round-trip:** free-space + thin-lens known case through the bilinear law before trusting curved surfaces.

### Convention pitfalls (each silently flips a block)

- **Reduced vs geometric angle -- the big one.** Your state is `ux=L/N` (geometric direction ratio). Classic ABCD/Coddington/`Q` machinery assumes the **optical/reduced momentum `n*L`** (symplectic, det=1 across an index change). In raw `L/N` the Jacobian will *not* be symplectic and index factors sit in different blocks. Pick one, keep it inside `P_k`, and match whatever your Q-tensor propagation expects -- otherwise you get a consistent-but-wrong index scaling on `C`/`D`.
- **`Q` vs `Q^-^1`.** GBD code (Ashcraft-Douglas) propagates `Q^-^1` with `(C+D Q^-^1)(A+B Q^-^1)^-^1`; textbooks write `Q`. Pin against free-space + thin-lens first.
- **Prefer the vector form `(n*?)` over substituting angles** (Kochkina's explicit recommendation) -- avoids `costheta` sign bugs tied to `n` orientation and curvature-sign convention.
- **Tangential vs sagittal labeling:** tangential = in the plane of incidence, carries the `1/(costheta_i costheta_t)` obliquity; sagittal = perpendicular, factor 1. Swapping them is the most frequent bug.
- **B-block projection `Pperp=I-tt?`** and **geometric (not axial) path `d`** for skew rays -- omitting either is a classic high-NA error.
- **Reflection:** `n2->-n1` *and* handle the coordinate handedness flip.
- **TIR / grazing:** `k<0` -> kill ray; `?k` at `k->0` and `costheta->0` give NaN/inf gradients -- guard with `?(k+?)` and validity masks (DeepLens). Aperture-edge clipping is piecewise-constant (zero/NaN gradient) -- unrelated to the surface Jacobian but bites if a base ray sits on an aperture edge.

### Known-hard / deferred
- **GRIN/inhomogeneous** transfer is an ODE, not a Newton intersection -> needs differential-RK (Stone-Forbes 1997) or AD-through-the-integrator (Balasubramanian & Campbell, unverified). Homogeneous surfaces are fully closed-form.
- **General (non-orthogonal) astigmatism** off-axis makes `Q` and `P` full complex-symmetric 2x2 -- diagonal-only code is subtly wrong at tilted surfaces off-axis. Implement the Campbell rotated-operator / full-`K` form, not just the two scalar Coddington equations. (This is precisely the ~theta^2 off-axis astigmatism your per-surface tensor-Q GBD targets.)

### Source-confidence flags (from the reports)
- Kochkina thesis full text retrieved -- highest confidence; Greco-Giusfredi 2007 **contains typos**, use Kochkina.
- Stone-Forbes 1992 (9:478) and Esser 2010 full equations are paywalled -- structure extracted, verify exact `A`/`D` obliquity placement numerically before trusting signs.
- The angle-1 "Ru/Lin&Cai 1991 Opt. Commun." citation with unverified authorship = **Alda, Wang & Bernabeu 1991, Opt. Commun. 80(5-6) 350** (matching PII), now verified.
- Balasubramanian & Campbell and the Springer 2023 freeform paper: existence confirmed, venue/authors/equations **not** verified -- treat as background.