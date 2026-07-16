# Making GBD Caustic-Accurate — Literature Synthesis (2026-07-15)

Research round for the question: *how do we make the Gaussian Beamlet
Decomposition (GBD) lens propagator accurate at caustics (foci, fold/cusp)?*
Five parallel literature agents: (1) GBS root-cause + seismic, (2) frozen
Gaussian / Herman–Kluk, (3) complex rays / CSP, (4) uniform asymptotics /
catastrophe integrals, (5) phase-space beam summation + the optical-GBD
practitioner literature.

## Root cause (code-confirmed AND literature-confirmed)

**GBD has NO amplitude singularity at caustics.** Each beamlet carries a
*complex* curvature `Q` with `Im Q` positive-definite, so `det Q ≠ 0` along the
whole ray and the `sqrt(det …)` amplitude never blows up (verified directly in
`propagate_beamlets_freespace`: "each principal sqrt is continuous through a
focus"). The classical ray singularity `1/√(det Q)` is *regularized by
construction*.

The residual caustic inaccuracy is therefore a **phase / interference
reconstruction** problem, driven by three things:
1. each beamlet's paraxial (2nd-order) transverse phase is only a quadratic
   approximation of the true Airy/Pearcey caustic phase — the caustic is built
   by *interference of many regular beams*, not by any single beam;
2. sensitivity to the **free beam-width parameter ε** (can create spurious
   "pseudo-caustics"; White–Norris–Bayliss–Burridge 1987);
3. beam **discretization / density** (Klimeš 1986 error bound).

In the *optical* GBD literature the same fact appears as two named failure
modes (Ashcraft–Douglas 2021; Ashcraft et al. 2023): the **"low-pass filtering
problem"** (soft-edged beamlets can't reconstruct a hard aperture/mask edge's
high spatial frequencies) and **"decomposition loss"** (all rays vignette at a
focal-plane mask, so the beamlet field is lost there). Not a `Q` singularity.

## Three routes to caustic-accurate GBD (ranked by ROI)

### Route 1 — Hybrid GBD → angular-spectrum/Fresnel hand-off  (cheapest, practitioner-proven)
Propagate the smooth fore-optics with GBD, hand the field to an EXACT
diffraction integral (angular-spectrum / Fresnel FFT) through the caustic/focal
region, and optionally **re-decompose** onto a fresh beamlet frame afterward.
This is the coronagraph community's standard fix.
- **Ashcraft, Douglas, Kim & Riggs (2023)**, "Hybrid propagation physics …
  coronagraphic example," arXiv:2310.20026 — *the* explicit recipe: "compute
  the field before the FPM with GBD and propagate it through the remaining
  coronagraph with traditional diffraction integrals." Also the "re-decompose
  the field" statement.
- Phase-space analog (re-expand the frame on each interaction): Ghannoum–Letrou
  frame-based beam bouncing (hal-01308276); Lugara & Letrou, Radio Sci. 38(2),
  8006 (2003).
- **Lumenairy already has both engines**: `angular_spectrum_propagate` (asm.py)
  and the phase-screen `apply_real_lens`. So Route 1 = wire the existing GBD and
  angular-spectrum propagators together with a caustic-bracketing switch. This
  is the "marriage" — but the correct partner is the FFT diffraction integral,
  not Maslov (far cheaper, exact, already built). Maslov stays as a rigorous
  oracle for the rare mid-system non-planar caustic surface a single hand-off
  plane can't bracket.

### Route 2 — Cheap GBD-internal improvements (stackable, partial)
Improve GBD's own near-focus fidelity without a hand-off:
- **Beam-width parameter (ε) choice** — specify the width at the ENDPOINT /
  receiver rather than the source ("added phase stability", Nowack 2003 Eq. 12);
  optimal/broad-beam ε (Červený et al., Eqs. 16–17); structure-adaptive width
  (Weber 1988); fixed-length optimization (Klimeš 1989). Seismic-proven to
  control caustic accuracy.
- **Frame oversampling** ν < 0.4 / higher beamlet count (Melamed 2009 exact
  Gaussian-beam frame; Lugara–Letrou 2003).
- **Truncated / Collins beamlets** for hard-edge spatial frequencies
  (Worku & Gross, JOSA A 36(5):859, 2019) — fixes the low-pass problem.
- **Plane-evaluation** acceleration to afford the higher counts (Ashcraft et
  al. 2024, arXiv:2404.12454; 34× CPU / ~67,500× GPU).
Caveat (White et al. 1987): parameter tuning makes GBS *finite and better* at
caustics but does not by itself guarantee the correct diffraction-catastrophe
amplitude — it's an improvement, not a completeness proof.

### Route 3 — Frozen Gaussian Approximation / Herman–Kluk  (principled, complete, bigger build)
Replace the amplitude prefactor with the **Herman–Kluk / FGA weight**
`a = (det Z)^{1/2}`, `Z = ∂_z(Q + iP)` built from the SAME ABCD/monodromy blocks
GBD already computes; **freeze** the beamlet width and reconstruct caustics by
interference of a **dense phase-space swarm**. Caustic-free by construction AND
cures beam-spreading (i.e. also fixes the diverging-relay ENERGY leak that
started this whole thread).
- **Lu & Yang (2011)**, "Frozen Gaussian approximation for high frequency wave
  propagation," arXiv:1010.1968 — the direct wave-equation transplant: FGA
  integral (2.1), frozen phase (2.3), ray ODEs (2.4), weight ODE (2.5) with
  `a = (det Z)^{1/2}`. Convergence proof: Lu & Yang (2012), CPAM, arXiv:1012.5055.
- **Baranger et al. (2001)**, arXiv:quant-ph/0105153 — the mechanism (Van Vleck
  `1/√(m_qp)` diverges at `m_qp = ∂q_t/∂p_0 = 0` = a focus; HK's `m_qp` sits
  *additively* inside the prefactor so it never vanishes). Also the honest
  caveat: a single HK beamlet does not conserve norm — accuracy lives in the
  swarm; benign for a lens (integrable transport).
- Rigor: Swart & Rousse (2009) arXiv:0712.0752; Robert (2010).
- Cost: dense phase-space swarm `O(ε^{-d/2})` beamlets + continuous
  Maslov-branch tracking of the complex sqrt. Retrofit: keeps ray+Q transport,
  swaps amplitude bookkeeping + launch density.

### Not recommended as a retrofit — complex rays / CSP
Rigorous through caustics (Kravtsov; Chapman et al. 1999) but carries the
**Stokes-phenomenon / saddle-selection** burden (which complex saddles switch
on/off across Stokes surfaces) — problem-specific and not a clean drop-in.
A single CSP regularizes its *own* focus for free (which GBD already exploits);
general-caustic *accuracy* needs the hard saddle machinery.

### The local-patch alternative (if we keep thawed GBD): Airy/Pearcey/Bessoid grafting
Detect caustic → classify (fold→Airy, cusp→Pearcey, axial focus→Bessoid) → fit
the canonical-integral control parameters from ray data → evaluate → blend by
matched asymptotics / partition of unity. Ludwig (1966); Kravtsov–Orlov book;
Stamnes (1991) Radio Sci. 26:1323; Connor–Hobbs canonical-integral codes;
Kirk et al. (2000) Bessoid. Cheap per-pixel (1–3-parameter special-function
lookup in a thin `~k^{-2/3}` boundary layer), far-field untouched. This is a
more analytic sibling of Route 1's hand-off.

## Recommendation

1. **Do Route 1 first** (hybrid GBD → angular-spectrum hand-off). Highest ROI,
   community-proven, and lumenairy already owns both engines — this is the
   "marriage" done with the right (cheap, exact) partner. Maslov demoted to a
   rare-case oracle.
2. **Stack Route 2** cheap levers (endpoint/adaptive ε, truncated beamlets)
   opportunistically — they raise GBD's own bar and shrink the hand-off region.
3. **Consider Route 3 (FGA)** as the deeper follow-on if a hand-off-free,
   intrinsically-caustic-accurate GBD is wanted — it uniquely also closes the
   diverging-relay energy leak, with a direct, rigorous wave-equation precedent.

Full per-source detail with formulas is in the session research transcripts.
