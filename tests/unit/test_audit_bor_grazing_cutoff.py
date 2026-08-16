"""AUDIT_BOR_PROPAGATING_CUTOFF_ENERGY_2026_07_13 regression gates.

Commit fca4665's dimensionless propagating-mode classifier chose the
real-axis cutoff ``q/k0 > 0.05`` (an ANGULAR cutoff: theta < 88 deg in
n=1.41) to preserve bit-identity with the old absolute threshold at the
validated k0=2.0 scale.  That cutoff silently dropped genuinely propagating
near-grazing orders, biasing per-order R/T low and leaking energy (2.28e-2 on
the lossless ring-grating reproducer here; the dropped mode carried exactly
that power).  Fixed by flooring the real-axis test at the q ~ 0 degenerate
point only (``q/k0 > 1e-6``) in all THREE classifier twins
(``bor_stack.solve``'s ``prop()``, ``bor_solve._physical_propagating``,
``_jax_bor._mask``).

The floor is compatible with the flux normalizer's field-norm fallback
(``|P| <= 1e-10 * fnrm``): the modal flux ratio scales as ``P/fnrm = q/k0``
for the limiting polarization family (verified empirically), so every kept
mode sits >= 4 decades above the fallback -- kept implies flux-normalized and
``|S|^2`` stays a true power fraction.

Gates (audit section 5): the reproducer at three unit scales; the
near-grazing band is populated; the fundamental-mode R pin (the lossless-trap
guard -- a fix that "closes" energy by renormalizing instead of restoring the
missing order fails it); JAX twin parity; the ``bor_solve`` twin.

Values were deliberately updated on 2026-07-26 for the audit W6-B1 staggered
wall-anchor flip -- see the DELIBERATE UPDATE block below the imports for the
old/new numbers, why the near-grazing gates moved to the ``48 + h/2`` cavity
(bit-identical to the pre-flip run) to keep their teeth, and the new gate 3b
that pins that discriminating power explicitly.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pytest

from lumenairy.elements.bor import BORStack

pytestmark = pytest.mark.slow      # dense modal eigensolves (N=256 half-spaces)

_LAM = 1.0e-6      # 1 um design wavelength (physical scale)

# --------------------------------------------------------------------------- #
#  DELIBERATE UPDATE 2026-07-26 -- audit W6-B1 staggered wall-anchor flip
#
#  ``coupled_radial_eigensolver.STAGGERED_WALL_ANCHOR`` flipped from the legacy
#  ``'ghost'`` (h = Rbig/N) to the corrected ``'rbig'`` (h = Rbig/(N + 0.5)).
#  The staggered outer stencil zeroes the tangential field at the GHOST NODE,
#  radius ``(N + 0.5) h``, so the legacy spacing put the PEC wall at
#  ``Rbig + h/2`` -- the discretized cavity was half a cell LARGER than the
#  requested ``Rbig``, and the box spectrum was FIRST order (p = 0.99 vs 1.99).
#
#  What that does to THIS file's reproducer (nominal Rbig = 48 um, N = 256,
#  so h/2 = 0.09375 um), MEASURED, identical at all three unit scales:
#      incident propagating orders   319       -> 318
#      fundamental-mode R            0.146135  -> 0.142290
#      min q/k0                      0.049293  -> 0.051165
#      full-set energy closure       ~1.2e-11  -> ~1.2e-11   (unchanged)
#  The order that disappears is the near-grazing one at q/k0 = 0.049293: it
#  belonged to the oversized cavity, not to the 48 um cavity that was asked
#  for.  Gates 1, 3 and 6 below carry the new values.
#
#  KEEPING THE TEETH.  Gates 2 and 3 exist to catch a re-introduced angular
#  cutoff, and both need a near-grazing order BELOW q/k0 = 0.05 to bite.  At
#  the corrected anchor the nominal 48 um cavity has none (min q/k0 = 0.051165)
#  and the truncation delta on the fundamental R collapses from 1.022e-3 to
#  0.0 -- i.e. a retune-in-place would silently defang them.  So those two
#  gates now request ``Rbig = 48.09375 um`` = 48 + h/2, which is EXACTLY the
#  physical cavity the legacy anchor was simulating: verified BIT-IDENTICAL
#  (r_node, r_face, h, all four stencils, and all 319 R/T/q/energy values) to
#  the pre-flip ``Rbig = 48 um`` run, so every number the original audit
#  published still reproduces -- 319 orders, min q/k0 = 0.049293, fundamental
#  R = 0.146135, the shipped-bug value 0.145113, and the 2.28e-2 energy leak.
# --------------------------------------------------------------------------- #

#: nominal cell radius (um): what a caller asks for
_RBIG_UM = 48.0
#: the cavity the pre-flip anchor actually simulated = 48 + h/2 at N = 256.
#: Requesting it at the corrected anchor is bit-identical to the pre-flip run,
#: and it is the geometry that keeps the near-grazing gates' teeth.
_RBIG_UM_AUDIT_CAVITY = 48.09375


def _reproducer(scale, ring_index=2.45 + 0j, N=256, rbig_um=_RBIG_UM):
    """The audit's lossless concentric ring grating between index-matched
    half-spaces, m=1, at unit scale ``scale`` (1 = metres)."""
    k0 = 2.0 * np.pi / (_LAM * scale)
    s = BORStack(Rbig=rbig_um * 1e-6 * scale, m=1, N=N,
                 n_superstrate=1.41 + 0j, n_substrate=1.41 + 0j)
    s.add_layer(0.5e-6 * scale, rings=(3.0e-6 * scale, 0.5, ring_index,
                                       1.41 + 0j))
    s.set_source(k0=k0)
    return s.solve(), k0


def _leak_with_angular_cutoff(res, k0, cut=0.05):
    """The audit's OWN leak metric: max|R+T-1| over incident modes once the
    classifier floor is raised to ``cut`` -- i.e. exactly what the pre-fix
    ``q/k0 > 0.05`` classifier produced.  Validated against the audit's
    published 2.28e-2 (this metric returns 2.2756e-02 on the audit cavity)."""
    qn = np.asarray(res["q"], float) / k0
    inc, out = res["inc"], res["out"]
    S11, S21 = res["S"][0], res["S"][2]
    keep = qn > cut
    ii, oo = inc[keep], out[keep]
    worst = 0.0
    for j in ii:
        rc = float(np.sum(np.abs(S11[np.ix_(ii, [j])]) ** 2))
        tc = float(np.sum(np.abs(S21[np.ix_(oo, [j])]) ** 2))
        worst = max(worst, abs(rc + tc - 1.0))
    return worst, int(np.sum(~keep))


@pytest.mark.parametrize("scale", [1.0, 1e6, 1e9], ids=["m", "um", "nm"])
def test_bor_grazing_reproducer_closes_at_all_scales(scale):
    """Gate 1: energy closure to 1e-9 with the FULL incident set at m / um / nm
    unit scales.  Pre-fix (the cutoff bug): one order short, max|R+T-1| =
    2.28e-2.

    DELIBERATE UPDATE (W6-B1 anchor flip): the count was pinned at **319** for
    the legacy ``Rbig + h/2`` cavity; the corrected anchor gives the 48 um
    cavity that was actually requested, which has **318** propagating orders
    (measured identically at all three scales).  Closure is unaffected:
    max|R+T-1| = 8.2e-12 / 6.8e-12 / 5.0e-12 at m / um / nm."""
    res, _k0 = _reproducer(scale)
    e = np.asarray(res["energy"], float)
    assert e.size == 318, f"incident-mode count {e.size} != 318"
    assert float(np.max(np.abs(e - 1.0))) < 1e-9


def test_bor_grazing_band_is_kept():
    """Gate 2: the near-grazing band the bug silenced (q/k0 in (1e-3, 0.05)
    with essentially-zero imag) is populated in the kept incident set, and the
    historical ``q/k0 > 0.05`` angular cutoff MEASURABLY leaks energy there.

    DELIBERATE UPDATE (W6-B1 anchor flip): this gate needs a mode below 0.05 to
    have any teeth.  The nominal 48 um cavity had one at q/k0 = 0.049293 only
    because the legacy anchor was really simulating 48 + h/2; at the corrected
    anchor its min q/k0 is 0.051165 and the gate would pass vacuously.  So it
    now requests ``_RBIG_UM_AUDIT_CAVITY`` -- bit-identical to the pre-flip run
    -- which restores q/k0 = 0.049293 exactly, and the leak assertion below
    (new) turns "the band is populated" into "the old cutoff really did lose
    2.28e-2 of energy here"."""
    res, k0 = _reproducer(1e6, rbig_um=_RBIG_UM_AUDIT_CAVITY)
    qn = np.asarray(res["q"], float) / k0
    band = (qn > 1e-3) & (qn < 0.05)
    assert np.any(band), "no kept incident mode in the near-grazing band"
    assert abs(qn.min() - 0.049293) < 1e-5, f"min q/k0 = {qn.min():.6f}"
    # with the full set the cascade closes; with the historical angular cutoff
    # it leaks the audit's published 2.28e-2 (metric reproduces 2.2756e-02)
    assert np.max(np.abs(np.asarray(res["energy"], float) - 1.0)) < 1e-9
    leak, ndropped = _leak_with_angular_cutoff(res, k0, 0.05)
    assert ndropped == 1, f"{ndropped} orders below the 0.05 cutoff (expect 1)"
    assert leak > 1e-2, f"angular-cutoff leak {leak:.4e} -- gate has no teeth"
    assert abs(leak - 2.2756e-2) < 5e-4, f"leak {leak:.6e} != audit 2.28e-2"


def test_bor_fundamental_mode_reflectance_pin():
    """Gate 3 (the lossless-trap guard), retuned value + re-derived INTENT.

    DELIBERATE UPDATE (W6-B1 anchor flip): the pinned number was **0.146135**
    for the legacy ``Rbig + h/2`` cavity; the 48 um cavity actually requested
    gives **0.142290** (measured, all three unit scales).

    What this gate GUARDS is not the number but the mechanism: R must be the
    DIRECT sum of ``|S11|^2`` over the COMPLETE propagating set -- a fix that
    "closes" energy by renormalizing a truncated set must fail.  The number
    alone can no longer carry that (see the teeth gate below), so the property
    is now asserted directly."""
    res, k0 = _reproducer(1e6)
    q = np.asarray(res["q"], float)
    R = np.asarray(res["R"], float)
    T = np.asarray(res["T"], float)
    inc, S11 = res["inc"], res["S"][0]
    j = int(np.argmax(q))              # near-axis fundamental = largest q
    assert abs(R[j] - 0.142290) < 1e-4, f"fundamental R = {R[j]:.6f}"
    assert abs(float(np.asarray(res["energy"])[j]) - 1.0) < 1e-9
    # (i) NO renormalization anywhere: R is literally the sum over ``inc``,
    #     recomputed through the SAME elementwise kernel the library uses -- so
    #     this is an EXACT identity, not a tolerance.
    #
    # 2026-08-15 (docs/audits/FIX_RUNNER_PINS_2_2026_08_15.md, D5): the
    # reference here used to be a VECTORIZED recomputation
    # (``np.abs(S11[np.ix_(inc, [jj])]) ** 2`` on a strided 2-D complex slice)
    # compared at ``< 1e-15``.  That is NOT the library's arithmetic:
    # ``bor_stack.solve`` sums a Python list of SCALAR ``abs(...) ** 2``
    # (``bor_stack.py``, ``R = np.array([np.sum([abs(S11[jp, j]) ** 2 ...``).
    # So the residual being measured was numpy's SIMD complex-abs loop against
    # the scalar path -- of order 1 ULP per element over 318 elements, a
    # quantity that moves with numpy version, compiler and SIMD width and has
    # nothing to do with the claim.  MEASURED 1.1102e-16 (Win py3.14/np2.4.4) /
    # 1.6653e-16 (WSL py3.12/np2.5.1): 9.0x / 6.0x of headroom on a 1.5x
    # cross-build spread.  Recomputing through the library's own kernel makes
    # the residual EXACTLY 0.0 on both builds (verified), so the claim is
    # stated as the identity it actually is, with no bar to widen or tighten.
    direct = np.array([np.sum([abs(S11[jp, jj]) ** 2 for jp in inc])
                       for jj in inc])
    assert np.array_equal(R, direct), "R is not the raw modal sum"
    # (i-b) ... and that identity HAS TEETH.  An exact comparison is worthless
    #     unless a wrong implementation fails it, so inject the two shapes of
    #     "closed by construction" this gate exists to reject.  Both are
    #     ENGINEERED here, not measured off a build.
    #     (a) closure IMPOSED by scaling -- the lossless trap in its purest
    #         form.  Any such scaling changes the bits, so the identity above
    #         rejects it without needing a magnitude bar at all.
    imposed = R / (R + T)
    assert not np.array_equal(R, imposed)
    #     (b) the shipped bug's own shape: truncate the near-grazing tail and
    #         renormalize the survivors back to R + T = 1.  ``q/k0 > 0.052``
    #         drops exactly the min-q mode of THIS cavity (0.051165, see the
    #         DELIBERATE UPDATE block), which is what makes the demonstration
    #         land here at all; assert that it really dropped one, so the
    #         demonstration cannot quietly become a no-op.
    out, S21 = res["out"], res["S"][2]
    keep = (q / k0) > 0.052
    assert int(np.sum(~keep)) == 1, "the injected truncation dropped no mode"
    R_cut = np.array([float(np.sum(np.abs(S11[np.ix_(inc[keep], [jj])]) ** 2))
                      for jj in inc])
    T_cut = np.array([float(np.sum(np.abs(S21[np.ix_(out[keep], [jj])]) ** 2))
                      for jj in inc])
    faked = R_cut / (R_cut + T_cut)          # "closes" energy, wrong answer
    assert np.max(np.abs(faked + (T_cut / (R_cut + T_cut)) - 1.0)) < 1e-12
    # MEASURED max|faked - R| = 5.3813e-01, BIT-IDENTICAL on both builds -- 54x
    # above the 1e-2 bar and ~12 decades above the identity's 0.0, so the
    # discriminator is not marginal in either direction: it cannot pass by
    # round-off and it cannot fail by build (it is a physical change, the loss
    # of a whole propagating order, not an arithmetic one).
    assert np.max(np.abs(faked - R)) > 1e-2
    # (ii) the set is COMPLETE against the physical criterion, and closure is
    #      therefore earned rather than imposed
    assert R.size == 318 and T.size == 318
    assert np.max(np.abs(R + T - 1.0)) < 1e-9


def test_bor_fundamental_mode_pin_has_teeth_on_the_audit_cavity():
    """Gate 3b (NEW, W6-B1): the lossless-trap guard's discriminating power,
    on the cavity the audit actually measured (bit-identical to its pre-flip
    run).  Truncating the near-grazing tail the way the shipped bug did moves
    the fundamental R from 0.146135 to 0.145113 -- delta 1.022e-3, an order of
    magnitude above the 1e-4 pin tolerance, so the pin genuinely catches it.

    (On the nominal 48 um cavity at the corrected anchor the same truncation
    moves nothing -- delta 0.0, because no order sits below 0.05 there -- which
    is precisely why this gate exists separately.)"""
    res, k0 = _reproducer(1e6, rbig_um=_RBIG_UM_AUDIT_CAVITY)
    qn = np.asarray(res["q"], float) / k0
    R = np.asarray(res["R"], float)
    inc, S11 = res["inc"], res["S"][0]
    j = int(np.argmax(qn))
    assert R.size == 319                       # the audit's own mode count
    assert abs(R[j] - 0.146135) < 1e-4, f"full-set R = {R[j]:.6f}"
    keep = qn > 0.05                           # the shipped bug's angular cut
    truncated = float(np.sum(np.abs(S11[np.ix_(inc[keep], [inc[j]])]) ** 2))
    assert abs(truncated - 0.145113) < 1e-4, f"truncated R = {truncated:.6f}"
    assert abs(truncated - R[j]) > 1e-3, (
        f"truncation delta {abs(truncated - R[j]):.3e} <= the 1e-4 pin "
        f"tolerance -- the lossless-trap guard would not bite")


def test_bor_jax_twin_parity_on_reproducer():
    """Gate 4: the differentiable twin's mask carries the same constant --
    total reflected/transmitted power must match the NumPy path (its full-2N
    masked arrays sum over the same propagating sets)."""
    pytest.importorskip("jax")
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    res_np, _ = _reproducer(1e6)
    res_jx, _ = _reproducer(1e6, ring_index=jnp.asarray(2.45 + 0j))
    r_np = float(np.sum(np.asarray(res_np["R"])))
    t_np = float(np.sum(np.asarray(res_np["T"])))
    r_jx = float(np.sum(np.asarray(res_jx["R"])))
    t_jx = float(np.sum(np.asarray(res_jx["T"])))
    assert abs(r_np - r_jx) < 1e-8 * max(1.0, abs(r_np))
    assert abs(t_np - t_jx) < 1e-8 * max(1.0, abs(t_np))


def test_bor_solve_twin_classifier_keeps_grazing_band():
    """Gate 5: ``bor_solve._physical_propagating`` carries the same fixed
    constant.  Direct classifier unit test (synthetic mode table): the
    near-grazing band the bug silenced (q/k0 in (1e-6, 0.05), clean imag,
    low reldiv) must be KEPT; the q ~ 0 degenerate point, lossy modes, and
    high-reldiv (spurious) modes must still be rejected.

    (Historical note: this gate was originally classifier-only because the
    then-NODAL ``build_layer`` basis had a separate, PRE-EXISTING blow-up on
    large cells -- max|R+T-1| ~ 1e25..1e32 for Rbig >= ~12 lambda, IDENTICAL
    under the old and new classifier constants (A/B monkeypatch), traced to
    zero-flux spurious modes orienting by the sign of noise and making the
    interface transmission block singular.  ``build_layer`` now defaults to
    the spurious-free staggered basis, and gate 6 runs the cascade-level twin
    end-to-end.)"""
    from lumenairy.elements.bor.bor_solve import _physical_propagating

    k0 = 7.3            # arbitrary scale; classifier is dimensionless in q/k0
    qn = np.array([
        0.0493,          # the audit's dropped near-grazing mode -> KEEP
        2e-3,            # deep near-grazing band -> KEEP
        5e-7,            # below the q~0 degenerate floor -> reject
        0.5 + 1e-3j,     # lossy/evanescent (imag leg) -> reject
        0.8,             # ordinary propagating -> KEEP
        0.6,             # spurious (high reldiv) -> reject
    ])
    reldiv = np.array([1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 0.9])
    L = {"q": qn * k0, "reldiv": reldiv}
    keep = _physical_propagating(L, k0)
    assert list(keep) == [True, True, False, False, True, False]


def test_bor_solve_twin_cascade_closes_on_reproducer():
    """Gate 6: the ``build_layer``/``solve`` cascade twin end-to-end on the
    audit reproducer (um scale).  On the staggered default basis this must
    reproduce the production ``BORStack`` numbers exactly: the full incident
    mode set, machine-precision closure, and the gate-3 fundamental-mode R pin
    (per-mode R parity with ``BORStack`` re-measured at exactly 0.0 -- same
    basis, same cascade, agreeing classifier sets).  Pre-fix (nodal basis) this
    configuration returned max|R+T-1| ~ 4e32.

    DELIBERATE UPDATE (W6-B1 anchor flip): 319 -> **318** incident modes and
    R 0.146135 -> **0.142290**, tracking gates 1 and 3 (measured: closure
    6.79e-12, twin-vs-BORStack per-mode |dR| = 0.0)."""
    from lumenairy.elements.bor.bor_solve import build_layer, solve

    scale = 1e6
    k0 = 2.0 * np.pi / (_LAM * scale)
    Rbig, N = 48e-6 * scale, 256
    period, duty = 3.0e-6 * scale, 0.5
    n_hi, n_lo = 2.45 + 0j, 1.41 + 0j

    def eps_bg(r):
        return np.full_like(r, n_lo ** 2, dtype=complex)

    def eps_rings(r):
        e = np.full_like(r, n_lo ** 2, dtype=complex)
        e[(r % period) < duty * period] = n_hi ** 2
        return e

    Lh = build_layer(1, Rbig, N, eps_bg, k0)
    Lg = build_layer(1, Rbig, N, eps_rings, k0, thickness=0.5e-6 * scale)
    res = solve([Lh, Lg, Lh], k0)
    e = np.asarray(res["energy"], float)
    assert e.size == 318, f"incident-mode count {e.size} != 318"
    assert float(np.max(np.abs(e - 1.0))) < 1e-9
    jf = int(np.argmax(np.real(res["q_inc"])))     # near-axis fundamental
    assert abs(res["R"][jf] - 0.142290) < 1e-4, f"fundamental R = {res['R'][jf]:.6f}"
