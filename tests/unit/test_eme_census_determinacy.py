"""The 2-D vector EME mode census must not depend on the LAPACK build.

``layer_vector_modes`` returns the accepted ``qz^2`` set of a y-strip-sectioned
layer.  Until 2026-08-12 that set was decided, for near-threshold candidates, in
the last bits of the LAPACK reduction -- a library-side wrong answer, not a test
tolerance.  Two mechanisms, both inside ``_refine_accept``:

  (1) **A local minimiser on a function that is not unimodal.**  ``sigma_min``
      is a min-of-many-smooth-branches, so ONE detection cell carries ~1e-3
      wiggles at ~2e-3 spacing (31 local minima measured in the single cell
      [205.875, 206.125] of the Nx=16 reference grating).
      ``minimize_scalar(method="bounded")`` stops on whichever wiggle its
      golden/parabolic sequence reaches, and its x-tolerance is floored at
      ``sqrt(eps)|x| + xatol/3`` (~3.5e-6 at |qz^2| ~ 236, so ``xatol = 1e-7``
      buys nothing).  Measured: the genuine mode 205.9749757788 is reported at
      205.9786352762 on Windows/py3.14 and 205.9704915030 on WSL/py3.12 -- 3.7e-3
      and 4.5e-3 away -- and DROPPED on both, while the ubuntu runners keep it.

  (2) **A sqrt singularity that is not a mode.**  At a strip BAND EDGE
      (``ky_i -> 0``) the H-part ``V = (C U)/(i ky)`` diverges, so after column
      equilibration the forward/backward column pair of ``G`` becomes
      anti-parallel and ``sigma_min ~ C sqrt|qz^2 - q_edge|`` -- a real zero of
      ``sigma_min`` with no Maxwell solution behind it.  A minimiser stopping
      ``dq`` short reads ``C sqrt(dq)``, floored near 4e-4 by (1), and the
      rank-drop lands 1.09x-3.3x from ``ratio_tol``: a coin flip.  The ubuntu
      runners ACCEPT 235.8686333 on the W6 cell; both our mounts reject it.

Both are now adjudicated by physics: the STRUCTURAL bound
(``_pair_singularity_bound``) refuses a band-edge cusp on every build, and a
candidate inside ``_CENSUS_BAND`` is POLISHED to a converged zero before its
acceptance is read.  Everything outside the band keeps the pre-fix path, byte
for byte.

ORACLE built here, independent of everything under test -- the y-MONODROMY.
With ``d psi/dy = A_s(qz^2) psi`` the Bloch condition is
``det(M - t I) = 0``, ``M = expm(A_S h_S) ... expm(A_1 h_1)``,
``t = exp(i ky0 Ly)``.  It shares no machinery with the block-``G`` finder -- no
forward/backward split, no strip eigendecomposition, no equilibration -- so it
has no structural singularity at a band edge, and it is what separates a mode
from a cusp by nine decades below.  (It is usable only at small ``Nx``: the
monodromy's own dynamic range is ``exp(2 max|ky| Ly)``, which is 3e3 at Nx=8 and
1e13 at Nx=16, where every probe reads 1e-17 -- the cascade conditioning wall
that made the library use ``G`` in the first place.  The Nx=16 arm below uses
the in-tree 2-D-FD eigenvalue oracle instead.)

FAIL-BEFORE STRUCTURE (restructured 2026-08-13).  The defect being fixed is that
a near-threshold verdict was decided in the last bits, so WHETHER IT MANIFESTS
is itself a per-build fact: the first cut of this module asserted that the
pre-fix path drops the Nx=16 mode and that a 1-ULP nudge flips the W6 census,
and both readings failed on the ubuntu py3.11 verify shard -- whose LAPACK
happens to land the same candidates on the other side of the same bar.  That is
the campaign's own pattern (``FIX_CI_M1_T34_2026_08_06``: "asserts a per-build
fact as a universal one"), and it is treated the same way:

  * the FIXED path's claims are UNCONDITIONAL and anchored on the ORACLES --
    the census holds every oracle-confirmed mode, holds no band-edge cusp, and
    does not move under the whole nudge ladder.  Nothing here reads the pre-fix
    path at all;
  * the PRE-FIX demonstrations run on an ENGINEERED TIE AT THE CUT
    (:func:`_tie_at_the_cut`, :func:`_prefix_drop_cut`).  ``ratio_tol`` is the
    bar the verdict is read against; placing it inside the spread of readings
    the pre-fix path itself produces makes the flip -- and the drop -- true BY
    CONSTRUCTION on every build.  What a build is still allowed to decide is
    how WIDE that spread is, never that there is one;
  * the original live-cell demonstrations at the SHIPPED ``ratio_tol`` are kept
    and ADJUDICATED: reproduced-on-this-build or inert-on-this-build, printed
    with the reading either way, and PASSING either way.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import contextlib

import numpy as np
import pytest
from scipy.linalg import expm


@pytest.fixture(autouse=True)
def _deterministic_blas():
    """Pin BLAS to ONE thread for the whole file, as ``test_eme_2d_vector.py``
    does and for the same two reasons.  (1) These arms are eig-heavy -- each
    census is thousands of dense ``svd``/``eig`` calls on 64x64..160x160 blocks
    -- and on an UNCAPPED box the oversubscription is a three-decade wall-clock
    axis: measured 11.5 CPU-hours in 32 wall-minutes at 24 threads against 54 s
    at one.  (2) The build-emulation ladder below perturbs the minimiser by ONE
    ULP; a multi-threaded LAPACK reduction order is a much COARSER perturbation
    of the same kind, so leaving it free would confound the injector with the
    thing it emulates.  The module-level ``os.environ`` above is unreliable under
    pytest (another module may init BLAS first), so pin at RUNTIME."""
    try:
        from threadpoolctl import threadpool_limits
        cm = threadpool_limits(limits=1, user_api="blas")
    except ImportError:                        # threadpoolctl ships with numpy
        cm = contextlib.nullcontext()
    with cm:
        yield


from lumenairy.elements.eme import eme_2d_vector

PI = float(np.pi)

# --- measured bars, all two-sided ------------------------------------------ #
_MONO_MODE = 1e-14        # monodromy score at a true mode: measured <= 2.83e-18
#                           (5 decades under); at a cusp / a random qz^2
#                           >= 3.76e-10 (4 decades over)
_STRUCT_MODE = 1e-3       # sigma_min / structural bound at a mode: measured
#                           <= 1.52e-4 at the minimiser's stop and <= 3.4e-14
#                           converged; at a band edge >= 2.10e-1 (210x over)
_POLISH_AGREE = 1e-9      # relative spread of the polished zero across 17-,
#                           33- and 65-point localisations: measured 0.0
_FD_MODE = 1.0            # 2-D-FD oracle distance at a mode: measured <= 0.225
#                           on both reference cells; at a band-edge cusp
#                           >= 6.85 (30x over), the same bar ``verify=True``
#                           ships with as ``verify_tol``
_VALUE_RTOL = 1e-6        # a census entry vs its converged/oracle value: the
#                           clear-accept residual of S7.2 (Brent's own stopping
#                           point is kept where the verdict was unambiguous),
#                           measured <= 5.7e-9 relative
_CUSPS_W6 = (235.8686333682, 180.7703378418)     # the two band-edge cusps in
#                           the W6 window that some builds accept as modes
_MODES_W6 = (208.2502609719, 203.7161764512, 156.2813759062)
_CUSPS_N16 = (233.4775302159, 169.1623919091, 133.7501554302)
_RECOVERED = 205.9749757788   # the Nx=16 mode bounded Brent drops on our mounts
_PREFIX_STOP = 205.9786352762  # ... where Brent stops instead (3.66e-3 away)
_N16_CELL = (205.875, 206.125)  # the detection cell holding _RECOVERED (31 local
#                           minima of sigma_min measured inside it)
#: ``minimize_scalar(method="bounded")``'s own x-tolerance at ``_RECOVERED``,
#: ``sqrt(eps)|x| + xatol/3`` with the library's shipped ``xatol = 1e-7``: 3.1e-6.
#: This is the closest to a zero the minimiser is ENTITLED to stop, so it is the
#: universal bar on a census entry the fix leaves at Brent's stopping point
#: (S7.2 keeps it wherever the verdict was unambiguous).  Every build has this
#: floor and no build can beat it; ours land 3.1e-11 from the zero, 5 decades
#: inside it, because the candidate falls in the ambiguity band and is polished.
_BRENT_XFLOOR = float(np.sqrt(np.finfo(float).eps) * _RECOVERED + 1e-7 / 3.0)


def _grating(Nx, e_lo=1.0, e_hi=4.0, duty=0.5):
    xg = (np.arange(Nx) + 0.5) / Nx
    return np.where(xg < duty, e_hi, e_lo).astype(float)


def _cell(Nx):
    """The reference structured 2-strip cell used throughout the EME tests."""
    return [(_grating(Nx), 0.5), (np.full(Nx, 2.0), 0.5)]


_W6 = dict(Lx=1.0, Nx=8, Ly=1.0, k0=8.0, qz2_range=(150.0, 250.0), ky0=PI,
           n_scan=3)
_N16 = dict(Lx=1.0, Nx=16, Ly=1.0, k0=8.0, qz2_range=(130.0, 256.0), ky0=PI,
            n_scan=400)
#: The detection cell holding :data:`_RECOVERED`, widened to a window the finder
#: scans on its own.  Used by the engineered-tie arm of test 6 so the drop is
#: demonstrated on the ONE candidate it is about, at 1/25 the cost of the full
#: band and with no other candidate's verdict entering the reading.
_N16_NARROW = dict(Lx=1.0, Nx=16, Ly=1.0, k0=8.0, qz2_range=(205.5, 206.5),
                   ky0=PI, n_scan=9)


def _census(cell, **kw):
    return np.sort(eme_2d_vector.layer_vector_modes(_cell(cell["Nx"]),
                                                    **{**cell, **kw}))[::-1]


def _f16(q):
    """``sigma_min(G)`` on the Nx=16 reference cell -- the very function the
    finder minimises, with the finder's own band-edge guard."""
    try:
        return eme_2d_vector.dispersion_vec(_cell(16), 1.0, 16, 8.0, 0.0, q,
                                            PI, 1.0)
    except np.linalg.LinAlgError:
        return np.inf


def _absent(census, qz2, atol=1e-2):
    """``True`` iff no census entry sits within ``atol`` of ``qz2``.  Written as
    a membership test rather than a length test so a build that finds one MORE
    dip in the same window cannot turn a content claim into a counting one."""
    census = np.asarray(census, dtype=float)
    return census.size == 0 or float(np.min(np.abs(census - qz2))) > atol


def _reading(cell, qz2):
    """``(sigma_min / structural bound, gaps.min)`` at one ``qz^2`` -- the two
    numbers every acceptance in this module is decided on, read through the
    library's own :func:`_mode_reading` on the cell's own strips."""
    s, gaps, bound = eme_2d_vector._mode_reading(
        _cell(cell["Nx"]), cell["Lx"], cell["Nx"], cell["k0"], 0.0,
        float(qz2), cell["ky0"], cell["Ly"])
    return float(s[-1]) / bound, float(gaps.min())


def _fd(cell, qz2, ny=48):
    """Distance from ``qz^2`` to the nearest 2-D-FD oracle eigenvalue -- the
    library's own ``verify=True`` discriminator, independent of ``G``."""
    return eme_2d_vector._fd_eig_dist(_cell(cell["Nx"]), cell["Lx"],
                                      cell["Nx"], cell["Ly"], cell["k0"], 0.0,
                                      cell["ky0"], float(qz2), ny)


def _oracle_clean(cell, census, cusps, tag):
    """UNCONDITIONAL census content, stated against the ORACLES alone.

    Every entry must be (a) FD-confirmed -- an independent 2-D-FD eigenvalue
    within ``_FD_MODE``, the same test ``verify=True`` applies -- and (b) NOT a
    band-edge cusp, i.e. its ``sigma_min`` must sit ``_STRUCT_MODE`` below the
    structural bound that a cusp saturates.  And every known cusp of the window
    must be ABSENT.  None of this reads the pre-fix path, so none of it can go
    per-build."""
    for v in census:
        ratio, _g = _reading(cell, v)
        assert ratio < _STRUCT_MODE, (
            f"{tag}: the census entry {v!r} saturates the structural bound "
            f"(sigma_min / bound = {ratio:.3e}) -- it is a strip BAND EDGE, "
            f"not a mode")
        d = _fd(cell, v)
        assert d < _FD_MODE, (
            f"{tag}: the census entry {v!r} has no 2-D-FD eigenvalue near it "
            f"(distance {d:.4f}) -- it is spurious")
    for c in cusps:
        assert _absent(census, c), (
            f"{tag}: the band-edge cusp {c} is IN the census: {list(census)}")


# --------------------------------------------------------------------------- #
#  The independent oracle                                                      #
# --------------------------------------------------------------------------- #
def _monodromy_score(strips, Lx, Nx, Ly, k0, ky0, qz2):
    """``sigma_min(M - t I) / ||M - t I||`` -- zero exactly at a Bloch layer
    mode, and free of the block-``G`` basis entirely (see the module docstring).
    """
    M = None
    for eps, h in strips:
        E = expm(eme_2d_vector._strip_vector_generator(
            eps, Lx, Nx, k0, 0.0, np.sqrt(qz2)) * h)
        M = E if M is None else E @ M
    A = M - np.exp(1j * ky0 * Ly) * np.eye(4 * Nx)
    return float(np.linalg.svd(A, compute_uv=False)[-1] / np.linalg.norm(A, 2))


# --------------------------------------------------------------------------- #
#  The injectors                                                               #
# --------------------------------------------------------------------------- #
def _prefix_refine(monkeypatch):
    """Restore the PRE-FIX ``_refine_accept`` exactly: an empty ambiguity band
    (so nothing is ever polished) and an unreachable saturation ratio (so the
    structural test never fires).  What remains is Brent's answer read where
    Brent stopped -- the shipped 21802f9 behaviour."""
    monkeypatch.setattr(eme_2d_vector, "_CENSUS_BAND", (0.0, 0.0))
    monkeypatch.setattr(eme_2d_vector, "_STRUCTURAL_SAT", np.inf)


def _bracket_ulp(monkeypatch, k):
    """BUILD EMULATION.  Nudge the refinement bracket by ``|k|`` ULP.  A LAPACK
    build does not shift ``sigma_min`` uniformly -- it gives each evaluation its
    own last bit, which moves the minimiser's probe sequence.  Perturbing the
    bracket by one ULP is the smallest faithful, deterministic stand-in: the
    minimiser walks a different golden sequence over the same cell."""
    orig = eme_2d_vector.minimize_scalar
    direction = np.inf if k > 0 else -np.inf

    def wrapped(f, bounds=None, **kw):
        lo, hi = bounds
        for _ in range(abs(k)):
            lo = float(np.nextafter(lo, direction))
            hi = float(np.nextafter(hi, direction))
        return orig(f, bounds=(lo, hi), **kw)

    monkeypatch.setattr(eme_2d_vector, "minimize_scalar", wrapped)


_ULP_ARMS = (1, -1, 4, -4, 16, -16)
#: Widening rungs, walked ONLY when the narrow ladder leaves the pre-fix reading
#: of every known cusp bit-identical on this build -- i.e. when the emulation has
#: failed to move the minimiser at all and so has nothing to place a tie against.
#: Widen rather than delete: the reading's spread is what the injector needs, and
#: every build's bounded minimiser has one (the x-tolerance floor
#: ``sqrt(eps)|x| ~ 3.5e-6`` is 8 decades above one ULP of the bracket).
_ULP_ARMS_WIDE = _ULP_ARMS + (64, -64, 256, -256, 1024, -1024)


@contextlib.contextmanager
def _recorded_readings():
    """INSTRUMENT, not a substitute.  A pass-through wrapper on the library's
    ``_mode_reading`` that logs ``(qz^2, sigma_min, gaps.min, bound)`` for every
    acceptance reading the finder takes, and returns the library's own tuple
    unchanged -- so the census it observes is bit-for-bit the census it would
    have produced unobserved.  What it makes visible is the ONE number the
    accept/reject comparison is made on, which is what the engineered tie has to
    be placed against; nothing of the finder's logic is reimplemented here."""
    orig = eme_2d_vector._mode_reading
    log = []

    def wrapped(*a, **kw):
        s, gaps, bound = orig(*a, **kw)
        log.append((float(np.real(a[5])), float(s[-1]), float(gaps.min()),
                    float(bound)))
        return s, gaps, bound

    eme_2d_vector._mode_reading = wrapped
    try:
        yield log
    finally:
        eme_2d_vector._mode_reading = orig


def _prefix_ladder(monkeypatch, cell, arms, **kw):
    """Walk the PRE-FIX path over ``arms`` (plus the un-nudged arm ``0``),
    returning ``{k: (census, readings)}``."""
    out = {}
    for k in (0,) + tuple(arms):
        with monkeypatch.context() as mp:
            _prefix_refine(mp)
            if k:
                _bracket_ulp(mp, k)
            with _recorded_readings() as log:
                out[k] = (_census(cell, **kw), log)
    return out


def _tie_at_the_cut(rows, targets):
    """Place a SYNTHETIC NEAR-TIE at the accept/reject cut, and return the arm
    pair that must straddle it.

    ``ratio_tol`` is the bar the verdict is read against: ``_refine_accept``
    keeps a candidate iff ``gaps.min() < ratio_tol``.  WHERE the bounded
    minimiser halts is what sets that reading, and one ULP of a bracket endpoint
    moves where it halts -- so across the ladder ONE candidate is read at a whole
    SPREAD of values (measured for the W6 band-edge cusp: 9.51e-4 .. 2.60e-3,
    2.74x [M]).  Putting the bar at the GEOMETRIC MEAN of that measured spread
    makes the arm that read lowest ACCEPT and the arm that read highest REJECT,
    by construction, on any build.  All a build is left to decide is how wide its
    own spread is -- never that there is one, and never which side of a FIXED bar
    it happens to fall on, which is the per-build fact this replaces.

    Returns the dict ``{target, k_lo, k_hi, g_lo, g_hi, cut, spread}`` for the
    target with the widest spread, or ``None`` if no target moved at all."""
    best = None
    for t in targets:
        g = {}
        for k, (_q, log) in rows.items():
            near = [r for r in log if abs(r[0] - t) < 1e-2]
            if near:
                g[k] = min(near, key=lambda r: abs(r[0] - t))[2]
        if len(g) < 2:
            continue
        k_lo, k_hi = min(g, key=g.get), max(g, key=g.get)
        spread = g[k_hi] / g[k_lo]
        if best is None or spread > best["spread"]:
            best = dict(target=t, k_lo=k_lo, k_hi=k_hi, g_lo=g[k_lo],
                        g_hi=g[k_hi], spread=spread,
                        cut=float(np.sqrt(g[k_lo] * g[k_hi])))
    return best if best is not None and best["spread"] > 1.0 else None


def _prefix_drop_cut(g_brent):
    """The engineered cut for the RECALL demonstration, from the one reading the
    pre-fix path took on this build.

    The pre-fix path reads acceptance where bounded Brent halted; the fixed path
    polishes an in-band candidate to the converged zero and reads it THERE.  The
    two readings are separated by the physics, not by the build -- measured
    ``gaps.min`` 1.51e-3 at Brent's stop against 4.31e-12 at the converged zero,
    nine decades.  A bar placed anywhere strictly between them therefore makes
    the pre-fix path DROP the mode and the fixed path KEEP it, on every build.

    ``_CENSUS_BAND[1]`` is the only other constraint: the fixed path only
    polishes while ``gaps.min() <= _CENSUS_BAND[1] * ratio_tol``.  Dividing
    Brent's own reading by ``sqrt(_CENSUS_BAND[1])`` lands the bar at the
    geometric centre of the usable range ``(1, _CENSUS_BAND[1])`` -- DERIVED from
    the library's own constant, so if that constant moves the tie moves with
    it."""
    return g_brent / float(np.sqrt(eme_2d_vector._CENSUS_BAND[1]))


# =========================================================================== #
#  1.  The refused sqrt cusps are not modes -- by an independent condition     #
# =========================================================================== #
def test_the_refused_sqrt_cusps_are_not_modes_of_an_independent_condition():
    """The candidates the fix refuses are indistinguishable, to the monodromy
    condition, from an arbitrary ``qz^2``; the accepted modes sit nine decades
    below.  So refusing them is not a tightened tolerance -- they are not modes.
    """
    strips = _cell(8)
    kw = dict(strips=strips, Lx=1.0, Nx=8, Ly=1.0, k0=8.0, ky0=PI)
    modes = [_monodromy_score(qz2=q, **kw) for q in _MODES_W6]
    cusps = [_monodromy_score(qz2=q, **kw) for q in _CUSPS_W6]
    controls = [_monodromy_score(qz2=q, **kw) for q in (190.0, 220.0, 245.0)]
    assert max(modes) < _MONO_MODE, (
        f"the accepted modes do not solve the monodromy condition: {modes}")
    assert min(cusps) > _MONO_MODE, (
        f"a refused cusp DOES solve the monodromy condition: {cusps}")
    # and the cusps are not merely 'worse than a mode' -- they are ordinary
    assert min(cusps) > 0.1 * min(controls) and max(cusps) < 10 * max(controls)


# =========================================================================== #
#  2.  The structural bound is a bound, and saturates only at a band edge      #
# =========================================================================== #
def test_sigma_min_saturates_the_structural_bound_only_at_a_band_edge():
    """``_pair_singularity_bound`` is a THEOREM (``sigma_min <= sqrt(1 - c)``
    for the coalescing forward/backward pair), and the ratio it defines is what
    separates a band-edge cusp from a mode without reading any round-off."""
    strips = _cell(8)
    a = (strips, 1.0, 8, 8.0, 0.0)

    def ratio(q):
        s, _gaps, bound = eme_2d_vector._mode_reading(*a, q, PI, 1.0)
        assert float(s[-1]) <= bound * (1 + 1e-12), (
            f"the bound is not a bound at qz^2={q}: sigma_min {s[-1]:.3e} > "
            f"{bound:.3e}")
        return float(s[-1]) / bound

    at_modes = [ratio(q) for q in _MODES_W6]
    at_cusps = [ratio(q) for q in _CUSPS_W6]
    # and at the points the minimiser actually stops on, which is where the
    # library reads them (the cusp readings are dq-independent -- both the
    # numerator and the bound scale as sqrt(dq))
    at_cusps += [ratio(q) for q in (235.8686324974, 180.7703369636)]
    assert max(at_modes) < _STRUCT_MODE, f"a mode saturates the bound: {at_modes}"
    assert min(at_cusps) > eme_2d_vector._STRUCTURAL_SAT, (
        f"a band-edge cusp does not saturate the bound: {at_cusps}")
    assert min(at_cusps) / max(at_modes) > 1e3        # measured >= 1.4e10


# =========================================================================== #
#  3.  The fixed census is nudge-invariant; a TIE AT THE CUT flips the pre-fix #
# =========================================================================== #
def test_the_fixed_census_is_nudge_invariant_and_a_tie_at_the_cut_flips_the_prefix(
        monkeypatch):
    """The whole defect, in three layers -- none of which asks a build whether
    its own round-off happens to fall on one side of a fixed bar.

    (a) THE FIXED CENSUS, UNCONDITIONALLY.  On the W6 cell it holds every mode
        the MONODROMY oracle confirms (test 1), holds neither band-edge cusp,
        every entry is FD-confirmed and none saturates the structural bound --
        and it does not move, in size or in value, under the whole ULP ladder.
        Nothing in this layer reads the pre-fix path.

    (b) THE ENGINEERED TIE (the fail-before, deterministic on every build).
        The pre-fix verdict is ``gaps.min() < ratio_tol`` read where bounded
        Brent halted, and a 1-ULP nudge of the bracket moves where it halts:
        measured, ONE candidate -- the band-edge cusp -- is read across a 2.74x
        SPREAD of ``gaps.min`` over the ladder.  Put ``ratio_tol`` at the
        geometric mean of that spread and the arm that read lowest ACCEPTS the
        cusp while the arm that read highest REJECTS it, by construction.  At
        that same tie the FIXED path refuses it on every arm -- the structural
        bound is a property of ``G``, not of where a minimiser stopped (measured
        ``sigma_min / bound = 6.592e-01`` to 4 digits on all 13 arms).

    (c) THE LIVE CELL, at the SHIPPED ``ratio_tol``, ADJUDICATED.  Whether this
        build's own spread straddles ``1e-3`` is exactly the per-build fact that
        cannot be asserted -- it does on both our mounts (1/12 arms flip, gaining
        235.868633551) and does not on the ubuntu py3.11 runner.  It is measured
        and PRINTED either way; only (a) and (b) are asserted.
    """
    # ---- (a) the FIXED census, against the oracles alone ------------------ #
    clean = _census(_W6)
    for q in _MODES_W6:
        assert np.min(np.abs(clean - q)) < _VALUE_RTOL * q, (
            f"the fixed census is missing the monodromy-confirmed mode {q}: "
            f"{list(clean)}")
    _oracle_clean(_W6, clean, _CUSPS_W6, "W6 fixed")
    for k in _ULP_ARMS:
        with monkeypatch.context() as mp:
            _bracket_ulp(mp, k)
            q = _census(_W6)
        assert len(q) == len(clean), (
            f"the fixed census changed size under a {k:+d} ULP bracket nudge: "
            f"{list(q)} vs {list(clean)}")
        for v in q:
            assert np.min(np.abs(clean - v)) < 1e-4 * abs(v), (
                f"the fixed census moved a mode under a {k:+d} ULP nudge: "
                f"{list(q)} vs {list(clean)}")

    # ---- (b) the ENGINEERED TIE ------------------------------------------- #
    arms = _ULP_ARMS
    rows = _prefix_ladder(monkeypatch, _W6, arms)
    tie = _tie_at_the_cut(rows, _CUSPS_W6)
    if tie is None:                       # widen rather than give up (see above)
        arms = _ULP_ARMS_WIDE
        rows = _prefix_ladder(monkeypatch, _W6, arms)
        tie = _tie_at_the_cut(rows, _CUSPS_W6)
    assert tie is not None, (
        f"no rung of the bracket-ULP ladder {arms} moved the pre-fix reading of "
        f"EITHER known cusp {_CUSPS_W6} on this build, so there is no spread to "
        f"place a tie inside and this fail-before has no injector.  Widen the "
        f"ladder rather than deleting it -- the reading is a minimiser's "
        f"stopping value and every build's has a spread.")
    cut, target = tie["cut"], tie["target"]
    for k, want in ((tie["k_lo"], True), (tie["k_hi"], False)):
        with monkeypatch.context() as mp:
            _prefix_refine(mp)
            _bracket_ulp(mp, k)
            q = _census(_W6, ratio_tol=cut)
        got = not _absent(q, target, _VALUE_RTOL * target)
        assert got is want, (
            f"the engineered tie did not straddle: with ratio_tol = {cut:.6e} "
            f"placed between the {tie['k_lo']:+d} arm's reading "
            f"{tie['g_lo']:.6e} and the {tie['k_hi']:+d} arm's {tie['g_hi']:.6e},"
            f" the pre-fix census on the {k:+d} arm "
            f"{'lost' if want else 'gained'} the band-edge cusp {target} it "
            f"should have {'held' if want else 'refused'}: {list(q)}")
    fixed_cut = _census(_W6, ratio_tol=cut)
    for k in arms:
        with monkeypatch.context() as mp:
            _bracket_ulp(mp, k)
            q = _census(_W6, ratio_tol=cut)
        for c in _CUSPS_W6:
            assert _absent(q, c), (
                f"AT THE SAME TIE that flips the pre-fix path, the FIXED census "
                f"accepted the band-edge cusp {c} on the {k:+d} ULP arm: "
                f"{list(q)}")
        assert len(q) == len(fixed_cut), (
            f"the fixed census changed size at the engineered tie under a "
            f"{k:+d} ULP nudge: {list(q)} vs {list(fixed_cut)}")
    _oracle_clean(_W6, fixed_cut, _CUSPS_W6, "W6 fixed @tie")
    print(f"\nEME census tie [injector]: ratio_tol = {cut:.6e}, placed at the "
          f"geometric mean of this build's own pre-fix readings of {target} "
          f"({tie['g_lo']:.4e} on the {tie['k_lo']:+d} arm .. {tie['g_hi']:.4e} "
          f"on the {tie['k_hi']:+d}, {tie['spread']:.3g}x).  PRE-FIX straddles "
          f"it -- accepts the cusp on {tie['k_lo']:+d}, refuses it on "
          f"{tie['k_hi']:+d}; FIXED refuses it on all {len(arms) + 1} arms.")

    # ---- (c) the live cell at the SHIPPED tolerance, ADJUDICATED ---------- #
    base = rows[0][0]
    flips = {k: q for k, (q, _log) in rows.items() if k and len(q) != len(base)}
    sizes = {k: len(q) for k, q in flips.items()}
    span = (f"{tie['g_lo']:.4e} .. {tie['g_hi']:.4e} ({tie['spread']:.3g}x), "
            f"against the shipped bar 1.0e-03")
    if flips:
        extra = [float(v) for q in flips.values() for v in q
                 if _absent(base, v, _VALUE_RTOL * abs(v))]
        lost = sorted({float(v) for q in flips.values() for v in base
                       if _absent(q, v, _VALUE_RTOL * abs(v))})
        print(f"\nEME census tie [live cell]: the shipped bar sits INSIDE this "
              f"build's own spread of the {target} reading ({span}), so the "
              f"live-cell demonstration reproduces here -- {len(flips)} of "
              f"{len(arms)} ULP arms flip the pre-fix census (sizes {sizes} vs "
              f"{len(base)}), gaining {[float(f'{v:.10g}') for v in extra]} and "
              f"losing {[float(f'{v:.10g}') for v in lost]}.")
        for v in extra + lost:
            assert min(abs(v - c) for c in _CUSPS_W6) < _VALUE_RTOL * abs(v), (
                f"the pre-fix census flipped at the shipped tolerance, but by "
                f"gaining or losing {v!r}, which is not one of the known "
                f"band-edge cusps {_CUSPS_W6} -- a 1-ULP nudge moved a genuine "
                f"mode in or out")
    else:
        print(f"\nEME census tie [live cell]: the shipped bar sits OUTSIDE this "
              f"build's own spread of the {target} reading ({span}), so no arm "
              f"flips the pre-fix census here (all {len(arms)} return "
              f"{len(base)}) -- "
              f"the live-cell reading is inert on this build, and the "
              f"fail-before is carried by the engineered tie at "
              f"ratio_tol = {cut:.6e} above, which straddles by construction.")


# =========================================================================== #
#  4.  BYTE-NULL where the pre-fix decision was unambiguous                    #
# =========================================================================== #
@pytest.mark.parametrize("scale", [1.0, 10.0])
def test_the_census_is_byte_identical_where_the_prefix_path_was_unambiguous(
        monkeypatch, scale):
    """Containment, at both length scales.  The treatment reaches only the
    candidates whose verdict was round-off; every other one keeps the
    minimiser's own answer, BIT FOR BIT.

    Stated as the ``_CENSUS_BAND`` contract rather than as ``fixed == prefix``.
    The bald array equality is a PER-BUILD fact of the same class as the two
    restructured above: on a build whose pre-fix path accepts a W6 band-edge
    cusp -- the 2026-08-12 ubuntu runner did -- the fixed array MUST differ, by
    refusing it, and the byte-null test would fail for the fix working.  What is
    universal, and is the containment argument the fix actually makes, is:

      * every pre-fix entry whose reading fell OUTSIDE the ambiguity band comes
        back bit-identical (that is the 96.25%-of-candidates claim);
      * every entry the fixed path returns that is NOT one of those is a
        CONVERGED zero -- a reading decades below the band, i.e. the polish's
        output and not another stopping point.

    Where the two arrays do come out bit-identical (both our mounts, at both
    scales) that stronger reading is printed."""
    s = scale
    base = [(e, h * s) for e, h in _cell(8)]
    kw = dict(Lx=1.0 * s, Nx=8, Ly=1.0 * s, k0=8.0 / s,
              qz2_range=(150.0 / s ** 2, 250.0 / s ** 2), ky0=PI / s, n_scan=3)
    rtol_bar = 1e-3                    # ``layer_vector_modes``' shipped ratio_tol
    band = tuple(e * rtol_bar for e in eme_2d_vector._CENSUS_BAND)

    def _gapmin(q):
        _s, gaps, _b = eme_2d_vector._mode_reading(base, kw["Lx"], 8, kw["k0"],
                                                   0.0, float(q), kw["ky0"],
                                                   kw["Ly"])
        return float(gaps.min())

    fixed = eme_2d_vector.layer_vector_modes(base, **kw)
    with monkeypatch.context() as mp:
        _prefix_refine(mp)
        prefix = eme_2d_vector.layer_vector_modes(base, **kw)
    untreated = [q for q in prefix if not band[0] <= _gapmin(q) <= band[1]]
    for q in untreated:
        assert any(v == q for v in fixed), (
            f"not byte-null at scale {s}: the pre-fix entry {q!r}, whose "
            f"reading {_gapmin(q):.4e} is OUTSIDE the ambiguity band {band}, is "
            f"not returned bit-identically: {list(fixed)}")
    for v in fixed:
        if any(v == q for q in untreated):
            continue
        g = _gapmin(v)
        assert g < band[0], (
            f"at scale {s} the fixed census returned {v!r}, which is neither an "
            f"untreated pre-fix entry nor a converged zero (reading {g:.4e}, "
            f"band {band}): {list(fixed)} vs pre-fix {list(prefix)}")
    assert len(fixed) >= 3                                   # never vacuous
    if np.array_equal(fixed, prefix):
        print(f"\nEME census byte-null: at scale {s} the fixed and pre-fix "
              f"arrays are bit-identical ({len(fixed)} entries) -- this build's "
              f"pre-fix path already refused both W6 band-edge cusps, so the "
              f"treatment is a complete no-op here.")
    else:
        print(f"\nEME census byte-null: at scale {s} this build's pre-fix path "
              f"decided {len(prefix) - len(untreated)} W6 candidate(s) inside "
              f"the ambiguity band {band}, so the arrays differ by design "
              f"(fixed {list(fixed)} vs pre-fix {list(prefix)}); the "
              f"{len(untreated)} untreated entries come back bit-identical.")


# =========================================================================== #
#  5.  The polish converges the sqrt cusp and the simple V alike               #
# =========================================================================== #
def test_the_polish_converges_the_cusp_and_the_v_independently_of_localisation(
        monkeypatch):
    """The polisher's answer is set by the ARGUMENT tolerance, not by how the
    basin was localised: 17-, 33- and 65-point sub-grids agree, and ``|f|``
    collapses by decades from the minimiser's stopping value.  Both local forms
    are exercised -- the ``p = 1/2`` band-edge cusp and the ``p = 1`` mode."""
    strips = _cell(8)

    def f(q):
        try:
            return eme_2d_vector.dispersion_vec(strips, 1.0, 8, 8.0, 0.0, q,
                                                PI, 1.0, "dense")
        except np.linalg.LinAlgError:
            return np.inf

    for lo_b, hi_b, brent_x in ((235.75, 236.0, 235.8686324974),
                                (208.125, 208.375, 208.2502597917)):
        roots = []
        for sub in (17, 33, 65):
            with monkeypatch.context() as mp:
                mp.setattr(eme_2d_vector, "_POLISH_SUBGRID", sub)
                roots.append(eme_2d_vector._polish_zero(f, lo_b, hi_b))
        spread = (max(roots) - min(roots)) / abs(roots[0])
        assert spread <= _POLISH_AGREE, (
            f"the polished zero depends on its localisation: {roots}")
        assert f(roots[0]) < f(brent_x) / 100.0, (
            f"the polish did not deepen the zero in [{lo_b}, {hi_b}]: "
            f"{f(roots[0]):.3e} vs the minimiser's {f(brent_x):.3e}")


# =========================================================================== #
#  6.  The recovered mode is confirmed by the FD oracle, not by the pre-fix    #
# =========================================================================== #
def test_the_recovered_mode_is_confirmed_by_the_fd_oracle_not_by_the_prefix(
        monkeypatch):
    """The recall half, stated against the ORACLE rather than against a build.

    (a) UNCONDITIONAL.  The Nx=16 census CONTAINS 205.9749757788, on both
        solvers, at the converged zero -- ``sigma_min`` there is 8 decades under
        its value at ``_PREFIX_STOP``, where bounded Brent halts.  The in-tree
        2-D-FD eigenvalue oracle, independent of ``G``, puts an eigenvalue 0.076
        away from it, the closest match of the whole census; every other entry is
        FD-confirmed too, none saturates the structural bound, and none of the
        three known Nx=16 band-edge cusps is in it.

    (b) CONTAINMENT, in the form that is universal.  "The fix loses nothing" is
        NOT "the fixed census contains the pre-fix one" -- on a build whose
        pre-fix census holds a cusp, it must not.  The universal statement is
        that every pre-fix entry is EITHER kept OR refused as a band edge, which
        is asserted here and holds on any build.

    (c) THE ENGINEERED TIE (the fail-before, deterministic on every build).  The
        drop itself is per-build: our mounts halt at 205.9786352762 and read
        ``gaps.min = 1.51e-3`` just OVER the shipped 1e-3 bar, while the ubuntu
        py3.11 runner halts somewhere that reads just UNDER it and keeps the mode
        (3.99e-4 from the converged zero -- accepted, but not converged).  What
        is NOT per-build is that Brent's reading and the polished zero's reading
        are nine decades apart, so a bar placed between them -- at
        ``_prefix_drop_cut`` of the build's OWN Brent reading -- drops the mode
        pre-fix and keeps it fixed, by construction, everywhere.

    (d) The live cell at the SHIPPED bar, ADJUDICATED and printed.
    """
    fixed = _census(_N16)
    banded = _census(_N16, solver="banded")

    # ---- (a) content, against the oracles alone --------------------------- #
    for cen, tag in ((fixed, "dense"), (banded, "banded")):
        assert not _absent(cen, _RECOVERED, _VALUE_RTOL * _RECOVERED), (
            f"the {tag} census is missing the FD-confirmed mode {_RECOVERED}: "
            f"{list(cen)}")
    _oracle_clean(_N16, fixed, _CUSPS_N16, "N16 fixed")
    got = float(fixed[np.argmin(np.abs(fixed - _RECOVERED))])
    gb = float(banded[np.argmin(np.abs(banded - _RECOVERED))])
    assert abs(gb - got) < _VALUE_RTOL * abs(got), (
        f"the two solvers disagree on the recovered mode: {got!r} vs {gb!r}")
    # It is the CONVERGED zero, not a stopping point.  Stated in the two pieces
    # that are each universal: the POLISHER's own answer on the reference
    # detection cell is the zero to 1e-6 (localisation-independent -- test 5),
    # and the census entry IS that answer to within the bounded minimiser's own
    # x-tolerance floor, which is the closest any build is entitled to stop.
    x_pol = eme_2d_vector._polish_zero(_f16, *_N16_CELL)
    assert abs(x_pol - _RECOVERED) < 1e-6, (
        f"the polished zero of the detection cell {_N16_CELL} is {x_pol!r}, "
        f"not {_RECOVERED}")
    assert abs(got - x_pol) <= _BRENT_XFLOOR, (
        f"the census returned {got!r}, which is {abs(got - x_pol):.3e} from the "
        f"converged zero {x_pol!r} -- further than the minimiser's own "
        f"x-tolerance floor {_BRENT_XFLOOR:.3e}")
    f_stop, f_got = _f16(_PREFIX_STOP), _f16(got)
    assert f_got < f_stop / 1e2, (
        f"the returned {got!r} is not a converged zero: sigma_min {f_got:.3e} "
        f"against {f_stop:.3e} at the minimiser's stop {_PREFIX_STOP}")
    fd_got = _fd(_N16, got)
    fd_rest = [_fd(_N16, float(q)) for q in fixed if q != got]
    assert fd_rest, f"the Nx=16 census holds nothing but {got!r}: {list(fixed)}"
    assert fd_got <= 2.0 * max(fd_rest), (
        f"the recovered mode is a worse FD match ({fd_got:.4f}) than the rest "
        f"of the census ({fd_rest})")

    # ---- (b) containment, in its universal form --------------------------- #
    with monkeypatch.context() as mp:
        _prefix_refine(mp)
        prefix = _census(_N16)
    for q in prefix:
        if np.min(np.abs(fixed - q)) <= _VALUE_RTOL * abs(q):
            continue
        ratio, _g = _reading(_N16, q)
        assert ratio >= eme_2d_vector._STRUCTURAL_SAT, (
            f"the fix dropped the pre-fix entry {q!r}, which is NOT a band-edge "
            f"cusp (sigma_min / bound = {ratio:.3e}, FD distance "
            f"{_fd(_N16, q):.4f}): {list(fixed)} vs {list(prefix)}")

    # ---- (c) the ENGINEERED TIE, on the cell that holds the mode ---------- #
    with monkeypatch.context() as mp:
        _prefix_refine(mp)
        with _recorded_readings() as log:
            live = eme_2d_vector.layer_vector_modes(_cell(16), **_N16_NARROW)
    near = [r for r in log if abs(r[0] - _RECOVERED) < 0.5]
    assert near, (
        f"the pre-fix path took no acceptance reading within 0.5 of "
        f"{_RECOVERED} on the narrow cell {_N16_NARROW['qz2_range']}, so the "
        f"detection stage never offered the candidate and this injector has "
        f"nothing to place a tie against: {log}")
    g_brent = min(near, key=lambda r: abs(r[0] - _RECOVERED))[2]
    # the reading at the CONVERGED ZERO -- ``x_pol``, not the census entry
    # ``got``: where the fix clear-accepts (a build whose minimiser lands inside
    # ``_CENSUS_BAND``'s lower edge) ``got`` IS Brent's stop, and reading the
    # separation off it would compare that stop with itself.
    _ratio_pol, g_pol = _reading(_N16, x_pol)
    assert g_brent > 10.0 * g_pol, (
        f"the pre-fix reading at Brent's halt ({g_brent:.4e}) is not separated "
        f"from the converged zero's ({g_pol:.4e}), so there is no gap to place "
        f"a bar inside -- measured separation is 9 decades")
    cut = _prefix_drop_cut(g_brent)
    with monkeypatch.context() as mp:
        _prefix_refine(mp)
        pre_tie = eme_2d_vector.layer_vector_modes(_cell(16), ratio_tol=cut,
                                                   **_N16_NARROW)
    fix_tie = eme_2d_vector.layer_vector_modes(_cell(16), ratio_tol=cut,
                                               **_N16_NARROW)
    assert _absent(pre_tie, _RECOVERED), (
        f"with ratio_tol = {cut:.6e} placed between Brent's own reading "
        f"{g_brent:.4e} and the converged zero's {g_pol:.4e}, the PRE-FIX path "
        f"still accepted {_RECOVERED}: {list(pre_tie)} -- it read the rank drop "
        f"somewhere other than where it halted")
    assert not _absent(fix_tie, _RECOVERED, _VALUE_RTOL * _RECOVERED), (
        f"at the same tie the FIXED path did not return the converged zero "
        f"{_RECOVERED}: {list(fix_tie)}")
    print(f"\nEME census recall [injector]: ratio_tol = {cut:.6e}, placed "
          f"between this build's own pre-fix reading at Brent's halt "
          f"({g_brent:.4e}) and the converged zero's ({g_pol:.4e}).  PRE-FIX "
          f"DROPS {_RECOVERED} there ({list(pre_tie)}); FIXED polishes and "
          f"returns it ({list(fix_tie)}).")

    # ---- (d) the live cell at the SHIPPED bar, ADJUDICATED ---------------- #
    halt = min(near, key=lambda r: abs(r[0] - _RECOVERED))[0]
    if _absent(live, _RECOVERED):
        print(f"\nEME census recall [live cell]: this build's bounded Brent "
              f"halts at {halt:.10f}, reading gaps.min = {g_brent:.4e} OVER the "
              f"shipped 1.0e-03 bar, so the pre-fix path DROPS {_RECOVERED} on "
              f"the live cell -- the 2026-08-12 reading, reproduced here.")
    else:
        d = float(np.min(np.abs(np.asarray(live) - _RECOVERED)))
        print(f"\nEME census recall [live cell]: this build's bounded Brent "
              f"halts at {halt:.10f}, reading gaps.min = {g_brent:.4e} UNDER "
              f"the shipped 1.0e-03 bar, so the pre-fix path KEEPS the mode on "
              f"the live cell -- at {list(live)}, i.e. {d:.2e} from the "
              f"converged zero rather than {_VALUE_RTOL * _RECOVERED:.2e}.  The "
              f"drop is inert on this build; the fail-before is carried by the "
              f"engineered tie at ratio_tol = {cut:.6e} above.")
