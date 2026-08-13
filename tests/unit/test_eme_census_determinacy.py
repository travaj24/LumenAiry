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
_CUSPS_W6 = (235.8686333682, 180.7703378418)     # the two band-edge cusps in
#                           the W6 window that some builds accept as modes
_MODES_W6 = (208.2502609719, 203.7161764512, 156.2813759062)
_RECOVERED = 205.9749757788   # the Nx=16 mode bounded Brent drops on our mounts
_PREFIX_STOP = 205.9786352762  # ... where Brent stops instead (3.66e-3 away)


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


def _census(cell, **kw):
    return np.sort(eme_2d_vector.layer_vector_modes(_cell(cell["Nx"]),
                                                    **{**cell, **kw}))[::-1]


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
#  3.  FAIL-BEFORE: a one-ULP bracket nudge flips the pre-fix census           #
# =========================================================================== #
def test_a_one_ulp_bracket_nudge_flips_the_prefix_census_but_not_the_fixed_one(
        monkeypatch):
    """The whole defect in one measurement.

    PRE-FIX arm: perturbing the refinement bracket by ONE ULP -- far below any
    physical scale in the problem -- changes the returned census from three
    modes to four, the fourth being the band-edge cusp 235.8686 that the ubuntu
    runners also accepted.  A census that moves on a 1-ULP nudge of an interval
    endpoint is not a property of the layer.

    POST-FIX arm: every arm of the same ladder returns the same three modes.
    """
    clean_pre = None
    prefix_counts = []
    with monkeypatch.context() as mp:
        _prefix_refine(mp)
        clean_pre = _census(_W6)
        for k in _ULP_ARMS:
            with monkeypatch.context() as mp2:
                _prefix_refine(mp2)
                _bracket_ulp(mp2, k)
                prefix_counts.append((k, _census(_W6)))
    flips = [(k, q) for k, q in prefix_counts if len(q) != len(clean_pre)]
    assert flips, ("the pre-fix arm is no longer build-sensitive on this cell "
                   "-- the fail-before has gone vacuous")
    gained = np.concatenate([q for _k, q in flips])
    assert np.min(np.abs(gained - _CUSPS_W6[0])) < 1e-6, (
        "the pre-fix arm flipped, but not by accepting the known band-edge "
        f"cusp {_CUSPS_W6[0]}: {[(k, list(q)) for k, q in flips]}")

    clean = _census(_W6)
    assert len(clean) == len(clean_pre)      # the clean census is unchanged
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


# =========================================================================== #
#  4.  BYTE-NULL where the pre-fix decision was unambiguous                    #
# =========================================================================== #
@pytest.mark.parametrize("scale", [1.0, 10.0])
def test_the_census_is_byte_identical_where_the_prefix_path_was_unambiguous(
        monkeypatch, scale):
    """Containment.  On the W6 cell -- which DOES hold two ambiguous candidates,
    both band-edge cusps -- the fixed finder returns the pre-fix array BIT FOR
    BIT, at both length scales.  The treatment reaches only the candidates whose
    verdict was round-off; the modes keep the minimiser's own answer."""
    s = scale
    base = [(e, h * s) for e, h in _cell(8)]
    kw = dict(Lx=1.0 * s, Nx=8, Ly=1.0 * s, k0=8.0 / s,
              qz2_range=(150.0 / s ** 2, 250.0 / s ** 2), ky0=PI / s, n_scan=3)
    fixed = eme_2d_vector.layer_vector_modes(base, **kw)
    with monkeypatch.context() as mp:
        _prefix_refine(mp)
        prefix = eme_2d_vector.layer_vector_modes(base, **kw)
    assert np.array_equal(fixed, prefix), (
        f"not byte-null at scale {s}: fixed {list(fixed)} vs pre-fix "
        f"{list(prefix)}")
    assert len(fixed) >= 3                                   # never vacuous


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
#  6.  The dropped mode comes back, and the FD oracle confirms it              #
# =========================================================================== #
def test_the_dropped_mode_is_recovered_and_the_fd_oracle_confirms_it(
        monkeypatch):
    """The recall half.  On the Nx=16 grating both our mounts DROP the mode at
    205.9749757788 because bounded Brent halts 3.66e-3 away and reads the
    rank-drop there; the fixed finder returns it, at the converged zero, on
    either solver.  Confirmed by the in-tree 2-D-FD eigenvalue oracle, which is
    independent of ``G``: the recovered mode has an FD eigenvalue 0.076 away,
    as close as any mode the pre-fix census already contained."""
    strips = _cell(16)
    with monkeypatch.context() as mp:
        _prefix_refine(mp)
        prefix = _census(_N16)
    fixed = _census(_N16)
    banded = _census(_N16, solver="banded")

    assert np.min(np.abs(prefix - _RECOVERED)) > 1e-2, (
        f"the pre-fix census already contains {_RECOVERED} -- the recall arm "
        f"is vacuous on this build: {list(prefix)}")
    for cen, tag in ((fixed, "dense"), (banded, "banded")):
        assert np.min(np.abs(cen - _RECOVERED)) < 1e-6, (
            f"the {tag} census is still missing {_RECOVERED}: {list(cen)}")
    assert len(fixed) == len(prefix) + 1                 # exactly one gained
    for q in prefix:                                     # and nothing lost
        assert np.min(np.abs(fixed - q)) < 1e-9 * abs(q)

    # the value the acceptance used to be read at was 3.66e-3 away; it is now
    # the converged zero, and BOTH solvers land on it
    got = float(fixed[np.argmin(np.abs(fixed - _RECOVERED))])
    assert abs(_PREFIX_STOP - _RECOVERED) > 3e-3         # the pre-fix error
    assert abs(got - _RECOVERED) < 1e-6                  # collapsed >= 3 decades
    gb = float(banded[np.argmin(np.abs(banded - _RECOVERED))])
    assert abs(gb - got) < 1e-6 * abs(got)

    fd = eme_2d_vector._fd_eig_dist(strips, 1.0, 16, 1.0, 8.0, 0.0, PI, got, 48)
    fd_known = [eme_2d_vector._fd_eig_dist(strips, 1.0, 16, 1.0, 8.0, 0.0, PI,
                                           float(q), 48) for q in prefix]
    assert fd < 1.0, (
        f"the recovered qz^2 has no 2-D-FD eigenvalue near it (dist {fd:.3f})")
    assert fd <= max(fd_known) * 2.0, (
        f"the recovered mode is a worse FD match ({fd:.4f}) than the modes the "
        f"pre-fix census already held ({fd_known})")
