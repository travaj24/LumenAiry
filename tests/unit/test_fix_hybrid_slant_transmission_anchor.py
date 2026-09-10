"""The FRAME-ANCHOR phase on ``PMM2DStackHybrid``'s TRANSMITTED amplitudes --
defect D1 of ``docs/audits/VERIFY_PMM2D_STAGGERED_SLANT_2026_09_10.md`` S3.4,
fixed and documented in
``docs/audits/FIX_HYBRID_SLANT_TRANSMISSION_ANCHOR_2026_09_11.md``.

THE PHYSICS.  A slanted PATTERNED layer is solved in the sheared frame
``u = x - t z``, ``w = z``, anchored at the layer's own TOP face, so the state
the cascade carries is the FRAME Fourier coefficient.  At the layer's BOTTOM
that frame plane sits at ``u = x - t d``, so against the substrate's own lab
basis the transmitted amplitude of order ``m`` needs one unimodular phase
``exp(-i alpha_m . t d)`` (internal gauge) = ``exp(+i k0 (alpha_m . slant) d)``
on the PUBLIC amplitudes, accumulated over the sheared layers.  ``alpha_m`` is
real for every order, so the factor NEVER moves an efficiency: ``R``, ``T`` and
the REFLECTION Jones were exact without it, which is precisely why omitting it
was invisible to every energy check for the life of the 2026-08-16 slant metric.

TWO SHAPES CARRY A SLANT AND MUST NOT BE ANCHORED, because the engine solves
them as the plain vertical film and they never enter a frame: a UNIFORM layer
(``add_layer`` does not even store the slant) and a PATTERNED layer whose tile
is CONSTANT-VALUED (``_build_layer_modes`` short-circuits to
``_homogeneous_modes`` before the slant is read).  Anchoring either would cost
``1.371e+00`` on answers that are currently exact at ``0.000e+00``.

EVERY BAR BELOW IS DERIVED FROM A MEASUREMENT MADE ON **TWO BUILDS** on
2026-09-11 (``validation/probe_fix_hybrid_slant_anchor/p4_test_bars.py``,
results JSONs ``p4_test_bars_post_fix_{win,wsl}.json``), and the two readings
are stated in the assertion's comment:

  * WIN -- Windows 11, CPython 3.14.6, numpy 2.4.4, scipy 1.17.1
    (scipy-openblas), OMP/OPENBLAS/MKL = 1;
  * WSL -- Ubuntu, CPython 3.12.3, numpy 2.4.6, scipy 1.17.1
    (scipy-openblas), OMP/OPENBLAS = 1.

**The measured cross-build gap on this file's fixture is BELOW the fourth
significant digit on every one of the 34 quantities re-measured** -- WIN and
WSL print identically, to every digit, on all of them.  The bars are therefore
set from the SIGNAL separation (the ratios below run 12x .. 176x), never from
the spread, exactly as TESTING_STANDARDS rule 5 asks.

Shapes used, per ``docs/TESTING_STANDARDS.md``:

  * DECISIONS, not readings -- "the un-anchored arm does not improve when the
    oracle refines, while the shipped one does"; "no single GLOBAL phase can do
    what the per-order factor does";
  * TWO-SIDED arms built through the shipped API -- the pre-fix library is
    reconstructed by monkeypatching ``_slant_frame_walk`` to ``(0, 0)``, which
    IS the shipped decision point, so the fail-before demonstration runs on
    this build's own bytes;
  * every bit-identity claim is same-build, two-arm, by sha256 of raw bytes;
  * bounds are the ORACLE's own convergence step measured in the same run,
    never an absolute floor (the hybrid's Fourier truncation floor moves with
    ``n_orders``, the geometry and the build).

THE ORACLE.  A z-STAIRCASE of the same solid has no frame at all, so its
transmitted amplitudes are lab-referenced BY CONSTRUCTION, and running it
through the SAME engine removes every convention question.  THE WALK IS HALF A
PERIOD: at a WHOLE-period walk ``exp(2 pi i m) = 1`` for every order and the
anchor degenerates into a single global phase that no measurement could tell
from any other (the trap recorded in
``validation/probe_verify_slant/v8_hybrid_anchor.py``).

GEOMETRY NOTE.  ``px = 1.0 um`` at ``wl = 0.68 um`` keeps every order clear of a
half-space cut-off and no cell value equals a half-space ``eps``.  The first
fixture tried (``px = 1.2 um``, a cell containing ``eps = 1.0`` = the
superstrate) drove the slanted-layer-over-a-film cascade to ``sum R + T =
2.6e+27`` on some ``(n_orders, mount)`` pairs -- the documented
exactly-degenerate layer<->region mode match, which a slanted layer's
generalized cascade meets more readily than a vertical one.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import hashlib  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from lumenairy.elements.pmm import PMM2DStackHybrid, PMM2DStackPure  # noqa: E402
from lumenairy.elements.pmm import stack2d as _s2d  # noqa: E402

WL = 0.68e-6
K0 = 2.0 * np.pi / WL
PX = PY = 1.00e-6
DEP = 0.50e-6
DF = 0.25e-6
FEPS = 3.6
NSUP, NSUB = 1.0, 1.5
NXC, FINE = 6, 60
TSL = 1.0                     # t * d / px = 0.5 -- a HALF-period walk
WALK = TSL * DEP              # the accumulated frame offset, metres
BG = 1.44
XPROF = np.array([3.24, 3.24, 2.10, 1.15, 1.15, 1.15])
NORD = 5
MOUNTS = {"oblique25": (np.deg2rad(25.0), 0.0),
          "conical25_40": (np.deg2rad(25.0), np.deg2rad(40.0))}
_CACHE = {}

# ``n_orders = 5`` on this fixture leaves the closure at 1.008 .. 1.014 -- the
# 2-D hybrid's Fourier truncation residue at this size, which trips
# ``_warn_stack_energy``'s "R + T > 1" tripwire on every solve in the file.
# The reading is PINNED as an assertion in ``test_a5_...`` (and a genuine
# instability on this geometry reads 2.6e+27, 26 decades away), so the warning
# is filtered here rather than left to bury a real one.
pytestmark = pytest.mark.filterwarnings(
    r"ignore:.*energy not conserved.*:UserWarning")


def _cell(n=NXC):
    """An x-ASYMMETRIC cell -- its own mirror image is not a translate of it,
    so a sign error in the walk cannot hide behind a symmetry."""
    c = np.full((n, n), BG, dtype=complex)
    c[:, 0:n // 2] = np.repeat(XPROF, n // NXC)[:, None]
    return c


def _st(nord=NORD):
    return PMM2DStackHybrid(PX, PY, n_superstrate=NSUP, n_substrate=NSUB,
                            n_orders=nord)


def _sha(*arrs):
    m = hashlib.sha256()
    for a in arrs:
        m.update(np.ascontiguousarray(np.asarray(a)).tobytes())
    return m.hexdigest()


def _memo(key, fn):
    if key not in _CACHE:
        _CACHE[key] = fn()
    return _CACHE[key]


def _slanted(mount, nord=NORD, film=False):
    def build():
        th, ph = MOUNTS[mount]
        st = _st(nord)
        st.add_layer(DEP, eps_cell=_cell(), slant=(TSL, 0.0))
        if film:
            st.add_layer(DF, eps=FEPS)
        st.set_source(WL, theta=th, phi=ph)
        st._RTJ = st.solve()
        return st
    return _memo(("slanted", mount, nord, film), build)


def _stair(mount, K, film=False):
    """A fine z-STAIRCASE of the SAME solid: lab-referenced by construction."""
    def build():
        th, ph = MOUNTS[mount]
        st = _st()
        c = _cell(FINE)
        for k in range(K):
            sh = FINE * TSL * DEP / PX * (k + 0.5) / K
            assert abs(sh - round(sh)) < 1e-9, (K, k, sh)   # exact pixel walls
            st.add_layer(DEP / K, eps_cell=np.roll(c, int(round(sh)), axis=0))
        if film:
            st.add_layer(DF, eps=FEPS)
        st.set_source(WL, theta=th, phi=ph)
        st._RTJ = st.solve()
        return st
    return _memo(("stair", mount, K, film), build)


def _align(o1, o2):
    k1 = {tuple(int(v) for v in o1[i]): i for i in range(len(o1))}
    k2 = {tuple(int(v) for v in o2[i]): i for i in range(len(o2))}
    common = sorted(set(k1) & set(k2))
    return [k1[c] for c in common], [k2[c] for c in common]


def _rephase(st, w_alt, w_shipped=WALK):
    """The transmitted amplitudes ``st`` WOULD have returned had its frame
    anchor been ``w_alt`` metres instead of the shipped ``w_shipped``.

    ``w_alt = 0`` reconstructs the PRE-FIX library exactly (verified against a
    monkeypatched solve in ``test_c1_...`` below, and against the read-only
    main clone in the fix doc's census)."""
    a = st.per_order_amplitudes("transmission")
    P = np.exp(1j * K0 * a["kx"] * (w_alt - w_shipped))
    return a["Ex"] * P[None, :], a["Ey"] * P[None, :], a["orders"]


def _dT(st, ref, w_alt=WALK, w_shipped=WALK):
    """Per-order, both-polarization distance of ``st``'s transmitted
    amplitudes (re-anchored at ``w_alt``) from the lab-referenced ``ref``."""
    Ex, Ey, o1 = _rephase(st, w_alt, w_shipped)
    B = ref.per_order_amplitudes("transmission")
    j1, j2 = _align(o1, B["orders"])
    return max(float(np.max(np.abs(Ex[:, j1] - B["Ex"][:, j2]))),
               float(np.max(np.abs(Ey[:, j1] - B["Ey"][:, j2]))))


def _stair_step(mount, coarse=5, fine=15, film=False):
    """The ORACLE's OWN convergence step -- the derived bound every residual
    below is judged against (TESTING_STANDARDS rule 2)."""
    a = _stair(mount, fine, film).per_order_amplitudes("transmission")
    b = _stair(mount, coarse, film).per_order_amplitudes("transmission")
    j1, j2 = _align(a["orders"], b["orders"])
    return max(float(np.max(np.abs(a["Ex"][:, j1] - b["Ex"][:, j2]))),
               float(np.max(np.abs(a["Ey"][:, j1] - b["Ey"][:, j2]))))


def _P0(st):
    p0 = int(st._modal["p0"])
    return complex(np.exp(1j * K0 * st._modal["kx"][p0] * WALK))


# ==========================================================================
# A.  The transmitted amplitudes are LAB-referenced -- three arms, per order
# ==========================================================================
@pytest.mark.parametrize("mount", list(MOUNTS))
def test_a1_transmitted_amplitudes_are_lab_referenced(mount):
    """Against the engine's OWN fine staircase of the same solid, per order and
    both polarizations, the shipped amplitudes sit inside the oracle's own
    convergence uncertainty, while dropping the anchor or conjugating it is two
    decades worse.

    MEASURED (WIN == WSL to every printed digit):

    | arm | oblique 25 | conical 25-40 |
    |---|---|---|
    | shipped | 7.778e-03 | 7.381e-03 |
    | no anchor (the PRE-FIX library) | 1.113e+00 | 8.936e-01 |
    | the CONJUGATE anchor | 1.242e+00 | 1.303e+00 |
    | the oracle's own K3 -> K15 step | 4.118e-02 | 2.629e-02 |
    | amplitude scale | 6.694e-01 | 6.527e-01 |

    Bars: ``none/shipped`` and ``conj/shipped`` read 143x / 121x and 160x /
    176x, barred at 20x; the shipped residual sits 5.3x / 3.6x below the
    oracle's own coarsest rung, barred at 1x (i.e. "inside the oracle's own
    uncertainty", a derived bound, not a floor)."""
    s, k15 = _slanted(mount), _stair(mount, 15)
    shipped = _dT(s, k15)
    none = _dT(s, k15, 0.0)
    conj = _dT(s, k15, -WALK)
    oracle_step = _stair_step(mount, coarse=3, fine=15)
    assert shipped < oracle_step, (shipped, oracle_step)
    assert none / shipped > 20.0, (none, shipped)
    assert conj / shipped > 20.0, (conj, shipped)
    # and the conjugate is WORSE than doing nothing -- so the factor cannot be
    # a fudge absorbing an arbitrary residual (1.116x / 1.458x measured).
    assert conj > none


@pytest.mark.parametrize("mount", list(MOUNTS))
def test_a2_the_unanchored_arm_does_not_improve_when_the_oracle_refines(mount):
    """The decision that identifies the defect as a REFERENCE FRAME and not an
    accuracy gap: refine the staircase and the shipped arm converges toward it
    while the un-anchored one stands still.

    MEASURED, shipped vs K = 3 / 5 / 15 (WIN == WSL):
      oblique  3.887e-02  1.490e-02  7.778e-03   -> first/last 5.00x
      conical  2.677e-02  1.275e-02  7.381e-03   -> first/last 3.63x
    and the SAME comparison with the anchor removed:
      oblique  1.113e+00  1.114e+00  1.113e+00   -> first/last 1.000x
      conical  8.905e-01  8.944e-01  8.936e-01   -> first/last 0.9965x

    Bars: shipped improves by more than 2x (measured 5.00x / 3.63x); the
    un-anchored arm moves by less than 10% either way (measured 0.35% / 0.35%).
    Stated as first-to-last, NOT rung-by-rung strict monotonicity: the rungs
    are 0.7e-02 apart on a ladder whose own step is of that size."""
    s = _slanted(mount)
    ship = [_dT(s, _stair(mount, K)) for K in (3, 5, 15)]
    none = [_dT(s, _stair(mount, K), 0.0) for K in (3, 5, 15)]
    assert ship[0] / ship[-1] > 2.0, ship
    assert ship[-1] < ship[0]
    assert 0.9 < none[0] / none[-1] < 1.1, none


@pytest.mark.parametrize("mount", list(MOUNTS))
def test_a3_no_single_global_phase_can_replace_the_per_order_factor(mount):
    """The correction is genuinely PER ORDER.  Scan every global phase and take
    the best one the un-anchored amplitudes could possibly wear: it does not
    come close.

    MEASURED, best single global phase on the pre-fix arm (WIN == WSL):
    3.821e-01 (oblique) / 4.476e-01 (conical), against the shipped per-order
    7.778e-03 / 7.381e-03 -- 49.1x / 60.6x.  Barred at 10x."""
    s, k15 = _slanted(mount), _stair(mount, 15)
    Ex, Ey, o1 = _rephase(s, 0.0)
    B = k15.per_order_amplitudes("transmission")
    j1, j2 = _align(o1, B["orders"])
    best = min(max(float(np.max(np.abs(Ex[:, j1] * np.exp(1j * a)
                                       - B["Ex"][:, j2]))),
                   float(np.max(np.abs(Ey[:, j1] * np.exp(1j * a)
                                       - B["Ey"][:, j2]))))
               for a in np.linspace(-np.pi, np.pi, 721))
    shipped = _dT(s, k15)
    assert best / shipped > 10.0, (best, shipped)


@pytest.mark.parametrize("mount", list(MOUNTS))
def test_a4_zeroth_order_jones_transmission_carries_it_too(mount):
    """``jones_transmission()`` reads the SAME retained amplitudes, so it moves
    with them.  MEASURED shipped / un-anchored: 5.980e-03 / 1.113e+00
    (oblique), 7.381e-03 / 8.936e-01 (conical); ``arg P0`` = 1.952 / 1.496 rad,
    i.e. the zeroth order's own phase is a large fraction of a turn (a test
    that read only ``|J|`` would see nothing)."""
    s, k15 = _slanted(mount), _stair(mount, 15)
    P0 = _P0(s)
    shipped = float(np.max(np.abs(s.jones_transmission()
                                  - k15.jones_transmission())))
    none = float(np.max(np.abs(s.jones_transmission() * np.conj(P0)
                               - k15.jones_transmission())))
    assert abs(abs(P0) - 1.0) < 1e-14           # unimodular: no energy moves
    assert abs(np.angle(P0)) > 1.0              # ... and not a near-identity
    assert none / shipped > 20.0, (none, shipped)


def test_a5_the_fixture_closes_where_it_is_documented_to():
    """The fixture's own energy closure, asserted rather than warned past.

    ``n_orders = 5`` on this cell leaves ``sum R + T`` at ``1.011002`` /
    ``1.008581`` (slanted layer, oblique / conical) and ``1.012042`` /
    ``1.010179`` on the 15-slice staircase -- the Fourier truncation residue of
    a 2-D hybrid at this size, NOT an instability, which is why the file
    filters ``_warn_stack_energy``'s tripwire at module scope and pins the
    reading here instead.  A genuine instability on this geometry is 26 decades
    away (``sum R + T = 2.6e+27`` -- the fixture that was rejected while
    choosing this one), so the 1.05 bar cannot confuse the two, and the lower
    bar of 1.0 keeps a silently-lossy answer from passing."""
    for mount in MOUNTS:
        for st in (_slanted(mount), _stair(mount, 15),
                   _slanted(mount, film=True)):
            _o, R, T, _J = st._RTJ
            tot = float(np.max(R.sum(axis=1) + T.sum(axis=1)))
            assert 1.0 <= tot < 1.05, (mount, tot)


def test_a6_the_anchor_is_unimodular_on_every_order():
    """The factor cannot move an efficiency, PROPAGATING OR EVANESCENT, because
    ``alpha_m`` is real for every order.  That is the structural reason ``R``,
    ``T`` and every energy check were blind to the defect.  MEASURED
    ``max ||P_m| - 1|`` = 2.2e-16 (both builds, 121 orders)."""
    s = _slanted("conical25_40")
    a = s.per_order_amplitudes("transmission")
    P = np.exp(1j * K0 * (a["kx"] * WALK))
    assert np.max(np.abs(np.abs(P) - 1.0)) < 1e-13
    # not vacuous: the orders really do span a wide range of phases
    assert np.ptp(np.angle(P)) > 3.0
    # and some of them are EVANESCENT in the substrate (kz imaginary)
    assert np.any(np.abs(np.imag(a["kz"])) > 1e-9)


# ==========================================================================
# B.  Cross-engine: the PURE staggered engine, which anchors its own
# ==========================================================================
@pytest.mark.parametrize("mount", list(MOUNTS))
def test_b1_pure_and_hybrid_transmitted_jones_now_agree(mount):
    """The two engines' transmitted amplitudes are referenced the SAME way
    after the fix -- the cross-engine claim "a layer moves between the engines
    unchanged" now holds on the transmission side too.

    MEASURED, per order and both polarizations, pure (``n_modes = 4``,
    ``n_orders = 3``) against hybrid (``n_orders = 5``):

    |  | oblique 25 | conical 25-40 |
    |---|---|---|
    | shipped hybrid | 1.299e-02 | 9.955e-03 |
    | hybrid with the anchor REMOVED | 1.113e+00 | 8.912e-01 |
    | the hybrid's OWN n_orders 5 -> 7 step | 5.981e-03 | 4.568e-03 |

    Bars: the cross-engine gap is 2.17x / 2.18x the hybrid's own truncation
    step, barred at 10x (a DERIVED bound -- the engine's own convergence, not
    an absolute floor); removing the anchor is 85.7x / 89.5x worse, barred at
    20x."""
    s = _slanted(mount)
    th, ph = MOUNTS[mount]
    sp = PMM2DStackPure(PX, PY, n_superstrate=NSUP, n_substrate=NSUB,
                        n_modes=4, n_orders=3)
    sp.add_layer(DEP, eps_cell=_cell(), slant=(TSL, 0.0))
    sp.set_source(WL, theta=th, phi=ph)
    sp.solve(jones=True)
    Ap = sp.per_order_amplitudes("transmission")
    Ah = s.per_order_amplitudes("transmission")
    m1, m2 = _align(Ap["orders"], Ah["orders"])
    Ex0, Ey0, _o = _rephase(s, 0.0)
    shipped = max(float(np.max(np.abs(Ap["Ex"][:, m1] - Ah["Ex"][:, m2]))),
                  float(np.max(np.abs(Ap["Ey"][:, m1] - Ah["Ey"][:, m2]))))
    pre_fix = max(float(np.max(np.abs(Ap["Ex"][:, m1] - Ex0[:, m2]))),
                  float(np.max(np.abs(Ap["Ey"][:, m1] - Ey0[:, m2]))))
    A7 = _slanted(mount, nord=7).per_order_amplitudes("transmission")
    n1, n2 = _align(Ah["orders"], A7["orders"])
    own_step = max(float(np.max(np.abs(Ah["Ex"][:, n1] - A7["Ex"][:, n2]))),
                   float(np.max(np.abs(Ah["Ey"][:, n1] - A7["Ey"][:, n2]))))
    assert shipped < 10.0 * own_step, (shipped, own_step)
    assert pre_fix / shipped > 20.0, (pre_fix, shipped)


# ==========================================================================
# C.  FAIL-BEFORE, and the bit-identity gates -- through the shipped code
# ==========================================================================
@pytest.mark.parametrize("mount", list(MOUNTS))
def test_c1_pre_fix_arm_moves_transmission_only(monkeypatch, mount):
    """THE FAIL-BEFORE, executed on this build's own bytes.  ``_slant_frame_walk``
    IS the shipped decision point, so forcing it to ``(0, 0)`` reproduces the
    pre-fix library exactly -- and the whole point of the defect is that
    nothing else moves:

      * ``sha256(R)``, ``sha256(T)`` and ``sha256(jones_reflection)`` are
        BYTE-IDENTICAL between the two arms;
      * ``sha256(jones_transmission)`` differs;
      * the patched arm's per-order transmission against the staircase reads
        the pre-fix number (1.113e+00 / 8.936e-01), 143x / 121x the shipped
        7.778e-03 / 7.381e-03.

    That is also the cross-check that this file's ``_rephase(w_alt = 0)``
    reconstruction IS the pre-fix library: the two agree here to 0.0 exactly."""
    s = _slanted(mount)
    o, R, T, J = s._RTJ

    th, ph = MOUNTS[mount]
    monkeypatch.setattr(_s2d, "_slant_frame_walk", lambda _layers: (0.0, 0.0))
    st = _st()
    st.add_layer(DEP, eps_cell=_cell(), slant=(TSL, 0.0))
    st.set_source(WL, theta=th, phi=ph)
    o0, R0, T0, J0 = st.solve()

    assert _sha(R) == _sha(R0)                     # efficiencies untouched
    assert _sha(T) == _sha(T0)
    assert _sha(J) == _sha(J0)                     # reflection Jones untouched
    assert _sha(o) == _sha(o0)
    assert _sha(s.jones_transmission()) != _sha(st.jones_transmission())

    # the patched arm IS what _rephase(0.0) reconstructs, bit for bit
    Ex0, Ey0, _o = _rephase(s, 0.0)
    a0 = st.per_order_amplitudes("transmission")
    assert float(np.max(np.abs(Ex0 - a0["Ex"]))) < 1e-15
    assert float(np.max(np.abs(Ey0 - a0["Ey"]))) < 1e-15

    k15 = _stair(mount, 15)
    b = k15.per_order_amplitudes("transmission")
    j1, j2 = _align(a0["orders"], b["orders"])
    pre = max(float(np.max(np.abs(a0["Ex"][:, j1] - b["Ex"][:, j2]))),
              float(np.max(np.abs(a0["Ey"][:, j1] - b["Ey"][:, j2]))))
    assert pre / _dT(s, k15) > 20.0, pre


@pytest.mark.parametrize("mount", list(MOUNTS))
def test_c2_efficiencies_and_reflection_track_the_staircase(mount):
    """The other side of the same coin, measured against the oracle rather than
    against a patched arm: ``R``, ``T`` and the reflection Jones of the slanted
    layer already sit INSIDE the staircase's own convergence step, so there was
    never anything to correct there.

    MEASURED (WIN == WSL): dR 9.773e-04 / 1.196e-03, dT 2.413e-03 / 2.932e-03,
    dJones(refl) 2.471e-03 / 1.889e-03, against the staircase's own K5 -> K15
    step of dR 2.01e-03 / 1.82e-03 and dJones(refl) 8.33e-03 / 7.69e-03."""
    s, k15, k5 = _slanted(mount), _stair(mount, 15), _stair(mount, 5)
    oA, RA, TA, JA = s._RTJ
    oB, RB, TB, JB = k15._RTJ
    oC, RC, TC, JC = k5._RTJ
    i1, i2 = _align(oA, oB)
    _x, i3 = _align(oA, oC)
    step_R = float(np.max(np.abs(RB[:, i2] - RC[:, i3])))
    step_J = float(np.max(np.abs(JB - JC)))
    assert float(np.max(np.abs(RA[:, i1] - RB[:, i2]))) < 2.0 * step_R
    assert float(np.max(np.abs(JA - JB))) < step_J


def test_c3_a_slanted_uniform_layer_is_byte_identical_to_the_vertical_film():
    """A UNIFORM layer never enters a frame (``add_layer`` does not even store
    its slant), so it must not be anchored.  sha256-identical to the vertical
    film on the transmission Jones AND on the per-order amplitudes; two-sided:
    anchoring it would cost ``1.371e+00`` (both builds), which is 2.0x the
    amplitude scale itself."""
    th, ph = MOUNTS["oblique25"]
    u0, u1 = _st(), _st()
    u0.add_layer(DEP, eps=2.25)
    u1.add_layer(DEP, eps=2.25, slant=(TSL, 0.0))
    for st in (u0, u1):
        st.set_source(WL, theta=th, phi=ph)
        st.solve()
    assert _sha(u0.jones_transmission()) == _sha(u1.jones_transmission())
    a0 = u0.per_order_amplitudes("transmission")
    a1 = u1.per_order_amplitudes("transmission")
    assert _sha(a0["Ex"], a0["Ey"]) == _sha(a1["Ex"], a1["Ey"])
    # the layer dict really has no slant key -- that is WHY the sum skips it
    assert "slant" not in u1._layers[0]
    assert _s2d._slant_frame_walk(u1._layers) == (0.0, 0.0)
    # NOT vacuous: the anchor this layer would have taken is a big number
    if_anchored = float(np.max(np.abs(u1.jones_transmission() * _P0(u1)
                                      - u0.jones_transmission())))
    assert if_anchored > 1.0, if_anchored


def test_c4_a_constant_tile_slanted_layer_is_byte_identical_too():
    """A PATTERNED layer whose tile is CONSTANT-VALUED is short-circuited to
    ``_homogeneous_modes`` in ``_build_layer_modes`` BEFORE the slant is read,
    so it is solved as the vertical film and must not be anchored either --
    even though it DOES carry a stored slant.  This is the row a naive "sum
    over every layer with a slant keyword" would break.

    The gate is two-sided at the decision point AND at the modal build, so the
    helper and the build can never drift apart."""
    th, ph = MOUNTS["oblique25"]
    u0, u2 = _st(), _st()
    u0.add_layer(DEP, eps=2.25)
    u2.add_layer(DEP, eps_cell=np.full((NXC, NXC), 2.25 + 0j),
                 slant=(TSL, 0.0))
    for st in (u0, u2):
        st.set_source(WL, theta=th, phi=ph)
        st.solve()
    assert u2._layers[0]["slant"] == (TSL, 0.0)          # it IS stored
    assert _s2d._layer_enters_slant_frame(u2._layers[0]) is False
    assert _s2d._slant_frame_walk(u2._layers) == (0.0, 0.0)
    assert _sha(u0.jones_transmission()) == _sha(u2.jones_transmission())
    # ... and the modal build agrees with the helper, on both sides
    kw = dict(kxv=np.zeros(3), kyv=np.zeros(3), ox=np.arange(-1, 2),
              oy=np.arange(-1, 2), kx0=0.0, ky0=0.0, k0=K0)
    assert u2._build_layer_modes(u2._layers[0], **kw)[0] == "sym"
    p = _st()
    p.add_layer(DEP, eps_cell=_cell(), slant=(TSL, 0.0))
    assert _s2d._layer_enters_slant_frame(p._layers[0]) is True
    assert p._build_layer_modes(p._layers[0], **kw)[0] == "gen"
    assert _s2d._slant_frame_walk(p._layers) == (TSL * DEP, 0.0)


def test_c5_a_vertical_stack_takes_no_anchor_at_all():
    """The zero case is STRUCTURAL, not tolerance-based: a stack with no
    sheared region returns ``(0.0, 0.0)`` from the walk helper, which is the
    branch that skips the multiplication entirely and keeps every vertical
    answer bit-identical to the pre-fix library."""
    st = _st()
    st.add_layer(0.20e-6, eps=2.10)
    st.add_layer(DEP, eps_cell=_cell())
    st.add_layer(0.30e-6, eps=3.20, slant=(0.6, 0.0))   # uniform: dropped
    assert _s2d._slant_frame_walk(st._layers) == (0.0, 0.0)
    for L in st._layers:
        assert _s2d._layer_enters_slant_frame(L) is False


# ==========================================================================
# D.  Composition -- the walks ADD over the layers that enter a frame
# ==========================================================================
@pytest.mark.parametrize("mount", list(MOUNTS))
def test_d1_two_slanted_layers_sum_their_walks(mount):
    """One slanted layer of depth ``d`` split into TWO of ``d/2``: the frame
    simply continues, so the anchor is the SUM of the two walks.

    The split identity alone cannot see the sum (both arms would be equally
    wrong), so the gate is the two-half stack against the lab-referenced
    STAIRCASE, with the wrong sums engineered next to it.

    MEASURED (WIN == WSL):

    | arm | oblique 25 | conical 25-40 |
    |---|---|---|
    | split identity, dJones_transmission | 0.000e+00 | 0.000e+00 |
    | full sum (shipped) | 7.778e-03 | 7.381e-03 |
    | only ONE half in the sum | 6.336e-01 | 4.839e-01 |
    | no sum at all | 1.113e+00 | 8.936e-01 |

    Barred at 20x for both wrong sums (measured 81.5x / 65.6x and 143x /
    121x)."""
    th, ph = MOUNTS[mount]
    c2 = _st()
    c2.add_layer(DEP / 2, eps_cell=_cell(), slant=(TSL, 0.0))
    c2.add_layer(DEP / 2, eps_cell=_cell(), slant=(TSL, 0.0))
    c2.set_source(WL, theta=th, phi=ph)
    c2.solve()
    assert _s2d._slant_frame_walk(c2._layers) == (TSL * DEP, 0.0)
    # the layer-split identity, exactly (the two halves ARE the one layer)
    assert _sha(c2.jones_transmission()) == _sha(
        _slanted(mount).jones_transmission())
    k15 = _stair(mount, 15)
    full = _dT(c2, k15)
    half = _dT(c2, k15, WALK / 2)
    none = _dT(c2, k15, 0.0)
    assert half / full > 20.0, (half, full)
    assert none / full > 20.0, (none, full)


@pytest.mark.parametrize("mount", list(MOUNTS))
def test_d2_a_uniform_film_below_needs_no_round_trip_phase(mount):
    """Put a reflecting UNIFORM film UNDER the slanted layer and ask the two
    questions the composition rule raises: does the film enter the sum (no --
    a lateral offset is a gauge for a homogeneous region), and does the
    reflection that comes back UP through the sheared layer need a round-trip
    factor (also no -- the frame is anchored on the superstrate side)?

    MEASURED (WIN == WSL), slanted layer + a 0.25 um ``eps = 3.6`` film,
    against the same staircase plus the same film:

    | quantity | oblique 25 | conical 25-40 |
    |---|---|---|
    | dR | 8.570e-04 | 1.664e-03 |
    | dJones_reflection, as returned | 3.909e-03 | 3.859e-03 |
    | the staircase's own K5 -> K15 step | 1.15e-02 | 1.37e-02 |
    | reflection x P0 (a one-way round trip) | 2.282e-01 | 2.169e-01 |
    | reflection x P0^2 (a full round trip) | 2.582e-01 | 3.211e-01 |
    | transmission, shipped | 7.241e-03 | 5.971e-02 |
    | transmission, no anchor | 1.063e+00 | 8.564e-01 |
    | closure sum R + T | 1.011813 | 1.009474 |

    Barred: the reflection as returned is inside the oracle's own step (2.9x /
    3.6x below it) while EITHER round-trip factor is 55x .. 83x above it; the
    transmitted arm is barred at 5x (measured 147x / 14.3x)."""
    s, k15, k5 = (_slanted(mount, film=True), _stair(mount, 15, film=True),
                  _stair(mount, 5, film=True))
    oA, RA, TA, JA = s._RTJ
    oB, RB, TB, JB = k15._RTJ
    oC, RC, TC, JC = k5._RTJ
    _i, i2 = _align(oA, oB)
    _x, i3 = _align(oA, oC)
    step_J = float(np.max(np.abs(JB - JC)))
    P0 = _P0(s)
    assert float(np.max(np.abs(JA - JB))) < step_J
    assert float(np.max(np.abs(JA * P0 - JB))) > 10.0 * step_J
    assert float(np.max(np.abs(JA * P0 * P0 - JB))) > 10.0 * step_J
    # the film is homogeneous -> it contributes NOTHING to the sum
    assert _s2d._slant_frame_walk(s._layers) == (TSL * DEP, 0.0)
    assert _dT(s, k15, 0.0) / _dT(s, k15) > 5.0


def test_d3_the_walk_is_the_layer_sum_and_nothing_else():
    """A stack that mixes every shape at once: a vertical patterned layer, a
    slanted UNIFORM film, a CONSTANT-tile slanted cell and TWO genuinely
    sheared patterned layers.  Only the last two may appear in the sum.

    This is a pure DECISION on the shipped helper -- no solve, no tolerance --
    and it is the gate that keeps a future "sum over anything with a slant
    keyword" from silently returning."""
    st = _st()
    st.add_layer(0.20e-6, eps_cell=_cell())                       # vertical
    st.add_layer(0.11e-6, eps=2.10, slant=(0.9, 0.4))             # uniform
    st.add_layer(0.13e-6, eps_cell=np.full((NXC, NXC), 2.1 + 0j),
                 slant=(0.7, 0.3))                                # const tile
    st.add_layer(0.30e-6, eps_cell=_cell(), slant=(0.5, 0.25))    # SHEARED
    st.add_layer(0.40e-6, eps_cell=_cell(), slant=(0.5, 0.25))    # SHEARED
    wx, wy = _s2d._slant_frame_walk(st._layers)
    assert wx == pytest.approx(0.5 * 0.30e-6 + 0.5 * 0.40e-6, rel=1e-15)
    assert wy == pytest.approx(0.25 * 0.30e-6 + 0.25 * 0.40e-6, rel=1e-15)
    assert [bool(_s2d._layer_enters_slant_frame(L)) for L in st._layers] == [
        False, False, False, True, True]


# ==========================================================================
# E.  SCOPE -- a PATTERNED layer below a slanted one rides the shear
# ==========================================================================
def test_e1_a_patterned_layer_below_a_slanted_one_rides_the_shear():
    """The one shape where the far-field anchor is NOT the whole story, pinned
    so it cannot change silently.

    With a PATTERNED layer below a sheared one, the cascade matches that
    layer's LAB coefficients directly to the sheared region's FRAME
    coefficients, which is the physically realizable solid in which the lower
    layer is TRANSLATED by the accumulated walk -- the frame simply continuing
    downward.  ``PMM2DStackPure`` refuses this shape outright; the hybrid
    accepts it, so the behaviour is measured and documented here.

    Fixture: a QUARTER-period walk (so ``+walk`` and ``-walk`` are DIFFERENT
    translations of the 12-pixel lower cell -- at a half walk the cell is its
    own image) and a 15-slice staircase of the upper layer, with the lower
    layer placed three ways.

    MEASURED, oblique 25, ``n_orders = 5``, closure ``sum R + T = 1.01358``
    (WIN == WSL to every printed digit):

    | lower layer placed | per-order T | dR | dJones_reflection |
    |---|---|---|---|
    | AS WRITTEN | 3.715e-01 | 1.056e-02 | 4.995e-02 |
    | translated by +walk (riding the shear) | **1.554e-02** | **1.297e-03** | **7.462e-03** |
    | translated by -walk | 1.649e-01 | 8.378e-03 | 2.765e-02 |

    The ``+walk`` arm wins by 23.9x / 10.6x on the transmission and 6.7x / 3.7x
    on the REFLECTION -- which carries no anchor at all, so it is the reading
    that identifies the GEOMETRY rather than the phase.  The same rows at
    ``n_orders = 7`` read 3.752e-01 / 1.290e-02 / 1.637e-01 and dJones(refl)
    5.054e-02 / 9.284e-03 / 2.771e-02, i.e. the decision does not move with
    truncation (``validation/probe_fix_hybrid_slant_anchor/p2_composition.py``).
    Bars: 3x on the transmission, 2x on the reflection and on dR."""
    th, ph = np.deg2rad(25.0), 0.0
    nord, tsl_d, fine_d, nxd = 5, 0.5, 120, 12
    d1, dv = DEP, 0.30e-6
    yprof = np.array([1.44, 2.89, 2.89, 1.44, 1.44, 2.10, 2.10, 1.44, 1.44,
                      1.44, 1.44, 1.44])

    def lower(n=nxd, roll=0):
        c = np.full((n, n), BG, dtype=complex)
        c[:, 0:n // 2] = yprof[:, None]
        return np.roll(c, roll, axis=0)

    st = PMM2DStackHybrid(PX, PY, n_superstrate=NSUP, n_substrate=NSUB,
                          n_orders=nord)
    st.add_layer(d1, eps_cell=_cell(), slant=(tsl_d, 0.0))
    st.add_layer(dv, eps_cell=lower())
    st.set_source(WL, theta=th, phi=ph)
    oA, RA, TA, JA = st.solve()
    roll = int(round(nxd * tsl_d * d1 / PX))            # 3 pixels of 12
    assert roll == 3

    res = {}
    for tag, sh in (("as_written", 0), ("plus", +roll), ("minus", -roll)):
        s2 = PMM2DStackHybrid(PX, PY, n_superstrate=NSUP, n_substrate=NSUB,
                              n_orders=nord)
        c = np.full((fine_d, fine_d), BG, dtype=complex)
        c[:, 0:fine_d // 2] = np.repeat(XPROF, fine_d // NXC)[:, None]
        for k in range(15):
            off = fine_d * tsl_d * d1 / PX * (k + 0.5) / 15
            assert abs(off - round(off)) < 1e-9
            s2.add_layer(d1 / 15, eps_cell=np.roll(c, int(round(off)), axis=0))
        s2.add_layer(dv, eps_cell=lower(roll=sh))
        s2.set_source(WL, theta=th, phi=ph)
        oB, RB, TB, JB = s2.solve()
        i1, i2 = _align(oA, oB)
        a = st.per_order_amplitudes("transmission")
        b = s2.per_order_amplitudes("transmission")
        j1, j2 = _align(a["orders"], b["orders"])
        res[tag] = dict(
            T=max(float(np.max(np.abs(a["Ex"][:, j1] - b["Ex"][:, j2]))),
                  float(np.max(np.abs(a["Ey"][:, j1] - b["Ey"][:, j2])))),
            dR=float(np.max(np.abs(RA[:, i1] - RB[:, i2]))),
            dJ=float(np.max(np.abs(JA - JB))))
    # the anchored transmission and the un-anchored REFLECTION both point at
    # the SHEAR-CONTINUED geometry, by a wide margin, barred at 3x
    assert res["as_written"]["T"] / res["plus"]["T"] > 3.0, res
    assert res["minus"]["T"] / res["plus"]["T"] > 3.0, res
    assert res["as_written"]["dJ"] / res["plus"]["dJ"] > 2.0, res
    assert res["minus"]["dJ"] / res["plus"]["dJ"] > 2.0, res
    assert res["as_written"]["dR"] / res["plus"]["dR"] > 2.0, res


# ==========================================================================
# F.  The surfaces that are NOT affected -- pinned so a future change is loud
# ==========================================================================
def test_f1_the_frame_referenced_surfaces_are_only_the_transmission_ones():
    """The census, as a gate.  ``retain_internal`` (and therefore
    ``internal_field`` / ``layer_absorption``) REFUSES on a slanted stack, and
    ``solve_vs_wavelength`` retains no amplitudes at all, so there is no other
    public surface that could serve a frame-referenced field.  If any of these
    ever starts returning on a slanted stack it needs its own anchor decision,
    and this test is what says so."""
    th, ph = MOUNTS["oblique25"]
    st = _st()
    st.add_layer(DEP, eps_cell=_cell(), slant=(TSL, 0.0))
    st.set_source(WL, theta=th, phi=ph)
    with pytest.raises(NotImplementedError):
        st.solve(retain_internal=True)
    st.solve()
    with pytest.raises(ValueError):
        st.internal_field(0.5 * DEP)
    with pytest.raises(ValueError):
        st.layer_absorption()
    st2 = _st()
    st2.add_layer(DEP, eps_cell=_cell(), slant=(TSL, 0.0))
    st2.set_source(WL, theta=th, phi=ph)
    st2.solve_vs_wavelength([WL], theta=th, phi=ph)
    with pytest.raises(ValueError):
        st2.jones_transmission()
    # the VERTICAL control proves the refusals are about the slant, not about
    # the API: the same calls succeed with the slant removed
    v = _st()
    v.add_layer(DEP, eps_cell=_cell())
    v.set_source(WL, theta=th, phi=ph)
    v.solve(retain_internal=True)
    assert v.layer_absorption() is not None


def test_f2_a_slanted_patterned_layer_refuses_the_JAX_path():
    """The SAME silent shape, one dispatch away, found while censusing D1 and
    closed in the same commit family.

    ``add_layer`` already refuses ``slant=`` on a TRACED ``eps_cell``, on the
    ground that "the slanted layer runs the 4N generator and the generalized
    cascade, which the 2-D JAX surface does not implement".  But SIX other
    traced inputs reach the JAX dispatch -- a layer THICKNESS, the wavelength,
    ``theta``/``phi``, a half-space index, a traced uniform ``eps`` -- and the
    jnp twin has no notion of a slant at all (``grep -c slant
    lumenairy/elements/pmm/_jax_stack2d.py`` == 0).

    MEASURED BEFORE the guard (WIN, jax 0.11.0, x64), a slanted patterned layer
    with a TRACED THICKNESS: the solve RETURNED, energy-conserving and
    unwarned, **bit-identical to the VERTICAL stack** (``dR = dJones =
    0.000e+00``) and wrong against the correct NumPy slanted answer by
    ``dR 1.839e-02 / dJones 3.227e-02``.

    Three arms, so the guard cannot be read as "JAX plus slant raises":

      * a traced THICKNESS and a traced WAVELENGTH both raise;
      * the VERTICAL control on the same traced thickness still SOLVES, so the
        refusal is about the shear and not about the API;
      * a CONSTANT-tile slanted layer -- which never enters a frame and is a
        genuine no-op -- still SOLVES, so the guard uses the same
        ``_layer_enters_slant_frame`` decision as the anchor itself.
    """
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    def build(thickness, wl, cell_, slant):
        st = _st()
        st.add_layer(thickness, eps_cell=cell_, slant=slant)
        st.set_source(wl, theta=MOUNTS["oblique25"][0], phi=0.0)
        return st

    with pytest.raises(NotImplementedError, match="SLANTED"):
        build(jnp.asarray(DEP), WL, _cell(), (TSL, 0.0)).solve()
    with pytest.raises(NotImplementedError, match="SLANTED"):
        build(DEP, jnp.asarray(WL), _cell(), (TSL, 0.0)).solve()
    # the VERTICAL control on the SAME traced thickness still solves
    o, R, T, J = build(jnp.asarray(DEP), WL, _cell(), None).solve()
    assert np.asarray(R).shape == np.asarray(T).shape
    # ... and so does the CONSTANT-tile "slanted" layer, which is a real no-op
    o2, R2, T2, J2 = build(jnp.asarray(DEP), WL,
                           np.full((NXC, NXC), 2.25 + 0j), (TSL, 0.0)).solve()
    assert float(np.max(np.asarray(R2).sum(axis=1)
                        + np.asarray(T2).sum(axis=1))) < 1.05
