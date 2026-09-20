"""WP-C1 -- ``apply_aperture(edge='gray')`` is the default from v5.49.0.

The maintainer's ruling is recorded in
``docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/MAINTAINER_DECISIONS_2026_09.md``
section 1.1, on the measurement in ``fixes/WP-B11_REPORT.md`` section 2.9.  This
file is the gate for that ruling.  It asserts four things and proves that each
of three mutations is caught by a NAMED test:

1. the default is ``'gray'`` -- read from the signature AND from a call whose
   result is compared bit for bit against an explicit ``edge='gray'`` call;
2. the way back (``edge='hard'``) is BIT-IDENTICAL to the pre-5.49 answer, which
   is what makes a pinned hard-aperture number one keyword away;
3. the convergence DECISION on both spatial kernels: the grey arm's error falls
   monotonically at second order, the hard arm's does not fall monotonically at
   all -- measured against a closed form, not against a finer discretisation;
4. the rim fractions themselves: the sub-sample lattice is cell-centred, which
   is what makes a rim through a pixel centre read exactly one half and a
   centred aperture's mask mirror-symmetric.

Bars.  Every numeric bound below is derived from a measurement quoted beside it
and has decades (or, where the claim is exact arithmetic on small dyadic
rationals, no bar at all).  The ladder readings were reproduced to five
significant figures on Windows py3.14 / scipy-openblas and WSL py3.12 /
scipy-openblas (``validation/probe_c1_gray_edge/ladder_*.json``), and they
reproduce WP-B11 section 2.9's sixteen table entries to the last digit, so the
cross-build spread of these quantities is below the last printed digit while the
decisions below turn on factors of 1.5 and more.

Runtime: the whole file is a few seconds; the two full ladders (both kernels,
N = 128..1024, both arms) measure at 2.5 s and 1.9 s on the reference box.
"""
import inspect

import numpy as np
import pytest

from lumenairy.elements import elements as elem_mod
from lumenairy.propagators.hf import (
    propagate_huygens_fresnel_with_opl_callable)
from lumenairy.propagators.rs import rayleigh_sommerfeld_propagate

apply_aperture = elem_mod.apply_aperture

# --- the WP-B11 sec. 2.9 / WP-C1 geometry ---------------------------------
WAVELENGTH = 633e-9
A = 100e-6            # aperture RADIUS [m]
WINDOW = 512e-6       # full window [m]
LADDER_N = (128, 256, 512, 1024)
Z_RS = 16e-3          # above the spatial kernel's alias threshold at every N
Z_HF = 5e-3


# ===========================================================================
# the oracle and the two kernels
# ===========================================================================

def _closed_form_on_axis(z):
    """The EXACT on-axis field behind a circular aperture in a plane screen
    under unit-amplitude plane-wave illumination,

        U(0, 0, z) = e^{ikz} - (z / r_a) e^{ik r_a},   r_a = sqrt(z^2 + a^2)

    (the Rayleigh-Sommerfeld integral evaluated on axis; Born & Wolf sec.
    8.3.2).  It is a closed form: nothing in it depends on N, on the kernel or
    on the build, which is what makes it an oracle rather than a finer run.
    """
    k = 2.0 * np.pi / WAVELENGTH
    r_a = np.sqrt(z * z + A * A)
    return np.exp(1j * k * z) - (z / r_a) * np.exp(1j * k * r_a)


def _apertured(N, **edge_kw):
    dx = WINDOW / N
    E = np.ones((N, N), dtype=complex)
    return apply_aperture(E, dx, 'circular', {'diameter': 2.0 * A},
                          **edge_kw), dx


def _rs_relative_error(N, **edge_kw):
    E, dx = _apertured(N, **edge_kw)
    out = rayleigh_sommerfeld_propagate(E, z=Z_RS, wavelength=WAVELENGTH,
                                        dx=dx, kernel='spatial')
    u = _closed_form_on_axis(Z_RS)
    return abs(complex(out[N // 2, N // 2]) - u) / abs(u)


def _hf_relative_error(N, **edge_kw):
    E, dx = _apertured(N, **edge_kw)
    z, lam = Z_HF, WAVELENGTH

    def opl_fn(s1x, s1y, s2x, s2y):          # WAVES, per the units contract
        return np.sqrt((s1x - s2x) ** 2 + (s1y - s2y) ** 2 + z * z) / lam

    out = propagate_huygens_fresnel_with_opl_callable(
        E, opl_fn=opl_fn, output_grid_x=np.array([0.0]),
        output_grid_y=np.array([0.0]), input_grid_dx=dx)
    u = _closed_form_on_axis(z)
    return abs(complex(np.reshape(out, (-1,))[0]) - u) / abs(u)


KERNELS = {'rs_spatial': _rs_relative_error, 'hf_quadrature': _hf_relative_error}


# ===========================================================================
# 1 -- the default
# ===========================================================================

def _default_is_gray(fn):
    """The check itself, as a function of the entry point, so the mutation
    matrix below can run it against a mutant."""
    assert inspect.signature(fn).parameters['edge'].default == 'gray', (
        "apply_aperture's edge default must be 'gray' from v5.49.0 "
        "(MAINTAINER_DECISIONS_2026_09.md sec. 1.1)")
    E = np.ones((64, 64), dtype=complex)
    args = (E, 2e-6, 'circular', {'diameter': 6.3e-5})
    got = np.asarray(fn(*args))
    gray = np.asarray(apply_aperture(*args, edge='gray'))
    hard = np.asarray(apply_aperture(*args, edge='hard'))
    assert got.tobytes() == gray.tobytes(), (
        "a call with no edge= must return the grey answer BIT FOR BIT")
    assert got.tobytes() != hard.tobytes(), (
        "the grey and hard answers must differ -- otherwise this fixture "
        "has no boundary pixels and proves nothing")


def test_c1_the_default_edge_is_gray_from_the_signature_and_from_a_call():
    """Both halves matter: a signature can say ``'gray'`` while a body
    re-binds it, and a body can behave greyly while the signature -- which is
    what ``help()``, the docs build and every reader see -- still says
    ``'hard'``."""
    _default_is_gray(apply_aperture)


def test_c1_edge_samples_default_is_four_and_four_is_the_measured_knee():
    """The knee, re-measured here rather than quoted.

    RS spatial kernel, N = 512, the WP-C1 geometry, on-axis relative error
    against the closed form as ``edge_samples`` climbs (2026-09-20, Windows
    py3.14; WSL py3.12 agrees to five figures):

        1 -> 3.4207e-04   (n_sub = 1 IS the pixel-centre indicator)
        2 -> 1.6563e-04
        4 -> 8.1013e-05   <- the shipped default
        8 -> 8.3954e-05
       16 -> 8.3959e-05

    Decisions, not readings: 2 -> 4 must still gain at least 1.5x (measured
    2.04x), and 4 -> 8 must gain less than 1.1x (measured 0.965x -- it gets
    very slightly WORSE, because past the knee the residual is the
    propagator's, not the mask's).  Those two bars bracket the knee from both
    sides with factors of 1.36 and 1.14 of headroom on the measurements.
    """
    assert inspect.signature(
        apply_aperture).parameters['edge_samples'].default == 4
    e2 = _rs_relative_error(512, edge='gray', edge_samples=2)
    e4 = _rs_relative_error(512, edge='gray', edge_samples=4)
    e8 = _rs_relative_error(512, edge='gray', edge_samples=8)
    assert e2 / e4 >= 1.5, (e2, e4, e2 / e4)
    assert e8 / e4 > 1.0 / 1.1, (e4, e8, e8 / e4)


# ===========================================================================
# 2 -- the way back
# ===========================================================================

@pytest.mark.parametrize('shape,params', [
    ('circular', {'diameter': 1.7e-4}),
    ('annular', {'inner_diameter': 4e-5, 'outer_diameter': 1.7e-4}),
    ('rectangular', {'width_x': 1.33e-4, 'width_y': 0.91e-4}),
])
def test_c1_the_way_back_is_the_binary_mask_bit_for_bit(shape, params):
    """``edge='hard'`` must reproduce the pre-5.49 answer EXACTLY, and the
    pre-5.49 answer is definable here without the parent commit: it is
    ``where(pixel-centre indicator, E, 0)``, which is what the hard branch is.
    Exact-equality assertion; no bar is meaningful.

    (The stronger claim -- that this equals the parent commit's bytes as the
    parent commit itself produced them -- is measured archive-to-archive in
    ``validation/probe_c1_gray_edge/``: 14 of 15 aperture fixtures identical,
    the fifteenth a pre-existing JAX fast/slow-path divergence in the PARENT
    that the same change removes.  A unit test cannot import two trees.)
    """
    N, dx = 96, 2e-6
    rng = np.random.default_rng(3)
    E = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    if shape == 'circular':
        inside = (X ** 2 + Y ** 2) <= (params['diameter'] / 2) ** 2
    elif shape == 'annular':
        h2 = X ** 2 + Y ** 2
        inside = ((h2 >= (params['inner_diameter'] / 2) ** 2)
                  & (h2 <= (params['outer_diameter'] / 2) ** 2))
    else:
        inside = ((np.abs(X) <= params['width_x'] / 2)
                  & (np.abs(Y) <= params['width_y'] / 2))
    expect = np.where(inside, E, 0.0 + 0.0j)
    got = apply_aperture(E, dx, shape, params, edge='hard')
    assert got.tobytes() == expect.tobytes(), shape
    # ... and the default is NOT that, which is the whole point of the flip.
    assert apply_aperture(E, dx, shape, params).tobytes() != expect.tobytes()


def test_c1_the_way_back_survives_every_keyword_the_function_takes():
    """The way back must not be a special case of the plain centred call: a
    decentred, anamorphic, complex64 call with ``edge='hard'`` must still be
    the binary mask, bit for bit."""
    N, dx, dy = 48, 2e-6, 5e-6
    rng = np.random.default_rng(5)
    E = (rng.normal(size=(N, N))
         + 1j * rng.normal(size=(N, N))).astype(np.complex64)
    xc, yc, D = 3.7e-6, -1.3e-6, 5.5e-5
    x = (np.arange(N) - N / 2) * dx
    y = (np.arange(N) - N / 2) * dy
    X, Y = np.meshgrid(x, y)
    inside = ((X - xc) ** 2 + (Y - yc) ** 2) <= (D / 2) ** 2
    expect = np.where(inside, E, np.zeros((), dtype=E.dtype))
    got = apply_aperture(E, dx, 'circular', {'diameter': D},
                         xc=xc, yc=yc, dy=dy, edge='hard')
    assert got.dtype == np.complex64
    assert got.tobytes() == expect.tobytes()


# ===========================================================================
# 3 -- the convergence decision, on both kernels
# ===========================================================================

def _convergence_decision(err_fn):
    """Return the two ladders and assert the DECISION the default rests on.

    Against the closed form, over N = 128 / 256 / 512 / 1024:

      * the grey arm falls at every refinement step and by at least 32x
        overall (pure second order over three halvings is 64x; measured 59.3x
        on the RS spatial kernel and 68.4x on the HF quadrature, so the bar
        sits 1.85x below the worse of the two);
      * the hard arm does NOT fall at every step -- a circle's staircase area
        error is not monotone in the pitch, which is why it has no usable
        order (measured: RS 3.4207e-04 at N = 512 RISES to 5.2718e-04 at
        N = 1024, a 54 % increase; HF 1.1509e-03 rises to 1.7601e-03, 53 %.
        Both builds read those to five figures, so the rise is four decades
        outside the cross-build spread);
      * at the finest grid the grey arm is at least 5x better (measured 21.1x
        RS, 8.3x HF, so 1.67x of headroom on the worse kernel).
    """
    gray = [err_fn(N, edge='gray') for N in LADDER_N]
    hard = [err_fn(N, edge='hard') for N in LADDER_N]
    assert all(gray[i] > gray[i + 1] for i in range(len(gray) - 1)), gray
    assert gray[0] / gray[-1] >= 32.0, gray
    assert any(hard[i] <= hard[i + 1] for i in range(len(hard) - 1)), hard
    assert hard[-1] / gray[-1] >= 5.0, (hard[-1], gray[-1])
    return gray, hard


@pytest.mark.parametrize('kernel', sorted(KERNELS))
def test_c1_the_grey_edge_buys_a_convergence_rate_on_both_spatial_kernels(
        kernel):
    """This is the measurement the default move was decided on, re-run here
    against the closed form on the library's two spatial kernels."""
    gray, hard = _convergence_decision(KERNELS[kernel])
    # the DEFAULT arm rides the grey ladder, bit for bit, not merely closely
    for N in (128, 256):
        assert KERNELS[kernel](N) == KERNELS[kernel](N, edge='gray'), N


def test_c1_the_hard_arm_is_the_arm_without_an_order_not_merely_a_worse_one():
    """Stated separately because it is the qualitative half of the ruling and
    the easiest thing to lose in a re-pinning: the hard arm's failure is not
    a larger constant, it is the absence of a rate.  Measured order between
    successive rows (both builds, 2026-09-20): hard 1.31 / 3.29 / **-0.62**
    (RS) and 1.32 / 3.27 / **-0.61** (HF); grey 2.04 / 2.16 / 1.69 and
    2.05 / 2.03 / 2.02.  The decision: the hard arm has at least one NEGATIVE
    order and the grey arm has none.
    """
    for name, err_fn in sorted(KERNELS.items()):
        gray = [err_fn(N, edge='gray') for N in LADDER_N]
        hard = [err_fn(N, edge='hard') for N in LADDER_N]
        g_ord = [np.log2(gray[i] / gray[i + 1]) for i in range(3)]
        h_ord = [np.log2(hard[i] / hard[i + 1]) for i in range(3)]
        assert min(h_ord) < 0.0, (name, h_ord)
        assert min(g_ord) > 0.0, (name, g_ord)


# ===========================================================================
# 4 -- the rim fractions themselves
# ===========================================================================

#: a rectangular aperture whose four rims land EXACTLY on pixel centres, so
#: the covered fraction of a rim pixel is exact arithmetic on dyadic
#: rationals and needs no bar at all.
_RIM_N, _RIM_DX, _RIM_HALF = 32, 1e-6, 8e-6


def _rim_fractions(build):
    E = np.ones((_RIM_N, _RIM_N), dtype=complex)
    return np.real(np.asarray(build(
        E, _RIM_DX, 'rectangular',
        {'width_x': 2 * _RIM_HALF, 'width_y': 2 * _RIM_HALF})))


def _cell_centred_rim_checks(build):
    """The check, as a function of the builder, so the mutation matrix can
    run it against a mutant lattice."""
    f = _rim_fractions(build)
    # pixel i has centre (i - N/2)*dx, so +8 um is i = 24 and -8 um is i = 8.
    assert f[16, 24] == 0.5, f[16, 24]
    assert f[16, 8] == 0.5, f[16, 8]
    assert f[24, 16] == 0.5 and f[8, 16] == 0.5
    assert f[24, 24] == 0.25 and f[8, 8] == 0.25, (f[24, 24], f[8, 8])
    assert f[16, 16] == 1.0 and f[0, 0] == 0.0
    # A centred aperture's mask is mirror-symmetric -- about the grid's own
    # centre, which for the library's even-N convention
    # ``x = (arange(N) - N/2)*dx`` is index N/2, so index 0 has no partner and
    # the mirror pairs are i <-> N - i for i = 1 .. N-1.  (That asymmetry is
    # the even-N centring convention, not the mask's.)
    assert np.array_equal(f[:, 1:], f[:, :0:-1]), 'x-mirror'
    assert np.array_equal(f[1:, :], f[:0:-1, :]), 'y-mirror'


def test_c1_a_rim_through_a_pixel_centre_reads_exactly_one_half():
    """The sub-sample lattice is the CELL-CENTRED (midpoint) one: its
    ``edge_samples`` offsets are symmetric about the pixel centre, so a rim
    laid exactly on a pixel centre cuts exactly half the sub-samples and the
    pixel reads 0.5 -- and a corner pixel, cut on both axes, reads 0.25.
    Exact arithmetic on dyadic rationals; no bar.

    This is also what makes a centred aperture's mask mirror-symmetric, which
    a lattice anchored on the cell CORNERS is not (it would read 0.75 on the
    +x rim and 0.5 on the -x rim of the same aperture).
    """
    _cell_centred_rim_checks(apply_aperture)


def test_c1_the_grey_mask_is_the_separable_area_average_on_an_axis_aligned_rim():
    """For a rectangular aperture the covered fraction is separable, so the
    2-D mask must equal the outer product of the two 1-D sub-sample counts
    EXACTLY (both sides are k/16 with k an integer, so the product is exact
    in binary floating point at the power-of-two default).  An independent
    1-D construction, not a copy of the 2-D one."""
    f = _rim_fractions(apply_aperture)
    x = (np.arange(_RIM_N) - _RIM_N / 2) * _RIM_DX
    offs = (np.arange(4) + 0.5) / 4 - 0.5
    c = np.zeros(_RIM_N)
    for o in offs:
        c = c + (np.abs(x + o * _RIM_DX) <= _RIM_HALF)
    c = c / 4
    assert np.array_equal(f, np.outer(c, c))


def test_c1_the_grey_fraction_is_bounded_and_blanks_a_blocked_pixel_exactly():
    """Every fraction lies in [0, 1]; fully-open pixels are exactly 1 and
    fully-blocked ones exactly 0 for ANY input value, including non-finite
    ones (the VERIFY-A8 select-don't-scale contract, which the default move
    now puts on the default path)."""
    N, dx = 64, 1e-6
    for fill in (np.nan + 0j, np.inf + 0j, -np.inf + 0j):
        E = np.full((N, N), fill, dtype=np.complex128)
        # ``inf * frac`` in the OPEN region raises numpy's own invalid-value
        # diagnostic; the library's comment at the return says why it is not
        # silenced (it would cost two full-grid temporaries).  Not this
        # test's subject.
        with np.errstate(invalid='ignore'):
            out = apply_aperture(E, dx, 'circular', {'diameter': 2e-5})
        corners = out[[0, 0, -1, -1], [0, -1, 0, -1]]
        assert np.all(corners == 0), (fill, corners)
    frac = np.real(apply_aperture(np.ones((N, N), dtype=complex), dx,
                                  'circular', {'diameter': 2.05e-5}))
    assert frac.min() == 0.0 and frac.max() == 1.0
    assert np.all((frac >= 0.0) & (frac <= 1.0))
    # and there ARE partial pixels -- otherwise the grey path is a no-op here
    assert np.any((frac > 0.0) & (frac < 1.0))


# ===========================================================================
# 5 -- every backend arm of the ONE implementation takes the same default
# ===========================================================================

def test_c1_the_jax_arm_takes_the_same_default_as_the_numpy_arm():
    """``apply_aperture`` is one ``array_namespace``-dispatched body, so the
    JAX arm cannot take a different default -- but "cannot" is a claim about
    the code, and this asserts it about the answer.  No skip: when JAX is
    absent the test asserts that fact, so it never silently drops out."""
    import importlib.util
    if importlib.util.find_spec('jax') is None:
        assert importlib.util.find_spec('jax') is None
        return
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    N, dx = 64, 2e-6
    rng = np.random.default_rng(9)
    E = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
    args = (dx, 'circular', {'diameter': 6.3e-5})
    j_default = np.asarray(apply_aperture(jnp.asarray(E), *args))
    n_default = np.asarray(apply_aperture(E, *args))
    assert j_default.tobytes() == n_default.tobytes()
    j_hard = np.asarray(apply_aperture(jnp.asarray(E), *args, edge='hard'))
    n_hard = np.asarray(apply_aperture(E, *args, edge='hard'))
    assert j_hard.tobytes() == n_hard.tobytes()
    assert j_default.tobytes() != j_hard.tobytes()


def test_c1_the_system_chain_takes_the_same_default_on_both_backends():
    """v5.49.0 routed the JAX chain's ``'aperture'`` element through the same
    ``apply_aperture`` the NumPy chain calls, so one element dict gets one
    answer on either backend -- rim included -- and ``{'edge': 'hard'}`` is
    the way back on either backend.  Before, the JAX kernel carried its own
    pixel-centre indicator, which the flip would have left disagreeing with
    the NumPy chain under a cross-backend bar (5 % of pixels) too loose to
    see it."""
    import importlib.util
    from lumenairy.propagators.system import propagate_through_system
    N, dx = 64, 2e-6
    rng = np.random.default_rng(13)
    E = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
    base = {'type': 'aperture', 'shape': 'circular',
            'params': {'diameter': 6.3e-5}}
    hard_elem = dict(base, edge='hard')
    np_default, _ = propagate_through_system(E, [base], WAVELENGTH, dx=dx)
    np_hard, _ = propagate_through_system(E, [hard_elem], WAVELENGTH, dx=dx)
    direct = apply_aperture(E, dx, 'circular', {'diameter': 6.3e-5})
    direct_hard = apply_aperture(E, dx, 'circular', {'diameter': 6.3e-5},
                                 edge='hard')
    assert np.asarray(np_default).tobytes() == np.asarray(direct).tobytes()
    assert np.asarray(np_hard).tobytes() == np.asarray(direct_hard).tobytes()
    assert np_default.tobytes() != np_hard.tobytes()
    if importlib.util.find_spec('jax') is None:
        assert importlib.util.find_spec('jax') is None
        return
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    from lumenairy.propagators.system import propagate_through_system_jax
    for elem, ref in ((base, np_default), (hard_elem, np_hard)):
        for verbose in (False, True):     # the jit'd kernel and the slow path
            got = np.asarray(propagate_through_system_jax(
                jnp.asarray(E), [elem], WAVELENGTH, dx, verbose=verbose))
            assert got.tobytes() == np.asarray(ref).tobytes(), (elem, verbose)


# ===========================================================================
# 5b -- VERIFY-C1 round 2: what the jit'd kernel must carry, and what it must
#       refuse.  Both closures answer findings in
#       ``docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/
#       VERIFY_WP-C1.md`` -- M3, the one mutation that survived its 2796-id
#       sweep, and D1, the split refusal contract.
# ===========================================================================


def _jax_or_none():
    """JAX, or ``None`` when it is absent.

    No ``pytest.skip``: a skip would silently remove these ids from the gate
    on exactly the runner where JAX is missing, which is the shape
    ``docs/TESTING_STANDARDS.md`` rule 4 names.  Each caller asserts the
    absence instead.
    """
    import importlib.util
    if importlib.util.find_spec('jax') is None:
        return None
    import jax
    jax.config.update('jax_enable_x64', True)
    return jax


@pytest.mark.parametrize('n_sub', [1, 4])
def test_c1_the_jit_kernel_carries_the_elements_edge_samples(n_sub):
    """The jit'd JAX kernel must render the rim the ELEMENT asked for, not the
    library default, and must do it bit for bit with the eager route.

    ``_system_element_signature`` puts ``edge_samples`` in the jit'd kernel's
    STATIC signature.  VERIFY-C1 mutated that one line to ``n_sub = None`` and
    the mutant SURVIVED the whole 61-file / 2796-id sweep on Windows and the
    17-id WP-C1 file on WSL -- while really diverging: with
    ``{'edge_samples': 8}`` the NumPy chain and the eager route read
    ``d3ca82780ee20bbd`` and the jit'd kernel read ``5608c91cd1d4ebae``
    (measured 2026-09-20, both builds).  This id is the one that goes red.

    The two values are the two ends of the contract: ``1`` is exactly the
    pre-5.49 pixel-centre indicator and ``4`` is the shipped default, and the
    last assertion pins that they really are different masks on this fixture,
    so the bit identities above cannot be three routes agreeing on one
    default.

    Bit identity, not a tolerance: the three routes run the same body on the
    same input, so anything short of equal bytes is a divergence -- a mask sum
    is an integer count over ``n_sub**2`` with no BLAS in it, so there is no
    cross-build spread for a bar to sit inside.
    """
    jax = _jax_or_none()
    if jax is None:
        import importlib.util
        assert importlib.util.find_spec('jax') is None
        return
    import jax.numpy as jnp
    from lumenairy.propagators.system import (propagate_through_system,
                                              propagate_through_system_jax)
    N, dx = 96, 1.25e-6
    rng = np.random.default_rng(311)
    E = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
    elements = [{'type': 'aperture', 'shape': 'circular',
                 'params': {'diameter': 9.1e-5}, 'edge_samples': n_sub}]
    npy, _ = propagate_through_system(E, elements, WAVELENGTH, dx=dx)
    jit_ = np.asarray(propagate_through_system_jax(
        jnp.asarray(E), elements, WAVELENGTH, dx))
    eager = np.asarray(propagate_through_system_jax(
        jnp.asarray(E), elements, WAVELENGTH, dx, verbose=True))
    assert jit_.tobytes() == eager.tobytes(), (
        f"the jit'd JAX kernel and the eager JAX route disagree at "
        f"edge_samples={n_sub} -- the element's value is not reaching one "
        f"of them")
    assert jit_.tobytes() == np.asarray(npy).tobytes(), (
        f"the jit'd JAX kernel did not honour edge_samples={n_sub} from the "
        f"element dict (it differs from the NumPy chain on the same dict)")
    other = 4 if n_sub == 1 else 1
    other_el = [dict(elements[0], edge_samples=other)]
    other_npy, _ = propagate_through_system(E, other_el, WAVELENGTH, dx=dx)
    assert np.asarray(other_npy).tobytes() != np.asarray(npy).tobytes(), (
        f"edge_samples {n_sub} and {other} render the same mask on this "
        f"fixture, so the identities above prove nothing")


# VERIFY-C1 D1.  Six element dicts ``apply_aperture`` refuses, plus three
# legal spellings that must NOT be refused.  Measured 2026-09-20 on both
# builds BEFORE the fix: ``{'edge_samples': 2.5}`` and ``{'edge_samples':
# '4'}`` were ACCEPTED by the jit'd route (2.5 silently using 2) and raised
# ``ValueError`` on the NumPy chain and the eager JAX route, because
# ``_system_element_signature`` coerced with ``int()`` / ``str()`` before
# ``apply_aperture`` ever saw the value.
#
# VERIFY-C1-ROUND2 R2 adds the last three rows.  ``edge_samples_bool_false``
# was green for the WRONG reason -- ``int(False) == 0 < 1`` refused it as "not
# positive", not as "a bool" -- so the bool a caller would actually write,
# ``True``, was ACCEPTED as 1, which is bit-for-bit the pre-5.49 hard rim.
# Measured 2026-09-20 on both builds and on all four entry points before the
# round-3 guard: ``True`` and ``numpy.True_`` accepted, ``False`` refused.
_BAD_EDGE_ELEMENTS = [
    ('edge_unknown_string', {'edge': 'soft'}),
    ('edge_none', {'edge': None}),
    ('edge_empty_string', {'edge': ''}),
    ('edge_samples_zero', {'edge_samples': 0}),
    ('edge_samples_negative', {'edge_samples': -2}),
    ('edge_samples_non_integer_float', {'edge_samples': 2.5}),
    ('edge_samples_string', {'edge_samples': '4'}),
    ('edge_samples_bool_false', {'edge_samples': False}),
    ('edge_samples_bool_true', {'edge_samples': True}),
    ('edge_samples_numpy_bool_true', {'edge_samples': np.True_}),
]

_GOOD_EDGE_ELEMENTS = [
    ('edge_hard', {'edge': 'hard'}),
    ('edge_gray', {'edge': 'gray'}),
    ('edge_samples_integral_float', {'edge_samples': 4.0}),
]


def _refusal(fn):
    """``(raised, message)`` for one route, so two routes are compared on the
    VERDICT and on the TEXT, not merely on "something went wrong"."""
    try:
        fn()
    except (ValueError, TypeError) as e:
        return True, f"{type(e).__name__}: {e}"
    return False, ''


@pytest.mark.parametrize('label,bad', _BAD_EDGE_ELEMENTS)
def test_c1_all_three_chain_routes_refuse_a_bad_edge_element_identically(
        label, bad):
    """A chain's ``'edge'`` / ``'edge_samples'`` keys are a CONTRACT, and a
    contract has two sides: one element dict is accepted by all three routes
    or refused by all three, with the SAME diagnostic.

    WP-C1's claim is "one implementation, one default and one way back".
    VERIFY-C1 confirmed that bit for bit of the FIELD and found it untrue of
    the REFUSAL, because the jit'd route read the element through
    ``_system_element_signature``, whose static signature must be hashable and
    so coerced the value first.  The refusal now lives in
    ``elements._validate_edge_kwargs`` -- the same function
    ``apply_aperture``'s own body calls, reached from the one place both
    backends read the element (``_aperture_edge_kwargs``) -- so this id
    asserts ONE guard three times rather than three guards hoped to agree.

    Message equality, not merely verdict equality: two routes raising for
    DIFFERENT reasons satisfy a verdict-only assertion while still disagreeing
    about what is legal.  The message is the library's own text and carries no
    build-dependent quantity, so there is no bar here to be per-build.
    """
    jax = _jax_or_none()
    from lumenairy.propagators.system import (propagate_through_system,
                                              propagate_through_system_jax)
    N, dx = 64, 1.25e-6
    rng = np.random.default_rng(313)
    E = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
    elements = [dict({'type': 'aperture', 'shape': 'circular',
                      'params': {'diameter': 5.3e-5}}, **bad)]

    got = {'numpy': _refusal(
        lambda: propagate_through_system(E, elements, WAVELENGTH, dx=dx))}
    if jax is not None:
        import jax.numpy as jnp
        got['jax_eager'] = _refusal(lambda: propagate_through_system_jax(
            jnp.asarray(E), elements, WAVELENGTH, dx, verbose=True))
        got['jax_jit'] = _refusal(lambda: propagate_through_system_jax(
            jnp.asarray(E), elements, WAVELENGTH, dx))
    else:
        import importlib.util
        assert importlib.util.find_spec('jax') is None

    assert got['numpy'][0], (
        f"{label}: the NumPy chain ACCEPTED an element apply_aperture "
        f"refuses -- the element reader is not validating at all")
    assert len(set(got.values())) == 1, (
        f"{label}: the chain routes disagree about this element: "
        + ' | '.join(f'{k}={v!r}' for k, v in sorted(got.items())))
    assert 'apply_aperture' in got['numpy'][1], got['numpy'][1]


@pytest.mark.parametrize('label,good', _GOOD_EDGE_ELEMENTS)
def test_c1_the_three_chain_routes_accept_the_legal_edge_elements(
        label, good):
    """The other side of the same contract, so the refusal census above cannot
    be satisfied by a guard that refuses everything -- and the three routes
    still answer the legal element identically, byte for byte."""
    jax = _jax_or_none()
    from lumenairy.propagators.system import (propagate_through_system,
                                              propagate_through_system_jax)
    N, dx = 64, 1.25e-6
    rng = np.random.default_rng(313)
    E = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
    elements = [dict({'type': 'aperture', 'shape': 'circular',
                      'params': {'diameter': 5.3e-5}}, **good)]
    npy, _ = propagate_through_system(E, elements, WAVELENGTH, dx=dx)
    assert np.all(np.isfinite(np.asarray(npy)))
    if jax is None:
        import importlib.util
        assert importlib.util.find_spec('jax') is None
        return
    import jax.numpy as jnp
    for verbose in (False, True):
        got = np.asarray(propagate_through_system_jax(
            jnp.asarray(E), elements, WAVELENGTH, dx, verbose=verbose))
        assert got.tobytes() == np.asarray(npy).tobytes(), (label, verbose)


_STOP_PRESCRIPTION = {
    'elements': [
        {'surf_num': 1, 'element_type': 'surface', 'radius': np.inf,
         'glass_after': 'air', 'is_stop': True, 'semi_diameter': 1.2e-3,
         'comment': 'STOP'},
        {'surf_num': 2, 'element_type': 'surface', 'radius': 0.05,
         'glass_after': 'N-BK7', 'semi_diameter': 2e-3},
        {'surf_num': 3, 'element_type': 'surface', 'radius': -0.05,
         'glass_after': 'air', 'semi_diameter': 2e-3},
    ],
    'all_thicknesses': [2e-3, 3e-3, 20e-3],
    'aperture_diameter': 2.4e-3,
}


def test_c1_evaluates_way_back_is_one_keyword_and_reaches_the_stop():
    """``lumenairy.evaluate`` renders every ``is_stop=True`` surface as an
    ``'aperture'`` element, so its answer moved with the default -- and
    ``aperture_edge='hard'`` is its way back, the one keyword the Migration
    table promises for every entry point in it.

    VERIFY-C1 D4.  Before round 2 this entry point had no route back a caller
    could reach: ``evaluate`` took no rim argument and built its element list
    internally, so the only way to the old answer was the PRIVATE
    ``_prescription_to_elements`` plus a hand-driven chain.

    MEASURED 2026-09-20 archive-to-archive, ``git archive 49ddf4bd`` against
    this tree, both builds
    (``validation/probe_verify_c1/d4_evaluate_*.json``):

        Windows  pre-5.49 default  e7b1f67b9b19d547
                 5.49 default      59115e8eb2b2b0d0
                 aperture_edge='hard'  e7b1f67b9b19d547   <- byte-identical
        WSL      0b97c205be347dfa / 193bf1d920e2ac88 / 0b97c205be347dfa

    The assertions here are the same decisions inside one tree: bit identity
    against the hand-built hard chain, non-identity against the default (so
    the keyword is not accepted-and-ignored), and ``edge_samples=1`` landing
    on the hard answer, which is what proves the SECOND keyword reaches the
    element too.  Bytes, not tolerances.
    """
    import lumenairy as la
    from lumenairy.propagators.system import (_prescription_to_elements,
                                              propagate_through_system)
    src = la.Source.gaussian(N=128, dx=40e-6, wavelength=633e-9, w0=1.0e-3)

    emitted = _prescription_to_elements(_STOP_PRESCRIPTION)
    apertures = [e for e in emitted if e.get('type') == 'aperture']
    assert len(apertures) == 1, emitted
    assert 'edge' not in apertures[0] and 'edge_samples' not in apertures[0], (
        "the emitted element must name no rim when the caller named none, "
        "or it pins today's default into tomorrow's answer")

    hand_hard = [dict(e, edge='hard') if e.get('type') == 'aperture' else e
                 for e in emitted]
    out = propagate_through_system(np.asarray(src.E), hand_hard, 633e-9,
                                   dx=40e-6)
    hard = np.asarray(out[0] if isinstance(out, tuple) else out)

    default = np.asarray(la.evaluate(_STOP_PRESCRIPTION, src).field)
    back = np.asarray(la.evaluate(_STOP_PRESCRIPTION, src,
                                  aperture_edge='hard').field)
    gray1 = np.asarray(la.evaluate(_STOP_PRESCRIPTION, src,
                                   aperture_edge='gray',
                                   aperture_edge_samples=1).field)
    assert back.tobytes() == hard.tobytes(), (
        "evaluate(aperture_edge='hard') is not the pre-5.49 answer")
    assert back.tobytes() != default.tobytes(), (
        "evaluate's answer does not move with the rim -- then the keyword is "
        "accepted and ignored, or the STOP surface is not being rendered")
    assert gray1.tobytes() == hard.tobytes(), (
        "aperture_edge_samples is not reaching the element -- n_sub = 1 IS "
        "the pixel-centre indicator")

    # The rim keywords are refused by evaluate with apply_aperture's own
    # message, before any propagation runs -- the same guard, not a third.
    for kw in ({'aperture_edge': 'soft'}, {'aperture_edge_samples': 2.5},
               {'aperture_edge_samples': 0}):
        with pytest.raises((ValueError, TypeError), match='apply_aperture'):
            la.evaluate(_STOP_PRESCRIPTION, src, **kw)


# ===========================================================================
# 6 -- the mutation matrix
# ===========================================================================
#
# Each row names a mutation that would silently undo part of this work
# package, and the NAMED test above that catches it.  The rows run the named
# test's own check function against a mutant and assert that it FAILS; a row
# that passed under mutation would mean the named test is not load-bearing.


def _mutant_default_back_to_hard(*args, **kw):
    """Mutation 1: the default silently reverts to the pre-5.49 staircase."""
    kw.setdefault('edge', 'hard')
    return apply_aperture(*args, **kw)


_mutant_default_back_to_hard.__signature__ = inspect.Signature([
    p if p.name != 'edge' else p.replace(default='hard')
    for p in inspect.signature(apply_aperture).parameters.values()])


def _mutant_corner_lattice(E, dx, shape, params, edge='gray', edge_samples=4,
                           dy=None, xc=0.0, yc=0.0):
    """Mutation 3: the grey mask's sub-sample lattice is anchored on the cell
    CORNERS instead of being centred on the pixel, so every boundary fraction
    is biased by half a sub-cell and the mask loses its mirror symmetry."""
    dy = dx if dy is None else dy
    Ny, Nx = E.shape
    x = (np.arange(Nx) - Nx / 2) * dx
    y = (np.arange(Ny) - Ny / 2) * dy
    offsets = np.arange(edge_samples) / edge_samples - 0.5      # the mutation
    frac = np.zeros((Ny, Nx))
    for oy in offsets:
        for ox in offsets:
            X, Y = np.meshgrid(x + ox * dx, y + oy * dy)
            if shape == 'rectangular':
                m = ((np.abs(X - xc) <= params['width_x'] / 2)
                     & (np.abs(Y - yc) <= params['width_y'] / 2))
            else:
                h2 = (X - xc) ** 2 + (Y - yc) ** 2
                m = h2 <= (params['diameter'] / 2) ** 2
            frac = frac + m
    frac = frac / (edge_samples * edge_samples)
    return np.where(frac > 0, E * frac, 0.0 + 0.0j)


def test_c1_mutation_the_default_silently_back_to_hard_is_caught():
    """Caught by
    ``test_c1_the_default_edge_is_gray_from_the_signature_and_from_a_call``."""
    with pytest.raises(AssertionError):
        _default_is_gray(_mutant_default_back_to_hard)


def test_c1_mutation_edge_samples_moved_off_the_knee_is_caught():
    """Caught by
    ``test_c1_edge_samples_default_is_four_and_four_is_the_measured_knee``.

    Both directions are mutations: 1 collapses the grey mask back to the
    pixel-centre indicator (measured identical to the hard reading,
    3.4207e-04 at N = 512), and 8 pays four times the boundary work for a
    reading that is very slightly worse.  The named test's signature half
    catches either; this row exercises it by patching the default in place.
    """
    original = apply_aperture.__defaults__
    for bad in (1, 2, 8, 16):
        patched = tuple(bad if v == 4 else v for v in original)
        assert patched != original, original
        apply_aperture.__defaults__ = patched
        try:
            with pytest.raises(AssertionError):
                assert inspect.signature(
                    apply_aperture).parameters['edge_samples'].default == 4
        finally:
            apply_aperture.__defaults__ = original
    assert apply_aperture.__defaults__ == original
    # and the ladder half still reads the knee where it was measured
    assert (_rs_relative_error(512, edge='gray', edge_samples=1)
            == _rs_relative_error(512, edge='hard')), (
        "n_sub = 1 IS the pixel-centre indicator; if that stops being true "
        "the knee ladder is measuring something else")


def test_c1_mutation_a_wrong_grey_boundary_fraction_is_caught():
    """Caught by ``test_c1_a_rim_through_a_pixel_centre_reads_exactly_one_half``
    (the mutant reads 0.75 on the +x rim, 0.5 on the -x rim and 0.5625 in the
    corner, so it fails the half, the quarter AND the mirror symmetry)."""
    with pytest.raises(AssertionError):
        _cell_centred_rim_checks(_mutant_corner_lattice)
    # ... and by the separable-area identity, independently
    f = np.real(_mutant_corner_lattice(
        np.ones((_RIM_N, _RIM_N), dtype=complex), _RIM_DX, 'rectangular',
        {'width_x': 2 * _RIM_HALF, 'width_y': 2 * _RIM_HALF}))
    x = (np.arange(_RIM_N) - _RIM_N / 2) * _RIM_DX
    offs = (np.arange(4) + 0.5) / 4 - 0.5
    c = np.zeros(_RIM_N)
    for o in offs:
        c = c + (np.abs(x + o * _RIM_DX) <= _RIM_HALF)
    assert not np.array_equal(f, np.outer(c / 4, c / 4))
