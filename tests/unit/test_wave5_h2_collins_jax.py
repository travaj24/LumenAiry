"""Wave-5 hygiene-2, H2-2 (audit item 18) -- ``_collins_transport`` on JAX.

Before 5.48.0 the Collins transport was NumPy-only, not by design but by
plumbing: it called ``np.asarray`` / ``np.ascontiguousarray(..., dtype=
np.complex128)`` and then ``_fft2``, ``_collins_angle_support``,
``_collins_exact_kernel_correction``, ``_collins_space_support``,
``_collins_sampling_stats`` and ``_bluestein_centred_2d``, and the only backend
any of that could reach was NumPy's.  The fix is ONE ``xp`` threaded through
the chain -- no ``_jax`` twin of anything -- using the module's own
``(xp, is_jax, bld)`` triple (:func:`~lumenairy.propagators.carrier.
_backend_of`), its own device-move helper (``_to_dev``), its own
``exp(i*phase)`` builder (``_tf_phase_to_H``, already shared by
``_exact_tf_2d_xp`` and ``_fresnel_tf_2d_xp``) and one new FFT-pair selector
(``_fft2_pair``) that IS ``lumenairy.backend.fft2``'s dispatch.

THE BARS, AND WHERE THEY COME FROM.

Nothing here compares JAX to NumPy against a number somebody liked.  The bar is
MEASURED FIRST, in :func:`_fft_spread_bar`, from the only thing that can
legitimately differ: the two backends' FFTs.  NumPy's path is pyFFTW or
scipy.fft's pocketfft; JAX's is XLA's own transform.  They are different
algorithms over the same arithmetic, so they differ in the last bits, and the
transport applies two of them (the measurement FFT and the chirp-Z's pair) plus
the Bluestein convolution's three.  The bar is the measured single-transform
spread times a stated factor for the chain depth, re-measured on the running
build at the FIXTURE's own shape -- so it tracks a numpy release, an XLA
release, or a change of FFT backend rather than pinning today's.

``jax_enable_x64`` is forced on: in float32 the comparison would be measuring
JAX's default dtype policy, not the port.

WHAT THIS FILE DOES NOT GATE, AND WHERE IT IS GATED INSTEAD (V-D14).  Two of
H2-2's threadings are invisible to every VALUE test here, and that was measured
rather than argued: ``_backend_of`` returns ``bld = np`` for NumPy AND for JAX,
so only CuPy would see the ``bld`` threading in ``_collins_axis_chirp``, and
``np.asarray`` / ``np.ascontiguousarray`` agree on VALUES for every input, so
``_as_c_order``'s contiguity is unobservable too.  Both mutations leave all of
this file's ids and all 93 of the author's bit-identity keys unchanged.  They
are closed STRUCTURALLY, and by someone else's file, which is why they are not
re-gated here -- one gate per claim:
``tests/unit/test_verify_wave5_hyg2.py::
test_the_axis_chirp_builds_on_the_bld_it_is_handed`` (a recording namespace)
and ``::test_as_c_order_makes_the_numpy_path_contiguous`` (a layout assertion).
"""
from __future__ import annotations

import numpy as np
import pytest

import lumenairy.propagators.carrier as CA

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

WL = 633e-9
N = 64
DX = 8e-6
R_IN = -0.05
Z = 5e-3
R_REF = -0.045


@pytest.fixture(scope="module", autouse=True)
def _x64():
    """float64 everywhere.  A float32 JAX would make every bar below a
    statement about dtype policy instead of about the port."""
    prev = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _gauss(n=N, dx=DX, w=60e-6, dtype=np.complex128):
    ax = (np.arange(n) - n // 2) * dx
    X, Y = np.meshgrid(ax, ax)
    return np.exp(-(X ** 2 + Y ** 2) / (w * w)).astype(dtype)


def _rel(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def _fft_spread_bar(env, chain_depth=6.0):
    """The bar, MEASURED on this build, this shape, this pair of FFTs.

    One forward transform of the fixture through each backend's own 2-D FFT,
    compared relatively.  That is the irreducible per-transform disagreement.
    ``chain_depth`` is an UPPER BOUND on how many the transport applies: the
    measurement transform (1), the exact-kernel correction's inverse (1, on the
    'exact' arm only) and the Bluestein primitive's three (two 2-D FFTs plus
    the chirp-kernel transform) -- five on the 2-D arm, and six on the
    separable arm, which spends three 1-D passes per axis instead.  Round-off
    through a chain of unitary transforms adds at worst linearly, so
    ``chain_depth`` times the single-transform spread is a bound and not a fit.

    A floor of ``32 * eps`` is added because a spread measured as exactly zero
    (the two backends agreeing bit for bit on a small easy case, which does
    happen) would otherwise make the bar zero and the test a coin toss.
    """
    from lumenairy.propagators.fft_infra import _fft2
    a = np.asarray(_fft2(np.ascontiguousarray(env, dtype=np.complex128)))
    b = np.asarray(jnp.fft.fft2(jnp.asarray(env, dtype=jnp.complex128)))
    spread = float(np.linalg.norm(a - b) / np.linalg.norm(a))
    return max(chain_depth * spread, 32.0 * float(np.finfo(np.float64).eps))


# ===========================================================================
# 1. The chain really is ONE implementation
# ===========================================================================

def test_the_fft_pair_is_the_librarys_own_dispatcher_not_a_second_copy():
    """``_fft2_pair`` must hand back the library's existing callables, not
    wrap them.

    Two claims.  (a) On the NumPy side the callables are ``fft_infra._fft2`` /
    ``_ifft2`` BY IDENTITY -- and identity is the point, not equivalence:
    ``_bluestein_2d`` keys its chirp-kernel FFT cache on
    ``fft2 is fft_infra._fft2``, so a wrapper would silently disable that cache
    for every Collins leg.  (b) On both sides the callable agrees with
    ``lumenairy.backend.fft2``, which is the library's declared backend FFT
    entry point, so there is one dispatch and not two.
    """
    from lumenairy.backend import fft2 as backend_fft2
    from lumenairy.propagators.fft_infra import _fft2, _ifft2

    f, i = CA._fft2_pair(np, False)
    assert f is _fft2 and i is _ifft2

    env = _gauss()
    assert np.array_equal(np.ascontiguousarray(f(np.ascontiguousarray(
        env, dtype=np.complex128))).view(np.float64),
        np.ascontiguousarray(backend_fft2(np.ascontiguousarray(
            env, dtype=np.complex128))).view(np.float64))

    fj, ij = CA._fft2_pair(jnp, True)
    ej = jnp.asarray(env, dtype=jnp.complex128)
    assert np.array_equal(np.asarray(fj(ej)), np.asarray(backend_fft2(ej)))
    assert np.array_equal(np.asarray(ij(ej)),
                          np.asarray(jnp.fft.ifft2(ej)))


def test_no_jax_twin_of_the_collins_chain_exists():
    """The rule this package is held to: ONE ``xp``-parametrised
    implementation per kernel, no per-flavour copies.

    A structural inventory, so a future ``_collins_transport_jax`` is caught by
    the gate rather than by a reviewer.  Every private Collins name in the
    module is listed and none may carry a backend suffix; and the module must
    not import a ``*_jax*`` sibling for this chain.

    The source half of the check reads the module's CODE, docstrings stripped
    (2026-09-19).  A raw substring search over the whole file fired on a
    docstring that merely CITED this test file by name -- a false positive on
    a cross-reference, which is the opposite of what the gate is for.  Stripped
    of docstrings the search still covers every import, every definition and
    every call, which is where a per-flavour twin would have to appear.
    """
    import ast
    import pathlib
    names = [n for n in dir(CA) if n.startswith('_collins')]
    assert names, "the Collins chain vanished; this inventory is stale"
    for n in names:
        low = n.lower()
        for suffix in ('_jax', '_jnp', '_cupy', '_cp', '_np', '_numpy'):
            assert not low.endswith(suffix), (
                f"{n} looks like a per-backend copy of a Collins helper; the "
                f"chain is meant to be one xp-parametrised implementation")
    src = pathlib.Path(CA.__file__).read_text(encoding='cp1252')
    tree = ast.parse(src)
    code = '\n'.join(
        _body_source(src, n) for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef))
    # module-level statements too, minus the module docstring
    code += '\n' + '\n'.join(
        ast.get_source_segment(src, s) or ''
        for s in tree.body
        if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant)
                and isinstance(s.value.value, str))
        and not isinstance(s, ast.FunctionDef))
    assert '_collins_jax' not in code and 'collins_jax' not in code


def _body_source(src, node):
    """A function's source with its DOCSTRING removed.

    A census that reads docstrings counts a cross-reference as a second
    implementation.  Every statement is taken through
    ``ast.get_source_segment`` except a leading string expression.
    """
    import ast
    body = node.body
    if (body and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)):
        body = body[1:]
    return '\n'.join(ast.get_source_segment(src, s) or '' for s in body)


def test_the_exact_dispersion_is_written_once():
    """ONE kernel, ONE implementation -- the campaign's standing rule, gated.

    MEASURED 2026-09-19 (VERIFY-WAVE5-HYGIENE2 V-D22): THREE transcriptions of
    ``sqrt(k^2 - |k s + q|^2) - k N + (s.q)/N`` lived in ``carrier.py`` --
    ``_exact_tf_2d_xp`` (``xp``-parametrised), ``_exact_envelope_tf_step``
    (NumPy-only, and written twice inside itself), and
    ``_collins_exact_kernel_correction`` (``(xp, is_jax, bld)``-parametrised
    by H2-2) -- with the same ``|s|^2 < 1`` guard written three times under
    three different error prefixes.  They are one kernel with two uses: the
    first two add the piston ``k z``, the third subtracts the paraxial
    ``|q|^2/(2k)`` instead.

    Three independent tokens are censused, not one, because a future author
    could split any one of them off on its own: the ``q = 0`` subtraction
    (``root0``), the radical's shifted-frequency form, and the evanescent
    guard.  Docstrings are stripped, so a cross-reference is not counted as an
    implementation.

    This is the gate that had to LAND WITH the consolidation.  The only thing
    that caught a conjugated exact kernel before it was
    ``test_audit2609_b4_collins_transport.py::TestSameTheorem``, an AGREEMENT
    test between two of these transcriptions -- and MEASURED on the
    consolidated tree, a sign flip of the single kernel leaves that file
    reading 9 passed, because both transports now move together.  The sign is
    pinned directly instead, in
    ``test_wave5_h2_near_focus_table.py::
    test_the_measured_departure_is_the_quartic_times_one_constant``.
    """
    import ast
    import pathlib
    src = pathlib.Path(CA.__file__).read_text(encoding='cp1252')
    tree = ast.parse(src)
    defs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    assert len(defs) > 100, (
        f"only {len(defs)} function definitions parsed out of carrier.py; the "
        f"census is not reading the module")

    def owners(token):
        return sorted({n.name for n in defs
                       if token in _body_source(src, n)})

    for token, what in (
            ('root0', 'the q = 0 subtraction k N'),
            ('ax * ax + ay * ay', 'the shifted-frequency radical'),
            ('s2 < 1.0', 'the evanescent-carrier guard')):
        got = owners(token)
        assert got == ['_exact_dispersion_phase'], (
            f"{what} ({token!r}) is implemented in {got} -- it must live in "
            f"_exact_dispersion_phase and nowhere else.  ONE "
            f"xp-parametrised implementation per numerical kernel is this "
            f"campaign's standing rule, and the specific cost of breaking it "
            f"here is measured: with two copies, the only guard on the "
            f"kernel's SIGN was an agreement test between them.")

    # ... and the one implementation is reached from all three sites.
    for site in ('_exact_tf_2d_xp', '_exact_envelope_tf_step',
                 '_collins_exact_kernel_correction'):
        node = next(n for n in defs if n.name == site)
        assert '_exact_dispersion_phase(' in _body_source(src, node), (
            f"{site} no longer calls _exact_dispersion_phase; a fourth "
            f"transcription has appeared, or the site was deleted")


@pytest.mark.parametrize("name", ('_collins_exact_kernel_correction',
                                  '_collins_axis_chirp'))
def test_the_threaded_helpers_default_to_the_numpy_build(name):
    """Every helper that gained a backend argument defaults to the historical
    NumPy build, which is why every existing caller and every existing test
    kept working without being touched."""
    import inspect
    sig = inspect.signature(getattr(CA, name))
    if name == '_collins_axis_chirp':
        assert sig.parameters['bld'].default is np
    else:
        assert sig.parameters['xp'].default is np
        assert sig.parameters['is_jax'].default is False
        assert sig.parameters['bld'].default is np


# ===========================================================================
# 2. The JAX path agrees with NumPy to the FFTs' own spread
# ===========================================================================

@pytest.mark.parametrize("gap_kernel", ('auto', 'fresnel', 'exact'))
def test_the_private_transport_agrees_across_backends(gap_kernel):
    """``_collins_transport`` itself, both backends, both kernels.

    The bar is measured on this build first (:func:`_fft_spread_bar`), and the
    assertion carries BOTH sides of it: the disagreement must be below the
    bar, and the bar must be below the smallest real signal -- which here is
    the exact-kernel refinement's own departure from the paraxial kernel,
    measured in the same call.  If the two ever met, the test would be reading
    noise and says so instead of passing.
    """
    env = _gauss()
    bar = _fft_spread_bar(env)
    kw = dict(dx_out=DX, dy_out=DX, N_out_x=N, N_out_y=N, R_ref=R_REF,
              gap_kernel=gap_kernel, on_collins_sampling='ignore')
    a = np.asarray(CA._collins_transport(env, R_IN, Z, WL, DX, DX, **kw))
    b = np.asarray(CA._collins_transport(
        jnp.asarray(env, dtype=jnp.complex128), R_IN, Z, WL, DX, DX, **kw))
    got = _rel(b, a)
    assert got < bar, (
        f"JAX vs NumPy {got:.3e} exceeds the measured FFT-spread bar "
        f"{bar:.3e} at gap_kernel={gap_kernel!r}")
    # the two-sided half: the bar has to be below something real
    fres = np.asarray(CA._collins_transport(
        env, R_IN, Z, WL, DX, DX, **dict(kw, gap_kernel='fresnel')))
    exact = np.asarray(CA._collins_transport(
        env, R_IN, Z, WL, DX, DX, **dict(kw, gap_kernel='exact')))
    signal = _rel(exact, fres)
    assert bar < signal / 10.0, (
        f"the measured bar {bar:.3e} is not a decade below the smallest real "
        f"signal on this fixture ({signal:.3e} between the exact and paraxial "
        f"kernels) -- the comparison is testing noise")


@pytest.mark.parametrize("transport", ('collins', 'sziklas'))
def test_the_public_entry_agrees_across_backends(transport):
    """The public entry point, so the claim is about what a caller gets.

    THE FIRST THING THIS ASSERTS IS THAT IT IS COMPARING TWO BACKENDS.  Until
    2026-09-19 the ``collins`` arm of this id compared NumPy to NumPy and
    could not have failed if ``_collins_transport``'s JAX arm had been
    deleted: ``_collins_carrier_leg`` opened with ``env_a = np.asarray(env)``,
    so the public leg demoted an eager JAX array to host NumPy and returned a
    ``numpy.ndarray`` that was BITWISE equal to the NumPy arm
    (VERIFY-WAVE5-HYGIENE2 V-D3).  A parity test whose two arms are the same
    array module is not a parity test, and nothing in the old assertion said
    so.  The type check below is therefore the PREMISE, not a nicety.
    """
    env = _gauss()
    bar = _fft_spread_bar(env)
    kw = dict(wavelength=WL, dx=DX, transport=transport,
              gap_kernel='fresnel')
    if transport == 'collins':
        kw['on_collins_sampling'] = 'ignore'
    a = CA.propagate_carrier_referenced(env, R_IN, Z, **kw)
    b = CA.propagate_carrier_referenced(
        jnp.asarray(env, dtype=jnp.complex128), R_IN, Z, **kw)

    # PREMISE: the two arms really are two backends.
    assert isinstance(a.env, np.ndarray), (
        f"the NumPy arm returned {type(a.env).__name__}, not an ndarray")
    assert not isinstance(b.env, np.ndarray), (
        f"transport={transport!r}: the JAX arm came back as "
        f"{type(b.env).__name__}, i.e. the public entry DEMOTED the caller's "
        f"array to host NumPy.  The comparison below would then be NumPy "
        f"against NumPy and would pass with the JAX arm of the transport "
        f"deleted -- which is exactly what it used to do.")
    assert type(b.env).__module__.split('.')[0] in ('jax', 'jaxlib'), (
        f"the JAX arm came back as {type(b.env).__module__}."
        f"{type(b.env).__name__}")

    got = _rel(b.env, a.env)
    assert got < bar, (
        f"transport={transport!r}: JAX vs NumPy {got:.3e} exceeds the measured "
        f"FFT-spread bar {bar:.3e}")
    # Two-sided: the bar has to sit below something real, or the agreement is
    # a statement about noise.  The signal here is the leg's own answer scale.
    assert bar < 1e-3, (
        f"the measured FFT-spread bar {bar:.3e} is not far below unity; the "
        f"relative comparison above has stopped discriminating")
    assert float(b.R) == float(a.R)
    assert float(b.dx) == float(a.dx)


def test_the_public_collins_leg_refuses_a_trace_by_name():
    """Under a trace the PUBLIC leg refuses with a designed ``ValueError``.

    It cannot do what ``_collins_transport`` does and offer a spelling that
    runs: the leg resolves its own OUTPUT LATTICE and its own QUADRATURE from
    the envelope's measured phase-space box, so there is a third measured
    decision with no caller-supplied alternative.  Before 2026-09-19 this
    raised a raw ``TracerArrayConversionError`` from ``np.asarray`` -- naming
    neither the leg nor a remedy -- EVEN when the caller passed both of the
    transport's documented ways out (V-D3).
    """
    amp = jnp.asarray(np.real(_gauss()))

    def merit(a):
        out = CA.propagate_carrier_referenced(
            a.astype(jnp.complex128), R_IN, Z, wavelength=WL, dx=DX,
            transport='collins', gap_kernel='fresnel',
            on_collins_sampling='ignore')
        return jnp.sum(jnp.abs(out.env) ** 2)

    with pytest.raises(ValueError) as exc:
        jax.grad(merit)(amp)
    msg = str(exc.value)
    assert 'Tracer' in msg
    assert '_collins_transport' in msg, (
        "the refusal must name the call that CAN be traced; a refusal with no "
        "way forward is the shape this repository refuses to ship")
    for needle in ('dx_out', 'gap_kernel', 'on_collins_sampling'):
        assert needle in msg, f"the refusal does not name {needle}"
    # ... and it is the DESIGNED error, not JAX's own concretisation failure.
    assert 'TracerArrayConversionError' not in type(exc.value).__name__


def test_the_astigmatic_arm_crosses_backends_too():
    """The per-axis arithmetic, which is a different code path in the same
    function (two ``A``, two ``D``, two chirps)."""
    env = _gauss()
    bar = _fft_spread_bar(env)
    kw = dict(dx_out=DX, dy_out=DX, N_out_x=N, N_out_y=N,
              R_ref=(-0.045, -0.07), gap_kernel='fresnel',
              on_collins_sampling='ignore')
    a = np.asarray(CA._collins_transport(env, (-0.05, -0.08), Z, WL, DX, DX,
                                         **kw))
    b = np.asarray(CA._collins_transport(
        jnp.asarray(env, dtype=jnp.complex128), (-0.05, -0.08), Z, WL, DX,
        DX, **kw))
    assert _rel(b, a) < bar


def test_a_tilted_exact_kernel_crosses_backends():
    """The tilt expansion is the one branch inside the exact-kernel correction
    that only runs when ``(L, M) != 0``, so it needs its own arm."""
    env = _gauss()
    bar = _fft_spread_bar(env)
    kw = dict(dx_out=DX, dy_out=DX, N_out_x=N, N_out_y=N, R_ref=R_REF,
              gap_kernel='exact', tilt=(0.12, 0.03),
              on_collins_sampling='ignore')
    a = np.asarray(CA._collins_transport(env, R_IN, Z, WL, DX, DX, **kw))
    b = np.asarray(CA._collins_transport(
        jnp.asarray(env, dtype=jnp.complex128), R_IN, Z, WL, DX, DX, **kw))
    assert _rel(b, a) < bar


def test_the_backend_does_not_silently_upcast_a_complex64_envelope():
    """The v4.14.1 no-silent-upcast contract, on the ported path."""
    env = _gauss(dtype=np.complex64)
    kw = dict(dx_out=DX, dy_out=DX, N_out_x=N, N_out_y=N, R_ref=R_REF,
              gap_kernel='fresnel', on_collins_sampling='ignore')
    assert CA._collins_transport(env, R_IN, Z, WL, DX, DX,
                                 **kw).dtype == np.complex64
    out = CA._collins_transport(jnp.asarray(env, dtype=jnp.complex64), R_IN,
                                Z, WL, DX, DX, **kw)
    assert out.dtype == jnp.complex64


def test_the_numpy_path_is_untouched_by_the_backend_plumbing():
    """The regression that matters most: the NumPy answer must be what it was.

    Byte equality against the chain spelled out by hand -- pre-chirp, centred
    chirp-Z, post-chirp, prefactor -- using the module's own helpers with
    their NumPy defaults.  If the plumbing ever changed a NumPy expression,
    this hand-spelling and the function would part company.

    (The archive-to-archive proof against 5.47.0 is
    ``validation/probe_wave5_hyg2/colbit_{win,wsl}_compare.json``: 84 keys,
    identical on both builds.  This is its in-suite counterpart.)
    """
    from lumenairy.propagators._bluestein import _bluestein_centred_2d
    from lumenairy.propagators.fft_infra import _fft2, _ifft2

    env = _gauss()
    Ax, B, Cx, Dx = CA._collins_envelope_abcd(R_IN, Z, R_REF)
    cdt = env.dtype
    k = 2.0 * np.pi / WL
    g = env * CA._collins_axis_chirp(N, DX, WL, B / Ax, dtype=cdt)[None, :]
    g = g * CA._collins_axis_chirp(N, DX, WL, B / Ax, dtype=cdt)[:, None]
    alpha = float(DX) * float(DX) / (WL * B)
    G = _bluestein_centred_2d(
        np.ascontiguousarray(g, dtype=cdt), alpha, alpha, N, N,
        k_centre_out_x=N / 2.0, k_centre_out_y=N / 2.0,
        sign=-1, xp=np, fft2=_fft2, ifft2=_ifft2, target_cdtype=cdt,
        separable=bool(CA._EXACT_READOUT_SEPARABLE_BLUESTEIN))
    G = G * CA._collins_axis_chirp(N, DX, WL, B / Dx, offset=-0.0,
                                   dtype=cdt)[None, :]
    G = G * CA._collins_axis_chirp(N, DX, WL, B / Dx, offset=-0.0,
                                   dtype=cdt)[:, None]
    want = complex(np.exp(1j * k * B) * DX * DX / (1j * WL * B)) * G

    got = CA._collins_transport(
        env, R_IN, Z, WL, DX, DX, dx_out=DX, dy_out=DX, N_out_x=N, N_out_y=N,
        R_ref=R_REF, gap_kernel='fresnel', on_collins_sampling='ignore')
    assert np.array_equal(np.ascontiguousarray(got).view(np.float64),
                          np.ascontiguousarray(want).view(np.float64))


# ===========================================================================
# 3. Under a trace: the two measured decisions are refused, not defaulted
# ===========================================================================

def _merit_factory(**kw):
    def merit(amp):
        env = amp.astype(jnp.complex128)
        out = CA._collins_transport(env, R_IN, Z, WL, DX, DX,
                                    dx_out=DX, dy_out=DX, N_out_x=N,
                                    N_out_y=N, R_ref=R_REF, **kw)
        return jnp.sum(jnp.abs(out) ** 2)
    return merit


@pytest.mark.parametrize("kw,needle", (
    (dict(gap_kernel='auto', on_collins_sampling='ignore'), "gap_kernel"),
    (dict(gap_kernel='fresnel', on_collins_sampling='warn'),
     "on_collins_sampling"),
    (dict(gap_kernel='auto', on_collins_sampling='warn'), "gap_kernel"),
))
def test_a_traced_envelope_refuses_the_decisions_it_cannot_measure(kw, needle):
    """A Tracer has no entries, so the ``k4`` kernel resolution and the Kelly
    guard cannot be evaluated.  Both are REFUSED, loudly, with the spelling
    that takes each decision named in the message.

    Not defaulted: silently taking 'fresnel' under a trace would be the same
    shape as the ``gap_kernel`` typo-falls-through defect (D4) that the
    vocabulary gate exists to remove, and a conservative grid-edge guard would
    refuse legs whose measured departure from the analytic ABCD field is
    5.6e-08 of peak.
    """
    amp = jnp.asarray(np.real(_gauss()))
    with pytest.raises(ValueError, match="Tracer") as exc:
        jax.grad(_merit_factory(**kw))(amp)
    assert needle in str(exc.value)
    assert "trace-safe" in str(exc.value)


def test_an_astigmatic_auto_is_accepted_under_a_trace_because_it_measures_nothing():
    """The rule is "refuses unless ``gap_kernel='fresnel'``"; the CODE is
    narrower, and the difference is gated rather than left to a reader
    (VERIFY-WAVE5-HYGIENE2 V-D11).

    An ASTIGMATIC carrier has no exact-kernel arm at all -- the kernel clause
    is guarded by ``Ax == Ay``, because ``sqrt(k^2 - qx^2 - qy^2)`` does not
    separate and the two axes have different ``B/A`` -- so ``'auto'`` takes no
    MEASURED decision there and is accepted.  Numerically benign: the eager
    path resolves the same configuration to ``'fresnel'`` too, which is the
    second half of this id.  MEASURED 2026-09-19: max|grad| = 2.0008 through
    ``jax.grad``, and the eager call reports ``kernel == 'fresnel'``.
    """
    amp = jnp.asarray(np.real(_gauss()))
    kw = dict(dx_out=DX, dy_out=DX, N_out_x=N, N_out_y=N,
              R_ref=(-0.045, -0.07), gap_kernel='auto',
              on_collins_sampling='ignore')

    def merit(a):
        out = CA._collins_transport(a.astype(jnp.complex128),
                                    (-0.05, -0.08), Z, WL, DX, DX, **kw)
        return jnp.sum(jnp.abs(out) ** 2)

    g = np.asarray(jax.grad(merit)(amp))
    assert np.all(np.isfinite(g)) and float(np.max(np.abs(g))) > 0.0, (
        "the astigmatic 'auto' arm did not run under a trace; if it now "
        "refuses, the docstring's exception paragraph is what is stale")
    # ... and the decision it would have taken eagerly is 'fresnel' anyway,
    # which is why accepting it is benign rather than a silent default.
    st = {}
    CA._collins_transport(_gauss(), (-0.05, -0.08), Z, WL, DX, DX,
                          stats_out=st, **kw)
    assert st['kernel'] == 'fresnel', (
        f"the eager astigmatic 'auto' resolved to {st['kernel']!r}; the traced "
        f"arm accepts 'auto' only because there is no exact arm to resolve to")
    # the explicit spelling is still refused, and earlier
    with pytest.raises(ValueError, match="ASTIGMATIC"):
        CA._collins_transport(_gauss(), (-0.05, -0.08), Z, WL, DX, DX,
                              **dict(kw, gap_kernel='exact'))


def test_a_traced_scalar_argument_is_refused_by_name():
    """Only the ENVELOPE may be traced (V-D13).

    ``jax.grad`` with respect to ``z`` used to raise
    ``ConcretizationTypeError ... The problem arose with the 'float'
    function`` from ``float(z)`` inside ``_collins_envelope_abcd``, and with
    respect to ``dx`` a ``TracerArrayConversionError`` -- neither naming
    Collins, the transport, or a remedy.  The leg's ABCD entries, its output
    lattice and its chirp screens are built from these as Python floats, so the
    refusal is a statement about the design and not a limitation to apologise
    for.
    """
    env = jnp.asarray(_gauss(), dtype=jnp.complex128)
    kw = dict(dx_out=DX, dy_out=DX, N_out_x=N, N_out_y=N, R_ref=R_REF,
              gap_kernel='fresnel', on_collins_sampling='ignore')

    def by_z(z):
        return jnp.sum(jnp.abs(CA._collins_transport(
            env, R_IN, z, WL, DX, DX, **kw)) ** 2)

    def by_R(r):
        return jnp.sum(jnp.abs(CA._collins_transport(
            env, r, Z, WL, DX, DX, **kw)) ** 2)

    for fn_, arg, name in ((by_z, Z, 'z'), (by_R, R_IN, 'R_in')):
        with pytest.raises(ValueError) as exc:
            jax.grad(fn_)(arg)
        msg = str(exc.value)
        assert f"{name} is a JAX Tracer" in msg, msg
        assert 'ENVELOPE' in msg and 'Differentiate with respect to the field' \
            in msg, msg


def test_a_closed_over_jax_constant_under_jit_is_refused_by_name():
    """The one traced shape ``_is_traced(env)`` cannot see (V-D12).

    A CONCRETE ``jax.numpy`` array closed over by a jitted function is not a
    Tracer, so the measuring branch correctly runs -- but inside a jit trace
    every ``jax.numpy`` operation is STAGED, so the measurement transform's
    OUTPUT is a Tracer and ``_collins_power_marginals``'s ``to_numpy`` raised
    ``TracerArrayConversionError``, for all four spellings INCLUDING the
    otherwise-allowed ``fresnel``/``ignore``.

    Both directions are asserted: the jnp constant is refused by name, and a
    closed-over NUMPY constant still runs (it is host data, and nothing about
    it is staged).
    """
    env_j = jnp.asarray(_gauss(), dtype=jnp.complex128)
    env_n = _gauss()
    kw = dict(dx_out=DX, dy_out=DX, N_out_x=N, N_out_y=N, R_ref=R_REF,
              gap_kernel='fresnel', on_collins_sampling='ignore')

    def closed_jnp(x):
        out = CA._collins_transport(env_j, R_IN, Z, WL, DX, DX, **kw)
        return jnp.sum(jnp.abs(out) ** 2) * x

    with pytest.raises(ValueError) as exc:
        jax.jit(closed_jnp)(1.0)
    msg = str(exc.value)
    assert 'concrete array' in msg and 'STAGES' in msg, msg
    assert 'as an ARGUMENT of the traced function' in msg, msg

    def closed_np(x):
        out = CA._collins_transport(env_n, R_IN, Z, WL, DX, DX, **kw)
        return jnp.sum(jnp.abs(jnp.asarray(out)) ** 2) * x

    assert float(jax.jit(closed_np)(1.0)) > 0.0, (
        "a closed-over NUMPY constant must still run under jit; if it does "
        "not, the refusal above is too broad")


def test_stats_out_is_refused_under_a_trace():
    """``stats_out`` asks for readings that do not exist for a traced array."""
    amp = jnp.asarray(np.real(_gauss()))

    def merit(a):
        out = CA._collins_transport(
            a.astype(jnp.complex128), R_IN, Z, WL, DX, DX, dx_out=DX,
            dy_out=DX, N_out_x=N, N_out_y=N, R_ref=R_REF,
            gap_kernel='fresnel', on_collins_sampling='ignore',
            stats_out={})
        return jnp.sum(jnp.abs(out) ** 2)

    with pytest.raises(ValueError, match="stats_out"):
        jax.grad(merit)(amp)


def test_an_eager_jax_array_is_not_a_tracer_and_measures_normally():
    """The refusal must be about TRACING, not about JAX.  An eager JAX array
    has concrete entries, so both decisions are taken exactly as on NumPy --
    including the warning the guard raises."""
    env = jnp.asarray(_gauss(w=500e-6), dtype=jnp.complex128)
    st = {}
    out = CA._collins_transport(env, -0.30, 0.29, 1.55e-6, 40e-6, 40e-6,
                                dx_out=40e-6 * 16, dy_out=40e-6 * 16,
                                N_out_x=N, N_out_y=N, R_ref=float('inf'),
                                gap_kernel='auto',
                                on_collins_sampling='ignore', stats_out=st)
    assert out.shape == (N, N)
    assert set(('k1', 'k2', 'k3', 'k4', 'kernel')) <= set(st)
    assert st['kernel'] in ('exact', 'fresnel')
    with pytest.warns(RuntimeWarning, match="under-sampled"):
        CA._collins_transport(env, -0.30, 0.29, 1.55e-6, 40e-6, 40e-6,
                              dx_out=40e-6 * 16, dy_out=40e-6 * 16,
                              N_out_x=N, N_out_y=N, R_ref=float('inf'),
                              gap_kernel='auto', on_collins_sampling='warn')


# ===========================================================================
# 4. The gradient, against a central finite difference (a ladder)
# ===========================================================================

#: the finite-difference ladder, extended upward to ``3e-1`` in 2026-09.
#: The old ladder stopped at ``1e-2`` and therefore sat ENTIRELY on the
#: cancellation branch without saying so; the best step on this merit is near
#: the TOP of the ladder, not in the middle of it.
_FD_LADDER = (3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5)


def _fd_ladder(merit, a, ij, g_ij, P0):
    """``[(h, rel_disagreement, cancellation_floor), ...]`` over the ladder.

    ``cancellation_floor`` is the derived bound on a central difference's
    ROUND-OFF error, relative to the gradient entry it is compared against::

        |fd - g| ~ eps |P| / h        ->      rel ~ eps |P| / (h |g|)

    It carries no truncation term; see the test below for why that is a
    measured fact about this merit and not an omission.
    """
    eps = float(np.finfo(np.float64).eps)
    out = []
    for h in _FD_LADDER:
        def _at(sign, mult=1):
            ap = a.copy()
            ap[ij] += sign * mult * h
            return float(merit(jnp.asarray(ap)))
        fd = (_at(+1) - _at(-1)) / (2.0 * h)
        out.append((h, abs(fd - g_ij) / abs(g_ij),
                    eps * abs(P0) / (h * abs(g_ij))))
    return out


def test_jax_grad_through_the_collins_transport_matches_a_central_difference():
    """``jax.grad`` of a scalar merit through the transport, against an
    independent central finite difference -- on a LADDER of step sizes, with
    the bar derived from the difference's own error rather than chosen.

    THE FLOOR MODEL, CORRECTED 2026-09-19 (VERIFY-WAVE5-HYGIENE2 V-D7).  A
    central difference has error ``~ P''' h^2 / 6`` (truncation) plus
    ``~ eps |P| / h`` (cancellation), and where BOTH exist the balance sits at
    ``h ~ eps^(1/3)`` with a relative error ``~ eps^(2/3) = 3.67e-11``.  That
    model does not apply here.  ``_collins_transport`` is LINEAR in the
    envelope, so ``P(a) = sum|L a|^2`` is an exact quadratic form and
    ``P''' == 0``: there is no truncation branch, the ladder has no U-curve
    minimum, and ``eps^(2/3)`` is ~4.7 decades looser than the measurement --
    loose enough that a gradient wrong by three decades would have passed.
    MEASURED on this build 2026-09-19: best 7.55e-15 at ``h = 1e-1``, against
    the old bar of 3.67e-10.

    THE BAR IS NOW THE CANCELLATION BRANCH ITSELF, at every rung:
    ``rel <= 10 * eps |P| / (h |g|)``.  It is derived, it is per-rung rather
    than a single number, and it tracks the fixture: MEASURED ratios of the
    disagreement to its own floor over the ten rungs are 0.077 .. 0.737, so
    the factor of ten carries 13.6x of headroom at the worst rung while the
    failure it must catch -- a wrong gradient -- is decades away.

    THE GRADIENT'S SHAPE.  The merit is transported POWER, which the leg
    conserves to grid accuracy, so ``d/da_ij sum|u_out|^2`` is proportional to
    ``a_ij``.  Its bar is derived from the leg's own measured departure from a
    scaled isometry, not pinned; see :func:`_gradient_shape_bars`.
    """
    amp0 = jnp.asarray(np.real(_gauss()))
    merit = _merit_factory(gap_kernel='fresnel',
                           on_collins_sampling='ignore')
    g = np.asarray(jax.grad(merit)(amp0))
    assert np.all(np.isfinite(g))

    a = np.asarray(amp0)
    ij = np.unravel_index(int(np.argmax(a)), a.shape)
    P0 = float(merit(amp0))
    rows = _fd_ladder(merit, a, ij, g[ij], P0)

    over = [(h, rel, fl) for h, rel, fl in rows if rel > 10.0 * fl]
    assert not over, (
        f"the central difference disagrees with jax.grad by more than ten "
        f"times its own cancellation floor eps|P|/(h|g|) at "
        f"{[f'h={h:.0e}: {r:.3e} vs {fl:.3e}' for h, r, fl in over]}; "
        f"P0={P0:.6e}, g_ij={g[ij]:.6e}")
    best_h, best, best_fl = min(rows, key=lambda r: r[1])
    # Two-sided: the bar must also be above the measurement by a real margin,
    # or it is not bounding anything.  MEASURED worst ratio 0.737.
    assert best < 10.0 * best_fl, (
        f"best {best:.3e} at h={best_h:.0e} against its floor {best_fl:.3e}")
    assert best > 0.0, (
        "the best central difference agrees with jax.grad EXACTLY, which a "
        "round-off-limited comparison cannot do; the two arms are not "
        "independent")

    _gradient_shape_bars(g, a)


def _gradient_shape_bars(g, a):
    """The gradient's SHAPE claim, with the bar derived from the leg itself.

    The merit is ``P(a) = a^T (L^H L) a`` with ``L`` the (linear) leg, so
    ``grad P = 2 (L^H L) a`` is exactly parallel to ``a`` when ``a`` is an
    eigenvector of ``L^H L`` -- i.e. when the leg acts on this envelope as a
    SCALED ISOMETRY.  It does not exactly: a 64x64 grid at 8 um clips a 60 um
    envelope, and the clipping is what makes the leg depart from one.

    So the shortfall of Pearson's correlation is not free to be pinned; it is
    SECOND ORDER in that departure.  Write ``c`` for the best scalar multiple
    and ``r`` for the residual's relative norm::

        c = <g, a> / <a, a>,    r = |g - c a| / |c a|,    1 - corr ~ r^2 / 2

    MEASURED on the shipped fixture, both builds 2026-09-19: ``r = 4.998e-04``,
    ``1 - corr = 3.1689e-07``, ``(1 - corr)/r^2 = 1.2686``.  The bars are
    ``r^2/100 < 1 - corr < 10 r^2`` -- 7.9x of headroom above and 127x below.

    WHY THE PINNED BAR IS GONE (V-D6).  ``corr > 1 - 1e-6`` had no stated
    origin, sat 0.50 decades from firing (3.1689e-07 against 1e-6), and
    measured a property of the FIXTURE rather than of the port: over envelope
    widths 30..100 um on the same grid, ``1 - corr`` moves 6.08e-11 ..
    1.13e-03, so a 17 % change of the fixture's width would have put it 7x
    over the bar.  The derived form moves with the fixture, because it is
    written against the same clipping that moves it.
    """
    m = a > 0.05 * a.max()
    c = float(np.dot(g[m], a[m]) / np.dot(a[m], a[m]))
    r = float(np.linalg.norm(g[m] - c * a[m]) / np.linalg.norm(c * a[m]))
    corr = float(np.corrcoef(g[m], a[m])[0, 1])
    # PREMISE: the leg is NEAR a scaled isometry on this fixture but not
    # exactly one.  Both ends matter -- at r = 0 the claim is vacuous, and
    # above ~1e-1 the second-order expansion the bar rests on stops holding.
    assert 0.0 < r < 1e-1, (
        f"PREMISE FAILED: the gradient's residual from the best scalar "
        f"multiple of the input is r = {r:.4e}; the 1 - corr ~ r^2/2 "
        f"expansion the bars below are derived from does not apply")
    assert (1.0 - corr) < 10.0 * r * r, (
        f"the gradient of transported power departs from proportionality to "
        f"the input amplitude by more than the leg's own non-isometry allows: "
        f"1 - corr = {1.0 - corr:.4e} against 10 r^2 = {10.0 * r * r:.4e} "
        f"(r = {r:.4e}).  The leg is not conserving power, or the gradient is "
        f"not the one it should be")
    assert (1.0 - corr) > 0.01 * r * r, (
        f"1 - corr = {1.0 - corr:.4e} is BELOW r^2/100 = {0.01 * r * r:.4e}.  "
        f"Pearson's shortfall is second order in the residual, so a "
        f"correlation this good with a residual this large means the two are "
        f"not being measured on the same entries")


def test_the_central_difference_through_this_merit_has_no_truncation_branch():
    """The premise the bar above rests on, MEASURED rather than argued.

    ``_collins_transport`` is linear in the envelope, so ``P = sum|L a|^2`` is
    an exact quadratic form and its third derivative is identically zero.  A
    quantity that is zero is not something a test may assume, so it is read
    off the ladder -- but NOT by a ratio.  A five-point third difference of a
    quadratic returns pure round-off, and round-off is free to come out
    EXACTLY 0.0: MEASURED 2026-09-19 on WSL py3.12, the estimate reads
    ``+0.00000e+00`` at ``h = 1e-4`` while Windows py3.14 reads
    ``+7.11e-03`` at the same rung.  A first cut of this id asserted a
    per-decade GROWTH ratio and was red on WSL for exactly that reason -- a
    ratio whose denominator or numerator may be zero is not a statistic, which
    is this repository's own S4 shape.

    THE BUILD-FREE FORM is a BOUND: a round-off third difference is bounded by
    the floor it comes from, ``eps |P| / h^3``, and zero satisfies that bound
    while a real derivative does not.  MEASURED at ``h = 1e-1``, estimate over
    its own floor: **0.36 (WIN) and 1.09 (WSL)**, against a bar of 100.  The
    falsification arm is the next id, where the same statistic on a genuinely
    cubic merit reads **1.9e+04 .. 7.0e+08** over the same ladder -- two
    decades clear of this bar on one side and six on the other.

    The second assertion is the one the previous id's bar actually needs: the
    implied truncation contribution ``|P'''| h^2 / (6|g|)`` stays under the
    cancellation floor at every rung, on both builds.
    """
    amp0 = jnp.asarray(np.real(_gauss()))
    merit = _merit_factory(gap_kernel='fresnel',
                           on_collins_sampling='ignore')
    a = np.asarray(amp0)
    ij = np.unravel_index(int(np.argmax(a)), a.shape)
    P0 = float(merit(amp0))
    eps = float(np.finfo(np.float64).eps)

    ests = []
    for h in (1e-1, 1e-2, 1e-3, 1e-4):
        def _at(sign, mult=1):
            ap = a.copy()
            ap[ij] += sign * mult * h
            return float(merit(jnp.asarray(ap)))
        ests.append((h, (_at(+1, 2) - 2 * _at(+1) + 2 * _at(-1) - _at(-1, 2))
                     / (2.0 * h ** 3)))

    over = [(h, p3, abs(p3) / (eps * abs(P0) / h ** 3))
            for h, p3 in ests if abs(p3) > 100.0 * eps * abs(P0) / h ** 3]
    assert not over, (
        f"the third-difference estimate exceeds 100x the round-off floor "
        f"eps|P|/h^3 at {[f'h={h:.0e}: {p:.3e} ({r:.1f}x)' for h, p, r in over]}"
        f" -- so it is NOT pure round-off: this merit has a real third "
        f"derivative and the cancellation-only bar in the previous id is not "
        f"the right model.  P0={P0:.6e}")
    # and the implied truncation contribution is below the cancellation floor
    # at every rung, which is the statement the bar actually needs.
    g_ij = float(np.asarray(jax.grad(merit)(amp0))[ij])
    for h, p3 in ests:
        trunc = abs(p3) * h * h / (6.0 * abs(g_ij))
        cancel = eps * abs(P0) / (h * abs(g_ij))
        assert trunc <= cancel, (
            f"at h={h:.0e} the implied truncation term {trunc:.3e} exceeds the "
            f"cancellation floor {cancel:.3e}; the ladder has a truncation "
            f"branch after all")


def test_the_same_ladder_does_find_a_truncation_branch_on_a_cubic_merit():
    """Falsification arm for the id above: the instrument is not blind.

    Cube the on-axis intensity and the merit is genuinely cubic in the
    perturbed entry, so ``P'''`` is a real number and the ladder must show the
    textbook ``h^2`` arm and a real U.  MEASURED 2026-09-19: the
    third-difference estimate is CONSTANT at +2.235e-04 over four decades
    (against the 1e3-per-decade growth of the quadratic merit), the
    disagreement falls exactly 10x per decade -- 7.69e-06, 6.92e-07, 7.69e-08,
    6.92e-09 at h = 1e-1, 3e-2, 1e-2, 3e-3 -- and the truncation model
    ``|P'''| h^2 / (6 |g|)`` predicts it to three significant figures.  The
    minimum is real, at h = 1e-4.
    """
    amp0 = jnp.asarray(np.real(_gauss()))

    def merit(amp):
        out = CA._collins_transport(
            amp.astype(jnp.complex128), R_IN, Z, WL, DX, DX, dx_out=DX,
            dy_out=DX, N_out_x=N, N_out_y=N, R_ref=R_REF,
            gap_kernel='fresnel', on_collins_sampling='ignore')
        return (jnp.abs(out[N // 2, N // 2]) ** 2) ** 3

    a = np.asarray(amp0)
    ij = np.unravel_index(int(np.argmax(a)), a.shape)
    g_ij = float(np.asarray(jax.grad(merit)(amp0))[ij])
    P0 = float(merit(amp0))

    rows = []
    for h in (1e-1, 3e-2, 1e-2, 3e-3):
        def _at(sign, mult=1):
            ap = a.copy()
            ap[ij] += sign * mult * h
            return float(merit(jnp.asarray(ap)))
        fd = (_at(+1) - _at(-1)) / (2.0 * h)
        p3 = (_at(+1, 2) - 2 * _at(+1) + 2 * _at(-1) - _at(-1, 2)) \
            / (2.0 * h ** 3)
        rows.append((h, abs(fd - g_ij) / abs(g_ij), p3))

    # PREMISE: P''' is a real, CONSTANT number here, and it sits DECADES above
    # its own round-off floor -- the opposite reading from the quadratic
    # merit, whose estimate is bounded BY that floor (and is exactly 0.0 at
    # one rung on WSL).  Both halves, because "constant" alone would also
    # describe a constant that is pure noise.
    eps = float(np.finfo(np.float64).eps)
    p3s = [abs(r[2]) for r in rows]
    assert max(p3s) / min(p3s) < 1.1, (
        f"PREMISE FAILED: the third-difference estimate is not constant over "
        f"the ladder ({['%.4e' % p for p in p3s]}), so this merit is not the "
        f"cubic control it is meant to be")
    floors = [abs(p3) / (eps * abs(P0) / h ** 3) for h, _rel, p3 in rows]
    assert min(floors) > 1e4, (
        f"PREMISE FAILED: the cubic merit's third difference is only "
        f"{min(floors):.3g}x its own round-off floor (measured 1.9e+04 .. "
        f"7.0e+08 on 2026-09-19).  The quadratic merit's id passes at <= 100x "
        f"that floor, so if this control drops toward 100 the two readings "
        f"stop being separable and both bars become noise")
    # THE CLAIM: the disagreement IS the truncation model, to 10 %, and it
    # falls as h^2 -- neither is true of the shipped quadratic merit.
    for h, rel, p3 in rows:
        pred = abs(p3) * h * h / (6.0 * abs(g_ij))
        assert abs(rel - pred) / pred < 0.1, (
            f"at h={h:.0e} the disagreement {rel:.4e} is not the truncation "
            f"model {pred:.4e}; P0={P0:.6e}")
    slope = np.polyfit(np.log([r[0] for r in rows]),
                       np.log([r[1] for r in rows]), 1)[0]
    assert abs(slope - 2.0) < 0.05, (
        f"the cubic merit's central difference falls as h^{slope:.4f}, not "
        f"h^2; the ladder is not resolving a truncation branch even where one "
        f"exists, which would make the previous id's conclusion vacuous")


def test_the_gradient_is_not_trivially_zero_or_constant():
    """The falsification arm: a gradient that were all-zero, or all-equal,
    would pass a correlation test by accident on a degenerate fixture."""
    amp0 = jnp.asarray(np.real(_gauss()))
    g = np.asarray(jax.grad(_merit_factory(
        gap_kernel='fresnel', on_collins_sampling='ignore'))(amp0))
    assert float(np.max(np.abs(g))) > 0.0
    assert float(np.std(g)) / float(np.max(np.abs(g))) > 1e-3


def test_jit_of_the_transport_matches_the_eager_traced_call():
    """``jax.jit`` traces the same way ``jax.grad`` does, so the refusal and
    the allowed spelling must behave identically under it."""
    env = jnp.asarray(_gauss(), dtype=jnp.complex128)

    def f(e):
        return CA._collins_transport(
            e, R_IN, Z, WL, DX, DX, dx_out=DX, dy_out=DX, N_out_x=N,
            N_out_y=N, R_ref=R_REF, gap_kernel='fresnel',
            on_collins_sampling='ignore')

    eager = np.asarray(f(env))
    jitted = np.asarray(jax.jit(f)(env))
    bar = _fft_spread_bar(np.asarray(env))
    assert _rel(jitted, eager) < bar

    def bad(e):
        return CA._collins_transport(
            e, R_IN, Z, WL, DX, DX, dx_out=DX, dy_out=DX, N_out_x=N,
            N_out_y=N, R_ref=R_REF, gap_kernel='auto',
            on_collins_sampling='ignore')

    with pytest.raises(ValueError, match="Tracer"):
        jax.jit(bad)(env)
