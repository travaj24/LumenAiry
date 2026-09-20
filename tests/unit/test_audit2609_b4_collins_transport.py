"""WP-B4 -- ``transport='collins'``: the Collins / ABCD-Fresnel carrier
transport with a freely chosen (Bluestein) output pitch.

What is pinned here, in the order the WP's acceptance gate asks for it:

* the DEFAULT does not move -- ``np.array_equal`` between the shipped call and
  the same call with ``transport='sziklas'`` spelled out, at every entry point
  (``TestDefaultIsByteIdentical``);
* the transport is the SAME THEOREM as the Sziklas step, checked three ways --
  against ``_carrier_step_fast`` on the co-moving lattice, against an analytic
  Gaussian-ABCD oracle written here (including the absolute piston and the Gouy
  phase, so nothing is hidden by a piston-free comparison), and against a
  DIRECT SUMMATION of the same integral that shares no FFT, no Bluestein and no
  chirp with the code under test (``TestSameTheorem``);
* gate (a), the NA x grid-extent matrix against that oracle
  (``TestGateAOracleMatrix``);
* gate (b), WP-A6's C1 mismatch matrix (``TestGateBMismatchMatrix``);
* gate (c), a two-group chain against a brute-force ASM +
  ``apply_real_lens_traced`` arm (``TestGateCTwoGroupChain``);
* gate (d), ``propagate_traced_carrier_chain_multi`` K = 1 against the chain and
  K = 2 against the hand-summed pair (``TestGateDMulti``);
* the Kelly (Appl. Opt. 53, 2861 (2014)) sampling guard, two-sided: the
  conditions are ratios against the Nyquist rate itself, a passing
  configuration is silent AND accurate, and a failing one fires AND is
  inaccurate (``TestKellyGuard``, ``TestKernelRefinement``);
* the quadrature selection is COMPLEMENTARY, not tuned: the chirp-Z form and
  the transfer-function form have exactly opposite sampling conditions, so
  every leg satisfies one of them and both agree at the crossover
  (``TestQuadratureComplementarity``);
* the near-focus apparatus is never entered on this transport, proved by
  poisoning all four of its entry points (``TestNoNearFocusApparatus``);
* the readout's Bluestein period stops being a function of the resolved leg
  (``TestReadoutPeriodDecoupling``).

Per ``docs/TESTING_STANDARDS.md``: no wall-clock assertion anywhere (the WP
report carries the timings), no ``pytest.skip`` on a resource precondition,
every bar derived from its own oracle's floor with the measured value and the
date in the comment.  The oracles in section 1 are written in this file from
the ``q``-parameter definition and from a direct quadrature; nothing below
calls the library to build a truth.
"""

import warnings

import numpy as np
import pytest

from lumenairy.propagators import carrier as C

EPS = float(np.finfo(np.float64).eps)


# ===========================================================================
# 1.  Oracles (written here; nothing in this section calls the transport)
# ===========================================================================
def _grid(n, d):
    return (np.arange(n, dtype=np.float64) - n / 2) * float(d)


def _gauss_env(n, dx, w):
    g = _grid(n, dx)
    return np.exp(-((g[None, :] ** 2 + g[:, None] ** 2) / w ** 2)).astype(
        np.complex128)


def _abcd_gauss(xo, w_in, r_beam, z, lam):
    """Analytic Gaussian field a distance ``z`` on, in THIS library's
    ``exp(-i omega t)`` / ``exp(+i k z)`` convention (CONVENTIONS sec. 7):

        1/q = 1/R + i lam/(pi w^2),   q2 = q + z,
        E(r) = exp(i k z) / (1 + z/q) * exp(i k r^2 / (2 q2)).

    Note the SIGN of the imaginary part.  Siegman's ``1/q = 1/R - i
    lam/(pi w^2)`` belongs to the opposite time convention; used as-is it
    conjugates the Gouy phase, which is invisible to a piston-free comparison
    and is exactly ``pi`` of error at a focus.  This form carries the absolute
    piston and the Gouy phase, so the comparisons below need no phase
    alignment.  Returns ``(E, w(z))``."""
    k = 2.0 * np.pi / lam
    q = 1.0 / (1.0 / r_beam + 1j * lam / (np.pi * w_in ** 2))
    q2 = q + z
    r2 = xo[None, :] ** 2 + xo[:, None] ** 2
    wz = float(np.sqrt(lam / (np.pi * (1.0 / q2).imag)))
    return (np.exp(1j * k * z) / (1.0 + z / q)
            * np.exp(1j * k * r2 / (2.0 * q2))), wz


def _collins_direct(env, R_in, z, lam, dx, xs, ys, R_ref=np.inf):
    """The Collins integral evaluated by DIRECT SUMMATION at the listed output
    points -- the independent quadrature.

    Shares nothing with the transport under test: no FFT, no Bluestein, no
    chirp-Z, no output lattice.  Written straight from the ABCD-Fresnel form in
    this library's convention,

        u_out(x) = exp(i k B)/(i lam B)
                   * sum_u env(u) exp(i k (A u^2 - 2 u x + D x^2)/(2 B)) du^2,

    with ``A = 1 + z/R_in``, ``B = z``, ``D = 1 - z/R_ref``."""
    k = 2.0 * np.pi / lam
    n = np.shape(env)[-1]
    u = _grid(n, dx)
    A = 1.0 if np.isinf(R_in) else 1.0 + z / R_in
    D = 1.0 if np.isinf(R_ref) else 1.0 - z / R_ref
    pre = np.exp(1j * k * A * u * u / (2.0 * z))
    g = np.asarray(env) * pre[None, :] * pre[:, None]
    out = np.empty((len(ys), len(xs)), dtype=np.complex128)
    for iy, yv in enumerate(ys):
        ey = np.exp(-1j * k * u * yv / z)
        for ix, xv in enumerate(xs):
            ex = np.exp(-1j * k * u * xv / z)
            s = complex(ey @ g @ ex) * dx * dx
            out[iy, ix] = (np.exp(1j * k * z) / (1j * lam * z) * s
                           * np.exp(1j * k * D * (xv * xv + yv * yv)
                                    / (2.0 * z)))
    return out


def _rel_l2(E, T):
    return float(np.linalg.norm(np.asarray(E) - np.asarray(T))
                 / np.linalg.norm(np.asarray(T)))


def _piston_free_rel_l2(E, T):
    E, T = np.asarray(E), np.asarray(T)
    ov = np.vdot(T, E)
    return _rel_l2(E / (ov / abs(ov)) if abs(ov) > 0 else E, T)


# --- the two shared fixtures -----------------------------------------------
_WL = 1.064e-6                 # deliberately neither WP-A6's 1.31 um nor
_N, _DX, _W = 1024, 4.0e-6, 0.30e-3  # VERIFY-A6's 0.85 / 0.633 um
_R = -40.0e-3                  # converging: the focus is 40 mm on
# The grid spans +/-6.83 beam radii, so the Gaussian's own truncation floor is
# exp(-6.83^2) = 8e-21 in amplitude and every bar below is the transform's
# rounding rather than the fixture's edge.


@pytest.fixture(scope='module')
def env_conv():
    return _gauss_env(_N, _DX, _W)


# ===========================================================================
# 2.  Vocabulary (CONVENTIONS sec. 2: the message starts with the function)
# ===========================================================================
class TestVocabulary:
    @pytest.mark.parametrize('bad', ['Collins', 'COLLINS', 'collin',
                                     'sziklas ', None, 0, b'collins'])
    def test_an_unrecognised_transport_is_refused_not_defaulted(self, bad,
                                                                env_conv):
        with pytest.raises(ValueError) as ei:
            C.propagate_carrier_referenced(env_conv, _R, 5e-3, _WL, _DX,
                                           transport=bad)
        msg = str(ei.value)
        assert msg.startswith('propagate_carrier_referenced: ')
        assert 'sziklas' in msg and 'collins' in msg

    def test_the_free_lattice_kwargs_are_refused_on_the_sziklas_transport(
            self, env_conv):
        """``dx_out`` / ``carrier_out`` have no referent on a transport whose
        output pitch IS ``m*dx``; accepting and ignoring them is the
        accept-and-ignore class D4/D11 adjudicated against.

        ``transport='sziklas'`` is now NAMED (WP-C3).  It used to be the
        default and this test used to reach the refusal by passing nothing;
        since 5.49.0 the default is ``'collins'``, on which these two keywords
        are the whole point of the transport.  So the refusal is a contract of
        the SZIKLAS transport, and the test says so -- restated, not
        re-pinned.  That the default now ACCEPTS them is the other half and is
        asserted below.
        """
        for kw in ({'dx_out': 1e-6}, {'carrier_out': np.inf}):
            with pytest.raises(ValueError, match='transport'):
                C.propagate_carrier_referenced(env_conv, _R, 5e-3, _WL, _DX,
                                               transport='sziklas', **kw)

    def test_the_free_lattice_kwargs_are_honoured_on_the_default(
            self, env_conv):
        """The other side of the refusal above: on the 5.49.0 default the two
        keywords are live, and they must actually CHANGE the returned
        lattice -- otherwise this test would pass on a transport that accepted
        and ignored them, which is the class the refusal exists for."""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            free = C.propagate_carrier_referenced(
                env_conv, _R, 5e-3, _WL, _DX, dx_out=1e-6)
            flat = C.propagate_carrier_referenced(
                env_conv, _R, 5e-3, _WL, _DX, carrier_out=np.inf)
        assert float(free.dx) == pytest.approx(1e-6, rel=0, abs=0)
        assert np.isinf(flat.R)

    def test_the_stop_plane_readout_keys_SELECT_the_sziklas_readout(self):
        """``standoff`` and ``on_focus_containment`` describe the Sziklas
        readout's stop plane.

        THE CONTRACT CHANGED WITH THE DEFAULT (WP-C3), and the change is
        recorded here rather than the test being deleted.  WP-B4 REFUSED these
        two keys on ``transport='collins'`` -- correctly at the time: that
        transport had no stop plane and no fallback, so the keys had no
        referent and accepting them would have been accept-and-ignore.  Since
        the chain's readout RESOLVES its quadrature they have a referent
        again, because the Sziklas readout is the route most chain readouts
        take, so naming one now SELECTS that route.  Nothing is accepted and
        ignored: the key does exactly what it says, and the stage says so.

        Both halves are asserted -- the route taken AND the reason published
        -- so a future change that went back to ignoring the key silently
        would fail here rather than reading as a pass.

        THE FIELD ARM RUNS ON A FIXTURE WHERE THE TWO ROUTES DIFFER (WP-C3
        round 2, VERIFY-WP-C3 section 7.1b).  As first written this test drove
        only ``final_distance = 8e-3``, where K1 = 82.36 and the default takes
        the Sziklas readout anyway -- so its ``array_equal`` arm could not
        fail, whatever the keys did.  The second fixture below is MEASURED to
        take the one-step route with no key named (K1 = 0.98606 at N = 1024,
        ``final_distance`` 46 mm, ``readout_route='collins'``), so naming a
        Sziklas-only key there really does flip the quadrature and the
        bit-identity assertion has something to say.  ``bandlimit`` joined the
        two stop-plane keys in round 2 (VERIFY-WP-C3 D4).
        """
        env, dx, r_in, groups = _chain_fixture()
        base = dict(r_in=r_in, ray_subsample=16, n_workers=1,
                    traced_kwargs=_CHAIN_TKW, final_leg='paraxial',
                    final_distance=8e-3)
        fr0 = dict(dx_out=0.5e-6, N_out=64)
        keys = (('standoff', 2e-3), ('on_focus_containment', 'ignore'),
                ('bandlimit', False))
        for key, val in keys:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                got = C.propagate_traced_carrier_chain(
                    env, groups, 1.31e-6, dx, transport='collins',
                    focus_readout=dict(fr0, **{key: val}), **base)
                ref = C.propagate_traced_carrier_chain(
                    env, groups, 1.31e-6, dx, transport='sziklas',
                    focus_readout=dict(fr0, **{key: val}), **base)
            st = got.stages[-1]
            assert st['readout_route'] == 'sziklas'
            assert st['readout_route_reason'] == 'sziklas_only_key'
            assert st['readout_route_k1'] is None, (
                'K1 was computed for a route the keyword had already '
                'decided; it is not the reason and must not be published '
                'as one')
            assert np.array_equal(np.asarray(got.field),
                                  np.asarray(ref.field))

        # ... and again where the two routes genuinely disagree.
        n2, fd2 = 1024, 46e-3
        dx2 = dx * 256 / n2
        env2 = _gauss_env(n2, dx2, 4.5e-3)
        base2 = dict(base, final_distance=fd2)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            free = C.propagate_traced_carrier_chain(
                env2, groups, 1.31e-6, dx2, transport='collins',
                focus_readout=dict(fr0), **base2)
        st0 = free.stages[-1]
        assert st0['readout_route'] == 'collins' and \
            st0['readout_route_k1'] <= 1.0, (
            f'the second fixture no longer takes the one-step route with no '
            f'key named, so the field arm below would be vacuous again: '
            f"{st0.get('readout_route')!r}, K1 = "
            f"{st0.get('readout_route_k1')!r}")
        for key, val in keys:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                got = C.propagate_traced_carrier_chain(
                    env2, groups, 1.31e-6, dx2, transport='collins',
                    focus_readout=dict(fr0, **{key: val}), **base2)
                ref = C.propagate_traced_carrier_chain(
                    env2, groups, 1.31e-6, dx2, transport='sziklas',
                    focus_readout=dict(fr0, **{key: val}), **base2)
            st = got.stages[-1]
            assert st['readout_route'] == 'sziklas'
            assert st['readout_route_reason'] == 'sziklas_only_key'
            assert st['readout_route_k1'] is None
            assert np.array_equal(np.asarray(got.field),
                                  np.asarray(ref.field)), (
                f'naming {key!r} selected the Sziklas readout by its stage '
                f'keys but did not return the Sziklas ANSWER')
            assert not np.array_equal(np.asarray(got.field),
                                      np.asarray(free.field)), (
                f'naming {key!r} left the answer where the un-keyed default '
                f'put it, so the key selected nothing')

    def test_an_astigmatic_exact_kernel_is_refused_on_collins(self, env_conv):
        with pytest.raises(ValueError, match='ASTIGMATIC'):
            C.propagate_carrier_referenced(
                env_conv, (-40e-3, -50e-3), 5e-3, _WL, _DX,
                transport='collins', gap_kernel='exact')

    def test_the_b_zero_readout_says_what_to_do(self, env_conv):
        with pytest.raises(ValueError, match='zero-length'):
            C._collins_focus_readout(env_conv, _R, 0.0, _WL, _DX, _DX,
                                     dx_out=1e-6, N_out=8)


# ===========================================================================
# 3.  The default does not move -- byte identity
# ===========================================================================
def _singlet(R1, R2, d, glass, ap, name):
    return {'name': name, 'aperture_diameter': ap, 'thicknesses': [d],
            'surfaces': [
                {'radius': R1, 'glass_before': 'air', 'glass_after': glass,
                 'conic': 0.0, 'radius_y': None, 'conic_y': None,
                 'aspheric_coeffs': None, 'aspheric_coeffs_y': None},
                {'radius': R2, 'glass_before': glass, 'glass_after': 'air',
                 'conic': 0.0, 'radius_y': None, 'conic_y': None,
                 'aspheric_coeffs': None, 'aspheric_coeffs_y': None}]}


_CHAIN_TKW = dict(on_undersample='silent', on_noncollimated='silent')


def _chain_fixture():
    """A two-group relay small enough to run twice per test.  Same shape as
    the D2 K=1 fixture (which is the shipped design-121 acceptance's own
    reduction), at a quarter of its grid."""
    n, dx, w, r_in = 256, 60e-6, 4.5e-3, 60e-3
    presc = _singlet(60e-3, -60e-3, 3e-3, 'N-BK7', 14e-3, 'p')
    return (_gauss_env(n, dx, w), dx, r_in,
            [{'prescription': presc, 'gap_before': 20e-3},
             {'prescription': presc, 'gap_before': 10e-3}])


class TestDefaultIsByteIdentical:
    """``transport`` defaults to ``'collins'`` since 5.49.0 (WP-C3), and the
    statement that makes "nothing existing moves" checkable has MOVED with it.

    It used to read "naming ``'sziklas'`` changes nothing", because that was
    the default.  It now reads, in two halves:

    * naming ``'collins'`` changes nothing, because that is the default --
      the first arm below;
    * naming ``'sziklas'`` returns the PRE-FLIP arithmetic in every bit, which
      cannot be proved against the working tree and is proved archive to
      archive instead (``validation/probe_c3_collins_default/``: 42 of 42 keys
      on both builds, with the base spelled to pass no ``transport=`` at all).
      ``tests/unit/test_c3_collins_default.py`` gates the structural property
      that keeps it true.

    THE CELLS THAT MOVE, and why they are the ones that move: exactly the legs
    where the chirp-Z quadrature is representable (``N dx^2 <= lambda
    |z_eff|``, VERIFY-WP-B4 F2).  On every other leg the transport RESOLVES to
    the transfer-function form, which IS the Sziklas step, and the two
    spellings stay bit-identical -- asserted below rather than assumed, with
    both sets counted so neither can quietly empty.
    """

    #: The five WP-B4 cells plus ONE that the flip actually moves.  MEASURED
    #: 2026-09-20: on this fixture (N = 1024 at 4 um) all five original cells
    #: resolve to the transfer-function form and are bit-identical on both
    #: spellings -- which is the flip's central claim and also means they
    #: cannot, by themselves, tell a working selection from a dead one.  The
    #: sixth cell is a leg PAST the carrier's geometric focus (``A = -0.5``),
    #: where the co-moving frame has inverted, ``Ax > 0`` fails and no
    #: transfer-function form exists: the chirp-Z runs and the answer moves.
    _CELLS = [(-40e-3, 5e-3), (-40e-3, -3e-3), (np.inf, 5e-3),
              (80e-3, 12e-3), ((-40e-3, -55e-3), 5e-3),
              (-40e-3, 60e-3)]

    @pytest.mark.parametrize('R,z', _CELLS)
    def test_the_single_step_is_equal_bit_for_bit_to_the_named_default(
            self, env_conv, R, z):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            a = C.propagate_carrier_referenced(env_conv, R, z, _WL, _DX)
            b = C.propagate_carrier_referenced(env_conv, R, z, _WL, _DX,
                                               transport='collins')
        assert np.array_equal(np.asarray(a.env), np.asarray(b.env))
        assert a.R == b.R and a.dx == b.dx

    def test_the_cells_that_resolve_to_the_transfer_function_form_do_not_move(
            self, env_conv):
        """Both sides of the resolution, counted.

        Every cell of ``_CELLS`` is run on both transports and the leg is
        asked which quadrature it resolved to.  A cell that resolved to the
        transfer-function form MUST be bit-identical (that form is the
        Sziklas step); a cell where the chirp-Z ran is allowed to differ, and
        at least one must, or the flip would be changing nothing here and this
        class would be gating nothing.
        """
        same, moved = [], []
        for R, z in self._CELLS:
            diag = {}
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                R_x, R_y, is_astig = C._parse_carrier(R, 'probe')
                C._collins_carrier_leg(
                    env_conv, ((R_x, R_y) if is_astig else R_x), z, _WL,
                    _DX, _DX, on_collins_sampling='ignore', diag=diag)
                a = C.propagate_carrier_referenced(env_conv, R, z, _WL, _DX)
                b = C.propagate_carrier_referenced(env_conv, R, z, _WL, _DX,
                                                   transport='sziklas')
            eq = (np.shape(a.env) == np.shape(b.env)
                  and np.array_equal(np.asarray(a.env), np.asarray(b.env))
                  and a.R == b.R and a.dx == b.dx)
            (same if diag['collins_form'] == 'tf' else moved).append(
                (R, z, diag['collins_form'], eq))
        for R, z, form, eq in same:
            assert eq, (
                f'R={R!r} z={z!r} resolved to the transfer-function form, '
                f'which IS the Sziklas step, yet the two spellings differ')
        assert moved, (
            'every cell resolved to the transfer-function form, so this '
            'fixture set cannot see the flip at all')
        assert any(not eq for _R, _z, _f, eq in moved), (
            f'the chirp-Z ran on {len(moved)} cell(s) and none of them moved; '
            f'either the two quadratures agree to the bit here (they do not) '
            f'or the transport is not being selected')

    def test_the_focus_crossing_split_is_not_entered_on_the_default(
            self, env_conv):
        """The near-focus branch: the cell the flip exists for.

        ``z = -R * 0.995`` lands inside the Sziklas bridge's own trigger, so
        the two transports take structurally different routes and the fields
        are NOT expected to agree.  What is asserted is that the default
        answer is finite and lands on a FINER lattice than the collapsing
        co-moving one -- the point of the one-step form near a focus being
        that its pitch is floored by the measured box instead of collapsing
        with the carrier.  (``z = -R`` exactly is not used here: the Sziklas
        transport cannot land on the focus at all -- its bridge re-references
        to ``R_out = 0`` and ``carrier_referenced_envelope`` refuses that --
        which is the cell ``TestNoNearFocusApparatus`` takes.)
        """
        z = -_R * 0.995
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            a = C.propagate_carrier_referenced(env_conv, _R, z, _WL, _DX)
            b = C.propagate_carrier_referenced(env_conv, _R, z, _WL, _DX,
                                               transport='sziklas')
        assert np.all(np.isfinite(np.asarray(a.env)))
        assert float(a.dx) < float(b.dx), (
            f'the default landed on {float(a.dx):.4e} m and the co-moving '
            f'route on {float(b.dx):.4e} m')

    def test_the_public_readout_is_still_the_sziklas_readout(self):
        """``carrier_referenced_focus_readout`` is still the STANDOFF readout:
        a caller of it cannot be routed onto the ONE-STEP Collins readout by
        accident, whatever they pass.

        RESTATED IN WP-C3 ROUND 2 (VERIFY-WP-C3 D8).  WP-B4 wrote this as
        "it gained nothing", asserting the absence of a ``transport``
        parameter.  5.49.0 gives it one -- but it selects which quadrature
        carries the beam to the STOP PLANE, not which readout runs: the
        standoff, the reconstruction, the Bluestein zoom and both guards are
        the same code on either setting, and ``_collins_focus_readout`` is
        still a separate entry this one never reaches.  The claim this test
        is about is therefore asserted directly -- the readout still HAS a
        stop plane on both settings, and still refuses the one-step form's
        own vocabulary -- instead of by the absence of a keyword.
        """
        import inspect
        p = inspect.signature(C.carrier_referenced_focus_readout).parameters
        assert 'on_collins_sampling' not in p, (
            'the standoff readout took the one-step form\'s Kelly guard '
            'keyword, which means it is no longer only the standoff readout')
        assert p['standoff'].default is None
        assert p['transport'].default == 'sziklas', (
            'the standoff readout moved its own default; this entry point has '
            'no other way back, and the resolver and guard around the leg '
            'are derived about the co-moving stop plane')
        # the STOP PLANE is still there on both settings, which is the
        # property "it is the Sziklas readout" actually means.
        n, dx = 128, 4e-6
        x = (np.arange(n) - n / 2) * dx
        X, Y = np.meshgrid(x, x)
        env = np.exp(-(X ** 2 + Y ** 2) / (120e-6 ** 2)).astype(
            np.complex128)
        for tr in ('sziklas', 'collins'):
            pd = {}
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                C.carrier_referenced_focus_readout(
                    env, -0.03, 0.03, 633e-9, dx, dx_out=2e-7, N_out=32,
                    standoff=1e-3, on_replica='ignore',
                    on_focus_containment='ignore', transport=tr,
                    _period_out=pd)
            assert pd['standoff'] == 1e-3 and 'containment' in pd, (
                f'transport={tr!r} did not go through a stop plane, so this '
                f'entry point is no longer the standoff readout: {pd!r}')

        # ``replica_fill``: its CONSEQUENCE, not its spelling.  WP-B4 pinned
        # the literal ``'repeat'`` here; WP-C5 item 3 moves that default to
        # ``'zero'`` in the same release, so a literal pin is a merge
        # conflict whose resolution is a coin-flip and whose loser ships a
        # test asserting the opposite of the source.  What this id is about
        # is that the standoff readout still OWNS the fill, so the default is
        # read off the signature and the behaviour it implies is asserted --
        # correct under BOTH values, on whichever one is live.
        fill = p['replica_fill'].default
        assert fill in ('repeat', 'zero'), (
            f'replica_fill grew a third value {fill!r} without this '
            f'vocabulary gate being told')
        n2, dx2, so2 = 128, 4e-6, 1e-4
        x2 = (np.arange(n2) - n2 / 2) * dx2
        X2, Y2 = np.meshgrid(x2, x2)
        e2 = np.exp(-(X2 ** 2 + Y2 ** 2) / (120e-6 ** 2)).astype(
            np.complex128)
        pd2 = {}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            F = np.asarray(C.carrier_referenced_focus_readout(
                e2, -0.03, 0.03, 633e-9, dx2, dx_out=2e-7, N_out=2048,
                standoff=so2, on_replica='ignore',
                on_focus_containment='ignore', _period_out=pd2))
        per = min(pd2['period'])
        win = 2048 * 2e-7
        assert win > per, (
            f'the fixture no longer reaches outside one Bluestein period '
            f'({win * 1e6:.4f} um against {per * 1e6:.4f} um), so it cannot '
            f'say what the fill does')
        edge = np.abs(F[:8, :8])
        if fill == 'zero':
            assert edge.max() == 0.0, (
                f"replica_fill defaults to 'zero' but the corner outside one "
                f"period came back at {edge.max():.6e}")
        else:
            assert edge.max() > 0.0, (
                f"replica_fill defaults to {fill!r} but the corner outside "
                f"one period came back empty")

    @pytest.mark.slow
    def test_the_chain_is_equal_bit_for_bit(self):
        """This chain does NOT move under the flip, and the reason is
        measured rather than hoped: every free leg of it resolves to the
        transfer-function form and its readout resolves to the Sziklas
        readout (K1 = 82.36 on the exit lattice), so the two spellings return
        the same array.

        The ONE thing that legitimately differs is the readout stage's new
        ``readout_route*`` keys, which are published on ``'collins'`` only --
        deliberately, because the ``'sziklas'`` ``stages`` list is a
        bit-identity key (WP-B4 sec. 4.2 digests ``repr(stages)``).  So the
        stage comparison is made with those three keys removed from the
        Collins side, and their presence is asserted separately, instead of
        the whole comparison being dropped.
        """
        env, dx, r_in, groups = _chain_fixture()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            a = C.propagate_traced_carrier_chain(
                env, groups, 1.31e-6, dx, r_in=r_in, ray_subsample=16,
                n_workers=1, final_distance=8e-3, traced_kwargs=_CHAIN_TKW,
                final_leg='paraxial',
                focus_readout=dict(dx_out=0.5e-6, N_out=64))
            b = C.propagate_traced_carrier_chain(
                env, groups, 1.31e-6, dx, r_in=r_in, ray_subsample=16,
                n_workers=1, final_distance=8e-3, traced_kwargs=_CHAIN_TKW,
                final_leg='paraxial', transport='sziklas',
                on_collins_sampling='error',
                focus_readout=dict(dx_out=0.5e-6, N_out=64))
        assert np.array_equal(np.asarray(a.field), np.asarray(b.field))
        assert a.dx == b.dx and a.R == b.R
        _route = ('readout_route', 'readout_route_k1', 'readout_route_reason')
        assert all(k in a.stages[-1] for k in _route)
        assert a.stages[-1]['readout_route'] == 'sziklas'
        assert a.stages[-1]['readout_route_k1'] > 1.0
        assert not any(k in b.stages[-1] for k in _route)
        # Every gap leg publishes its own ``collins_*`` readings on the
        # Collins spelling and nothing on the Sziklas one -- that is WP-B4's
        # design and is what ``on_collins_sampling``'s documentation
        # promises.  Both facts are asserted, then those keys are removed and
        # the REST of every stage is compared whole.
        _a = [{k: v for k, v in st.items()
               if not k.startswith('collins_') and k not in _route}
              for st in a.stages]
        assert any(st.get('collins_form') for st in a.stages), (
            'no stage published a collins_form, so this chain did not run '
            'the Collins leg and the comparison below is vacuous')
        assert not any(k.startswith('collins_')
                       for st in b.stages for k in st)
        assert _a == b.stages

    @pytest.mark.slow
    def test_the_multi_orchestrator_is_equal_bit_for_bit(self):
        env, dx, r_in, groups = _chain_fixture()
        fr = dict(dx_out=0.5e-6, N_out=64)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            a = C.propagate_traced_carrier_chain_multi(
                [{'field': env, 'carrier': r_in}], groups, 1.31e-6, dx,
                output_grid=fr, final_distance=8e-3, ray_subsample=16,
                n_workers=1, traced_kwargs=_CHAIN_TKW, final_leg='paraxial')
            b = C.propagate_traced_carrier_chain_multi(
                [{'field': env, 'carrier': r_in}], groups, 1.31e-6, dx,
                output_grid=fr, final_distance=8e-3, ray_subsample=16,
                n_workers=1, traced_kwargs=_CHAIN_TKW, final_leg='paraxial',
                transport='sziklas')
        assert np.array_equal(np.asarray(a.field), np.asarray(b.field))
        assert a.dx == b.dx and a.centre == b.centre


# ===========================================================================
# 4.  The same theorem, three independent ways
# ===========================================================================
class TestSameTheorem:
    def test_the_abcd_is_symplectic_for_every_reference(self):
        """``det = AD - BC == 1`` is what makes the envelope system a real
        optical system rather than an ansatz.  Checked over a decade-wide
        log-uniform sweep of both radii and the leg, so it is an identity and
        not a sampled agreement."""
        rng = np.random.default_rng(20260913)
        worst = 0.0
        for _ in range(20000):
            R_in = float(10.0 ** rng.uniform(-3, 0) * rng.choice([-1.0, 1.0]))
            R_rf = float(10.0 ** rng.uniform(-3, 0) * rng.choice([-1.0, 1.0]))
            z = float(10.0 ** rng.uniform(-4, -1) * rng.choice([-1.0, 1.0]))
            A, B, Cc, D = C._collins_envelope_abcd(R_in, z, R_rf)
            worst = max(worst, abs(A * D - B * Cc - 1.0))
        # Bar: 64 eps.  The determinant is a difference of products of numbers
        # whose ratio spans 10^3 here, so the cancellation floor is a few eps
        # times that condition number; measured worst 2026-09-13 = 1.1e-13 over
        # 20000 cells, and a sign or factor error is O(1), 13 decades up.
        assert worst < 64.0 * EPS * 1e3, worst

    @pytest.mark.parametrize('z', [12.0e-3, 20.0e-3])
    @pytest.mark.parametrize('gk', ['fresnel', 'auto'])
    def test_collins_on_the_co_moving_lattice_is_the_sziklas_step(
            self, env_conv, z, gk):
        """The factorisation claim, measured: asked for the lattice the Sziklas
        transport is forced onto, the Collins quadrature returns the Sziklas
        answer.  They are different quadratures of one integral, so the bar is
        their rounding, not zero.

        The two legs here are ones where BOTH quadratures are comfortably
        sampled (measured K1 = 0.34 and 0.16, and the chirp-Z's own period
        covers the co-moving window); the marginal legs, where they differ, are
        arbitrated against the direct sum in the next test rather than against
        each other.  Bar 1e-9 of peak; measured 2026-09-13: 7.6e-12 (12 mm)
        and 3.2e-12 (20 mm) on 'fresnel', 1.1e-11 and 3.3e-12 on 'auto'.  A
        factorisation error -- a wrong A, D or prefactor -- is O(1), eleven
        decades up."""
        m = (_R + z) / _R
        sz = C.propagate_carrier_referenced(env_conv, _R, z, _WL, _DX,
                                            gap_kernel=gk)
        co = C._collins_transport(env_conv, _R, z, _WL, _DX, _DX,
                                  dx_out=m * _DX, dy_out=m * _DX,
                                  N_out_x=_N, N_out_y=_N, R_ref=_R + z,
                                  gap_kernel=gk, on_collins_sampling='ignore')
        d = float(np.abs(co - np.asarray(sz.env)).max()
                  / np.abs(np.asarray(sz.env)).max())
        assert d < 1e-9, (z, gk, d)

    @pytest.mark.parametrize('z', [8.0e-3, -6.0e-3])
    def test_on_a_marginal_leg_the_direct_sum_sides_with_the_chirp_z(
            self, env_conv, z):
        """Where the two quadratures DISAGREE, something has to arbitrate, and
        it cannot be either of them.  The direct sum does: on both marginal
        legs (measured K1 = 0.57 at z = +8 mm and 1.08 at z = -6 mm) the
        chirp-Z reads ratio 1.000000 at every sampled point out to 1.6 beam
        radii, while the transfer-function quadrature departs -- 1.001321 at
        the outermost sample of the -6 mm leg, which is its own wrap-around,
        not the chirp-Z's.  This is why the test above compares the two only
        where both are sampled."""
        m = (_R + z) / _R
        sz = np.asarray(C.propagate_carrier_referenced(
            env_conv, _R, z, _WL, _DX, gap_kernel='fresnel').env)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            co = np.asarray(C._collins_transport(
                env_conv, _R, z, _WL, _DX, _DX, dx_out=m * _DX,
                dy_out=m * _DX, N_out_x=_N, N_out_y=_N, R_ref=_R + z,
                gap_kernel='fresnel', on_collins_sampling='ignore'))
        js = [_N // 2 + j for j in (0, 30, 60, 90, 120)]
        xs = [(j - _N / 2) * m * _DX for j in js]
        D = _collins_direct(env_conv, _R, z, _WL, _DX, xs, [0.0],
                            R_ref=_R + z)[0]
        rc = np.abs(np.array([co[_N // 2, j] for j in js]) - D) / np.abs(D)
        rs = np.abs(np.array([sz[_N // 2, j] for j in js]) - D) / np.abs(D)
        assert float(rc.max()) < 1e-9, rc
        assert float(rs.max()) > 10.0 * float(rc.max()), (rc, rs)

    def test_the_focus_readout_matches_the_analytic_gaussian_absolutely(
            self, env_conv):
        """ORACLE: ``_abcd_gauss``.  The comparison carries the ABSOLUTE phase
        -- piston and Gouy -- because the readout returns the field, not an
        envelope, and a piston-free comparison cannot see a conjugated Gouy
        phase (which is exactly pi at a focus)."""
        z = -_R
        w0 = _WL * abs(_R) / (np.pi * _W)
        dxo, nout = w0 / 8.0, 128
        E = np.asarray(C._collins_focus_readout(
            env_conv, _R, z, _WL, _DX, _DX, dx_out=dxo, N_out=nout,
            on_replica='error'))
        T, wz = _abcd_gauss(_grid(nout, dxo), _W, _R, z, _WL)
        assert wz == pytest.approx(w0, rel=2e-3)
        # Bar 1e-10.  The oracle is exact; the floor is the grid's own
        # truncation of the Gaussian at 2.73 w (amplitude exp(-7.46) = 5.8e-4
        # -- but the truncated tail contributes to the FOCAL field only at the
        # 1e-14 level because the transform of a Gaussian tail is itself a
        # tail).  Measured 2026-09-13: 2.17e-14 absolute relL2, and the power
        # ratio 1.00000000.  A conjugated Gouy phase reads 2.0 here.
        assert _rel_l2(E, T) < 1e-10, _rel_l2(E, T)
        p_in = float((np.abs(env_conv) ** 2).sum()) * _DX * _DX
        p_out = float((np.abs(E) ** 2).sum()) * dxo * dxo
        assert p_out / p_in == pytest.approx(1.0, abs=1e-6)

    def test_the_transform_equals_a_direct_summation_of_the_same_integral(
            self, env_conv):
        """ORACLE: ``_collins_direct`` -- the same integral by brute force, no
        FFT anywhere in it.  This is the arm that found the WP's own defect:
        the exact-kernel refinement applied over a degenerate reduced frame
        wrapped, and the core stayed right while the halo went 37x high at
        40 um and 520x at 100 um."""
        z = -_R
        dxo, nout = _WL * abs(_R) / (np.pi * _W) / 8.0, 128
        E = np.asarray(C._collins_focus_readout(
            env_conv, _R, z, _WL, _DX, _DX, dx_out=dxo, N_out=nout,
            on_replica='error'))
        js = [nout // 2 + j for j in (0, 3, 7, 13, 21, 34, 55)]
        xs = [(j - nout / 2) * dxo for j in js]
        D = _collins_direct(env_conv, _R, z, _WL, _DX, xs, [0.0])
        got = np.array([E[nout // 2, j] for j in js])
        rel = np.abs(got - D[0]) / np.abs(D[0])
        # The bar is the ORACLE's own floor, not a fixed number: a direct sum of
        # 1024^2 terms of magnitude |g| rounds at ``eps * sum|g|`` before the
        # prefactor, which relative to a SAMPLE is large wherever the sample is
        # small -- and these samples run from the peak down into the halo.  30x
        # that floor; measured 2026-09-13 the worst sample sits at 1.9x it
        # (2.5e-07 against a floor of 1.3e-07), and the defect the arm exists
        # for read 37x at the same radius.
        s_abs = float(np.abs(env_conv).sum()) * _DX * _DX / (_WL * abs(z))
        floor = EPS * s_abs / np.abs(D[0])
        assert np.all(rel < np.maximum(30.0 * floor, 1e-12)), (rel, floor)


# ===========================================================================
# 5.  Gate (a) -- the analytic-oracle matrix
# ===========================================================================
_GATE_A_NA = (0.03, 0.10, 0.30, 0.45)
_GATE_A_EXT = (1.5, 2.5, 4.0, 10.0)


def _gate_a_cell(na, ext, n=512, w_in=0.8e-3, lam=_WL):
    R0 = -w_in / na
    dx = 2.0 * ext * w_in / n
    z = -R0
    w0 = lam * abs(R0) / (np.pi * w_in)
    dxo, nout = w0 / 8.0, 64
    env = _gauss_env(n, dx, w_in)
    truth, _ = _abcd_gauss(_grid(nout, dxo), w_in, R0, z, lam)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        Es = C.carrier_referenced_focus_readout(
            env, R0, z, lam, dx, dx_out=dxo, N_out=nout, on_replica='ignore',
            on_focus_containment='ignore')
        Ec = C._collins_focus_readout(
            env, R0, z, lam, dx, dx, dx_out=dxo, N_out=nout,
            on_replica='ignore', on_collins_sampling='ignore')
    return (_piston_free_rel_l2(Es, truth), _piston_free_rel_l2(Ec, truth))


class TestGateAOracleMatrix:
    @pytest.mark.parametrize('na', _GATE_A_NA)
    @pytest.mark.parametrize('ext', _GATE_A_EXT)
    def test_collins_is_never_worse_than_sziklas(self, na, ext):
        """Gate (a).  Two-sided: the bar is the OTHER transport's own reading
        on the same cell, re-measured here, so nothing pins a build's number.
        Measured 2026-09-13 over the full 6 x 5 matrix: 0 of 30 cells worse,
        the ratio running 1.18x (NA 0.03, ext 1.5) to 148x (NA 0.45,
        ext 2.5)."""
        ls, lc = _gate_a_cell(na, ext)
        assert lc <= ls, (na, ext, ls, lc)

    @pytest.mark.parametrize('na', _GATE_A_NA)
    def test_the_small_extent_cells_are_materially_better(self, na):
        """"Materially better in the cells the small-extent branch exists for":
        ext 2.5 sits under the 3.695-beam-radius knee where
        ``_small_extent_focus_standoff_f`` takes over.  Bar 2x, against a
        measured 2.73x / 10.15x / 58.95x / 147.90x at NA 0.03 / 0.10 / 0.30 /
        0.45 on 2026-09-13 -- the smallest of them 1.4 decades over the bar."""
        ls, lc = _gate_a_cell(na, 2.5)
        assert ls / lc > 2.0, (na, ls, lc)

    @pytest.mark.parametrize('na', _GATE_A_NA)
    def test_on_a_wide_grid_collins_sits_on_the_truncation_floor(self, na):
        """The residual is the INPUT GRID's own truncation of the Gaussian and
        nothing else: at ext 10 the tail beyond the grid is exp(-100) and the
        reading is at the transform's rounding.  Measured 2026-09-13:
        1.3e-15 .. 4.5e-14 across NA, against sziklas' 3.1e-04 .. 2.9e-02."""
        ls, lc = _gate_a_cell(na, 10.0)
        assert lc < 1e-11, (na, lc)
        assert ls > 100.0 * lc, (na, ls, lc)


# ===========================================================================
# 6.  Gate (b) -- WP-A6's C1 mismatch matrix
# ===========================================================================
_MISMATCH_FR = (1.00, 0.99, 0.98, 0.95, 0.90)


@pytest.fixture(scope='module')
def mismatch_matrix():
    """WP-A6's own fixture: lambda 1.31 um, N 1024, w_in 1 mm, NA 0.05
    (R0 = -20 mm), ext 4.  ONE physical field, re-enveloped against each
    carrier; the peak ratio is against the matched (R/R0 = 1) readout, squared,
    exactly as ``p6c_mismatch.py`` defines it."""
    lam, n, w_in, na, ext = 1.31e-6, 1024, 1.0e-3, 0.05, 4.0
    R0 = -w_in / na
    dx = 2.0 * ext * w_in / n
    z, k = -R0, 2.0 * np.pi / lam
    w0 = lam * abs(R0) / (np.pi * w_in)
    dxo, nout = w0 / 8.0, 64
    x = _grid(n, dx)
    r2 = x[None, :] ** 2 + x[:, None] ** 2
    E_phys = np.exp(-r2 / w_in ** 2) * np.exp(1j * k * r2 / (2.0 * R0))
    truth, _ = _abcd_gauss(_grid(nout, dxo), w_in, R0, z, lam)
    out = {}
    for tr in ('sziklas', 'collins'):
        ref = None
        rows = []
        for fr in _MISMATCH_FR:
            R = fr * R0
            env = E_phys * np.exp(-1j * k * r2 / (2.0 * R))
            pd = {}
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                if tr == 'collins':
                    E = C._collins_focus_readout(
                        env, R, z, lam, dx, dx, dx_out=dxo, N_out=nout,
                        on_replica='ignore', on_collins_sampling='ignore',
                        _period_out=pd)
                else:
                    E = C.carrier_referenced_focus_readout(
                        env, R, z, lam, dx, dx_out=dxo, N_out=nout,
                        on_replica='ignore', on_focus_containment='ignore',
                        _period_out=pd)
            E = np.asarray(E)
            if ref is None:
                ref = E
            rows.append((fr,
                         float(np.abs(E).max() / np.abs(ref).max()) ** 2,
                         _piston_free_rel_l2(E, truth),
                         float(min(pd['period']))))
        out[tr] = rows
    return out


class TestGateBMismatchMatrix:
    def test_the_sziklas_column_reproduces_the_published_matrix(
            self, mismatch_matrix):
        """Fixture check, not a claim about the new transport: WP-A6 published
        1.000000 / 0.999514 / 0.998602 / 0.994304 / 0.985236 for these five
        rows, and VERIFY-A6 reproduced them to every printed digit.  If this
        arm drifts, the gate below is being scored on a different fixture."""
        got = [r[1] for r in mismatch_matrix['sziklas']]
        for g, want in zip(got, (1.000000, 0.999514, 0.998602, 0.994304,
                                 0.985236)):
            assert g == pytest.approx(want, abs=5e-5), got

    def test_collins_reads_unity_at_every_mismatch(self, mismatch_matrix):
        """Gate (b).  The prediction was "peak ratio 1.0000 at every R/R0,
        because the output pitch no longer depends on the carrier".  Measured
        2026-09-13: 1.000000 / 0.999999 / 0.999998 / 0.999986 / 0.999938 --
        1.0000 to four decimals on every row, against a sziklas column that
        falls to 0.985236.  Bar 1e-4, which the worst row clears by 1.6x and
        the sziklas column fails by 148x."""
        for fr, peak, _, _ in mismatch_matrix['collins']:
            assert abs(peak - 1.0) < 1e-4, (fr, peak)

    def test_the_readout_period_no_longer_depends_on_the_carrier(
            self, mismatch_matrix):
        """The mechanism behind the row above: the Sziklas period is
        ``N * dx_stop``, and the stop-plane pitch is resolved from the beam, so
        a carrier mismatch moves it.  The Collins period is ``lambda |z| / dx``
        of the INPUT grid -- the carrier is not in it.  Measured 2026-09-13:
        3353.60 um on all five rows (identical to the bit), against a sziklas
        column that moves."""
        pers = [r[3] for r in mismatch_matrix['collins']]
        assert len(set(pers)) == 1, pers
        szik = [r[3] for r in mismatch_matrix['sziklas']]
        assert max(szik) / min(szik) > 1.5, szik


# ===========================================================================
# 7.  Gate (c) -- a two-group chain against a brute-force ASM arm
# ===========================================================================
@pytest.fixture(scope='module')
def p5_arms():
    """The audit's own p5 method, re-implemented here: a 6 um waist 30 mm in
    front of two identical biconvex singlets 40 mm apart, read at HALF the
    paraxial image distance so the comparison plane is resolvable.  The BRUTE
    arm uses plain band-limited ASM on a grid fine enough to sample the full
    field and the SAME element call, so only the transport differs."""
    from lumenairy.elements import apply_real_lens_traced
    from lumenairy.glass import GLASS_REGISTRY
    from lumenairy.propagators.propagation import angular_spectrum_propagate
    from lumenairy.raytrace.seidel import system_abcd_prescription

    lam, ng = 1.31e-6, 1.5168
    GLASS_REGISTRY['_B4GLASS'] = (lambda wl: ng)
    sd = 10e-3

    def presc():
        return {'surfaces': [
            {'radius': 51.68e-3, 'glass_before': 'air',
             'glass_after': '_B4GLASS', 'semi_diameter': sd},
            {'radius': -51.68e-3, 'glass_before': '_B4GLASS',
             'glass_after': 'air', 'semi_diameter': sd}],
            'thicknesses': [5e-3], 'aperture_diameter': 2 * sd,
            'stop_index': 0}

    M, _, _, _ = system_abcd_prescription(presc(), lam)
    w0, z1 = 6.0e-6, 30e-3
    zR = np.pi * w0 ** 2 / lam
    r_in = z1 * (1.0 + (zR / z1) ** 2)
    w_l = w0 * np.sqrt(1.0 + (z1 / zR) ** 2)
    n = 2048
    dx = 2 * 3.0 * w_l / n
    env0 = _gauss_env(n, dx, w_l)
    tk = dict(amplitude_model='ray_density', preserve_input_phase='remap',
              remap_sampling='full')
    gap = 40e-3
    R_a = (M[0, 0] * r_in + M[0, 1]) / (M[1, 0] * r_in + M[1, 1])
    R_b = R_a + gap
    R_c = (M[0, 0] * R_b + M[0, 1]) / (M[1, 0] * R_b + M[1, 1])
    groups = [{'prescription': presc(), 'gap_before': 0.0},
              {'prescription': presc(), 'gap_before': gap}]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        Eb = apply_real_lens_traced(
            np.asarray(C.carrier_referenced_reconstruct(env0, r_in, lam, dx)),
            prescription=presc(), wavelength=lam, dx=dx, carrier=r_in,
            ray_subsample=2, **tk)
        Eb = angular_spectrum_propagate(np.asarray(Eb), gap, lam, dx)
        Eb = apply_real_lens_traced(Eb, prescription=presc(), wavelength=lam,
                                    dx=dx, carrier=R_b, ray_subsample=2, **tk)
        Eb = angular_spectrum_propagate(np.asarray(Eb), -R_c * 0.5, lam, dx)
        arms = {'brute': (np.abs(np.asarray(Eb)) ** 2, dx)}
        for tr in ('sziklas', 'collins'):
            r = C.propagate_traced_carrier_chain(
                env0, groups, lam, dx, r_in=r_in, ray_subsample=2,
                final_distance=-R_c * 0.5, final_leg='paraxial',
                traced_kwargs=tk, carrier_reference='sphere', transport=tr)
            arms[tr] = (np.abs(np.asarray(r.field)) ** 2, float(r.dx))
            arms[tr + '_res'] = r
    return arms


def _p5_reductions(I, d):
    n = I.shape[0]
    xx = (np.arange(n) - n / 2) * d
    t = I.sum()
    return (float(I.sum() * d * d),
            float((I.sum(0) * xx).sum() / t),
            float(np.sqrt((I * ((xx[None, :]) ** 2
                                + (xx[:, None]) ** 2)).sum() / t)))


@pytest.mark.slow
class TestGateCTwoGroupChain:
    def test_both_transports_reproduce_the_audit_s_own_readings(self, p5_arms):
        """Gate (c).  The audit recorded power ratio 1.000067 and r2m 1.13882
        vs 1.14389 mm (0.44 %) for the shipped transport on this fixture;
        ``'collins'`` must agree AT LEAST AS CLOSELY.  Measured 2026-09-13:
        both arms read power ratio 1.000067 and r2m 1.138817 mm against a brute
        1.143892 mm, i.e. 0.444 %."""
        pb, cb, rb = _p5_reductions(*p5_arms['brute'])
        best = None
        for tr in ('sziklas', 'collins'):
            p, c, r = _p5_reductions(*p5_arms[tr])
            assert p / pb == pytest.approx(1.0, abs=5e-4), (tr, p / pb)
            assert abs(r - rb) / rb < 0.01, (tr, r, rb)
            assert abs(c) < 1e-8, (tr, c)
            if best is None:
                best = abs(r - rb) / rb
            else:
                assert abs(r - rb) / rb <= best + 1e-12, (r, rb, best)

    def test_the_two_transports_agree_on_this_chain(self, p5_arms):
        """``'collins'`` evaluates the same integral by the same quadrature the
        default does, so the two agree -- WHERE THAT PREMISE HOLDS, which this
        id now measures instead of assuming.

        RESTATED 2026-09-19 (VERIFY-WAVE5-HYGIENE2 V-D19).  This was one
        ``np.array_equal`` on a 2048x2048 chain.  Bit equality is true only
        while every leg stays in the TRANSFER-FUNCTION half of the quadrature
        split, which is a MEASURED condition the assertion did not make --
        testing-standards shape S1/S5, an exact comparison conditioned on a
        reading that straddles a threshold.  It was not hypothetical: in one
        48-file sweep the two arms differed in the sixth significant figure
        (~8.5e-06 relative), which is a QUADRATURE SWITCH and not round-off.
        That failure did NOT reproduce -- four sweep-scale runs and two
        isolated runs, one failure, and both prefixes ending here pass -- and
        it is not attributable to this branch; the fragility is real either
        way, and by the repository's own standard ("flaky = bad math") a green
        rerun is not the answer.

        THE CLAIM IS NOW IN TWO PARTS.

        PREMISE, asserted first and failing on its own terms: every Collins
        leg took the transfer-function form, AND its decision reading is
        clear of the threshold by a margin.  The switch happens at
        ``max(K1, K3) > 1``; MEASURED on this fixture 2026-09-19,
        ``K1 = 1.9167``, ``K3 = 2.1387``, so the margin is 2.139 -- 114 %
        above the threshold.  These readings are products of about ten
        float64 quantities, so their cross-build spread is ~1e-15 relative;
        the premise bar of 1.5 sits fifteen decades above that spread and
        below the measurement, so what it can report is a FIXTURE walking
        toward the threshold, which is exactly the thing that would make the
        claim below conditional again.

        CLAIM: the two fields then agree to a DERIVED bar rather than to the
        bit.  Under the premise both arms run the same code, so the only
        admissible difference is a last-bit re-association; ``1e3 * eps`` of
        the peak bounds that generously for a 2048-point pairwise reduction
        (``log2(2048) = 11``).  MEASURED here: exactly 0.0, bit-identical.
        The gap on the other side is the failure this exists to catch, the
        quadrature switch at 8.5e-06 of peak -- 7.6 decades above the bar.
        """
        a = np.asarray(p5_arms['sziklas_res'].field)
        b = np.asarray(p5_arms['collins_res'].field)

        # --- PREMISE ---------------------------------------------------
        stages = [st for st in p5_arms['collins_res'].stages
                  if st.get('collins_form')]
        forms = [st['collins_form'] for st in stages]
        assert forms and set(forms) == {'tf'}, (
            f"PREMISE: the Collins arm did not stay in the transfer-function "
            f"half of the quadrature split (forms {forms}); the two arms are "
            f"then evaluating the same integral by DIFFERENT quadratures and "
            f"the agreement below is not a theorem about this chain")
        margins = [max(max(st.get('collins_k1') or (0.0, 0.0)),
                       max(st.get('collins_k3') or (0.0, 0.0)))
                   for st in stages]
        assert min(margins) > 1.5, (
            f"PREMISE: a leg's quadrature decision reads {min(margins):.4f}, "
            f"within 50 % of the threshold of 1 (measured 2.1387 on this "
            f"fixture 2026-09-19).  A reading that close is one fixture "
            f"change from flipping the route, and the agreement below would "
            f"then be conditional on a measurement no assertion makes.")

        # --- CLAIM -----------------------------------------------------
        peak = float(np.max(np.abs(a)))
        assert peak > 0.0, "PREMISE: the chain returned an empty field"
        bar = 1e3 * float(np.finfo(np.float64).eps) * peak
        got = float(np.max(np.abs(a - b)))
        assert got <= bar, (
            f"the two transports disagree by {got:.6e} (={got / peak:.3e} of "
            f"peak) against a derived re-association bar of {bar:.6e}.  Both "
            f"legs report the transfer-function form, so they ran the same "
            f"code: a difference at this scale is a quadrature or quadrature-"
            f"ORDER change, not round-off.")


# ===========================================================================
# 8.  Gate (d) -- the multi orchestrator
# ===========================================================================
@pytest.mark.slow
class TestGateDMulti:
    @pytest.mark.parametrize('tr', ['sziklas', 'collins'])
    def test_k1_reduces_to_the_single_congruence_chain(self, tr):
        """Gate (d), first half.  Compared with a tolerance rather than
        ``array_equal`` because both arms are live FFT work; measured margin
        2026-09-13 on both transports: exactly 0.0."""
        env, dx, r_in, groups = _chain_fixture()
        fr = dict(dx_out=0.5e-6, N_out=64)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            single = C.propagate_traced_carrier_chain(
                env, groups, 1.31e-6, dx, r_in=r_in, ray_subsample=16,
                n_workers=1, final_distance=8e-3, traced_kwargs=_CHAIN_TKW,
                final_leg='paraxial', focus_readout=fr, transport=tr)
            multi = C.propagate_traced_carrier_chain_multi(
                [{'field': env, 'carrier': r_in}], groups, 1.31e-6, dx,
                output_grid=fr, final_distance=8e-3, ray_subsample=16,
                n_workers=1, traced_kwargs=_CHAIN_TKW, final_leg='paraxial',
                transport=tr)
        A, B = np.asarray(single.field), np.asarray(multi.field)
        assert A.shape == B.shape and A.dtype == B.dtype
        assert float(np.abs(A - B).max()) <= 1e-10 * float(np.abs(A).max())
        assert multi.congruences[0]['stages'] == single.stages

    @pytest.mark.parametrize('tr', ['sziklas', 'collins'])
    def test_k2_is_the_hand_summed_pair(self, tr):
        """Gate (d), second half: with two congruences the orchestrator is only
        doing bookkeeping, so the recombination must be a plain add of two
        independent chain runs.  Measured 2026-09-13 on both transports:
        exactly 0.0."""
        env, dx, r_in, groups = _chain_fixture()
        specs = [{'field': env, 'carrier': r_in, 'name': 'a'},
                 {'field': 0.5 * env, 'carrier': r_in * 1.02, 'name': 'b'}]
        fr = dict(dx_out=0.5e-6, N_out=64, on_replica='ignore')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            k2 = C.propagate_traced_carrier_chain_multi(
                specs, groups, 1.31e-6, dx, output_grid=fr, readout_tile=None,
                final_distance=8e-3, ray_subsample=16, n_workers=1,
                traced_kwargs=_CHAIN_TKW, final_leg='paraxial', transport=tr)
            hand = [np.asarray(C.propagate_traced_carrier_chain(
                s['field'], groups, 1.31e-6, dx, r_in=s['carrier'],
                ray_subsample=16, n_workers=1, final_distance=8e-3,
                traced_kwargs=_CHAIN_TKW, final_leg='paraxial',
                focus_readout=fr, transport=tr).field) for s in specs]
        S = hand[0] + hand[1]
        assert float(np.abs(np.asarray(k2.field) - S).max()) \
            <= 1e-10 * float(np.abs(S).max())


# ===========================================================================
# 9.  The Kelly sampling guard
# ===========================================================================
def _stats(env, R, z, dx, dx_out, n_out, **kw):
    st = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        C._collins_transport(env, R, z, _WL, dx, dx, dx_out=dx_out,
                             dy_out=dx_out, N_out_x=n_out, N_out_y=n_out,
                             R_ref=np.inf, on_collins_sampling='ignore',
                             stats_out=st, **kw)
    return st


class TestKellyGuard:
    def test_the_conditions_are_ratios_against_the_nyquist_rate(self,
                                                                env_conv):
        """K1 and K2 are ``2 d nu_max / lambda``: at 1 the binding content sits
        exactly at the sample rate.  Checked by construction rather than by
        reading the code -- doubling the input pitch doubles K1 (the pre-chirp
        is sampled half as often) and doubling the output pitch doubles K2."""
        z = -_R
        dxo = _WL * abs(_R) / (np.pi * _W) / 8.0
        a = _stats(env_conv, _R, z, _DX, dxo, 64)
        b = _stats(_gauss_env(_N, 2 * _DX, _W), _R, z, 2 * _DX, dxo, 64)
        c = _stats(env_conv, _R, z, _DX, 2 * dxo, 64)
        assert b['k1'][0] / a['k1'][0] == pytest.approx(2.0, rel=0.05)
        assert c['k2'][0] / a['k2'][0] == pytest.approx(2.0, rel=1e-12)
        assert c['k1'][0] == pytest.approx(a['k1'][0], rel=1e-12)

    def test_the_support_radii_are_measured_not_geometric(self, env_conv):
        """The whole point of writing the guard against Kelly rather than
        against a geometric margin: the radius the condition is evaluated at is
        the field's own ``1 - _COLLINS_TAIL_FRAC``-power support, not the grid
        half-width.  Checked against the ANALYTIC containment radius of this
        Gaussian: the x-marginal of ``|exp(-r^2/w^2)|^2`` is a normal density of
        sigma = w/2, whose two-sided 1e-6 point is 4.892 sigma = 2.446 w =
        0.7338 mm.  Measured 2026-09-13: 0.7320 mm (one 4 um cell low, which is
        the lattice's own quantisation), against a grid half-width of 2.048 mm
        -- so the geometric form would read the condition 2.8x high."""
        st = _stats(env_conv, _R, 8e-3, _DX, 3.2e-6, _N)
        half = 0.5 * _N * _DX
        analytic = 2.446 * _W
        assert st['r_x'] == pytest.approx(analytic, abs=1.5 * _DX), st['r_x']
        assert st['r_x'] < 0.4 * half, (st['r_x'], half)

    def test_a_sampled_configuration_is_silent_and_right(self, env_conv):
        """First arm of the two-sided claim."""
        z = -_R
        dxo, nout = _WL * abs(_R) / (np.pi * _W) / 8.0, 128
        with warnings.catch_warnings(record=True) as W:
            warnings.simplefilter('always')
            E = np.asarray(C._collins_focus_readout(
                env_conv, _R, z, _WL, _DX, _DX, dx_out=dxo, N_out=nout,
                on_collins_sampling='error', on_replica='error'))
        assert not [w for w in W if 'Collins' in str(w.message)]
        T, _ = _abcd_gauss(_grid(nout, dxo), _W, _R, z, _WL)
        assert _rel_l2(E, T) < 1e-10

    @pytest.mark.parametrize('n,dx', [(256, 40e-6), (512, 20e-6),
                                      (1024, 10e-6)])
    def test_an_unsampled_configuration_fires_and_is_wrong(self, n, dx):
        """Second arm -- the FAIL-BEFORE, as a LADDER.  The state is ENGINEERED
        through the API (a coarse grid on a short leg drives the pre-chirp past
        Nyquist) rather than hoped for; the guard is shown to fire; the answer
        it fires on is shown to be wrong against the ANALYTIC Gaussian; and the
        complementary quadrature -- the one the transport actually selects at
        ``K1 > 1`` -- is shown to be exact on the same cell.  Each half is
        measured, none is inferred from another.

        Note the direct-summation oracle CANNOT arbitrate here and is not used:
        it evaluates the same DISCRETE sum, so it aliases identically.  Only a
        continuous truth can see this, which is what the analytic Gaussian is.

        Measured 2026-09-13, A = 0.9, B = 3 mm, w = 0.9 mm:

            n     dx      K1      chirp-Z relL2   transfer-function relL2
            256  40 um   49.694      1.13e+02            6.28e-11
            512  20 um   24.847      5.52e+01            6.28e-11
            1024 10 um   12.424      2.74e+01            6.28e-11
            2048  5 um    6.212      1.33e+01            6.28e-11

        -- the chirp-Z error tracks K1 (ratio 2.27 / 2.22 / 2.21 / 2.14), which
        is what says it IS the aliasing, and the transfer-function arm sits on
        the analytic oracle's own floor on every grid."""
        w, R, z = 0.9e-3, -30e-3, 3e-3
        A = 1.0 + z / R
        env = _gauss_env(n, dx, w)
        st = _stats(env, R, z, dx, abs(A) * dx, n)
        assert st['k1'][0] > 1.0, st['k1']
        with pytest.raises(RuntimeError, match='K1'):
            C._collins_transport(env, R, z, _WL, dx, dx, dx_out=abs(A) * dx,
                                 dy_out=abs(A) * dx, N_out_x=n, N_out_y=n,
                                 R_ref=R + z, on_collins_sampling='error')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            chirpz = np.asarray(C._collins_transport(
                env, R, z, _WL, dx, dx, dx_out=abs(A) * dx, dy_out=abs(A) * dx,
                N_out_x=n, N_out_y=n, R_ref=np.inf,
                on_collins_sampling='ignore'))
            tf = C._carrier_step_fast(env, R, z, _WL, dx, dx,
                                      gap_kernel='auto')
            tf_field = np.asarray(C.carrier_referenced_reconstruct(
                tf.env, tf.R, _WL, tf.dx))
        truth, _ = _abcd_gauss(_grid(n, abs(A) * dx), w, R, z, _WL)
        # Bars: the transfer-function arm against the analytic oracle's own
        # floor (6.3e-11, identical on all four grids, so it is the oracle and
        # not the grid), with two decades of slack; the chirp-Z arm above 1.0,
        # eleven decades up, so the two claims cannot be confused.
        assert _rel_l2(tf_field, truth) < 1e-8
        assert _rel_l2(chirpz, truth) > 1.0

    def test_the_period_is_the_input_grid_s_and_the_replica_guard_sees_it(
            self, env_conv):
        """K3 is disposed of by the EXISTING ``on_replica``, on this
        transport's own period, so the two guards cannot disagree."""
        z = -_R
        pd = {}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            C._collins_focus_readout(env_conv, _R, z, _WL, _DX, _DX,
                                     dx_out=1e-6, N_out=16,
                                     on_replica='ignore', _period_out=pd)
        assert pd['period'][0] == pytest.approx(_WL * abs(z) / _DX, rel=1e-12)
        wide = int(np.ceil(1.2 * pd['period'][0] / 1e-6))
        with pytest.raises(RuntimeError):
            C._collins_focus_readout(env_conv, _R, z, _WL, _DX, _DX,
                                     dx_out=1e-6, N_out=wide,
                                     on_replica='error')


class TestKernelRefinement:
    """``gap_kernel='exact'`` is a refinement over the REDUCED frame
    ``z_eff = B/A``, which is unbounded as a leg approaches the carrier's
    geometric focus.  K4 measures whether it is representable at all."""

    def test_the_refinement_is_applied_where_it_is_representable(self,
                                                                 env_conv):
        """K4 is small over the whole ordinary range, including well inside the
        near-focus zone: measured 2026-09-13 at A = 0.5 / 0.025 / 0.0025 /
        0.001 it reads 2.3e-07 / 8.9e-06 / 9.1e-05 / 2.3e-04, and the
        refinement runs at every one of them."""
        for z in (20e-3, 39e-3, 39.9e-3, -_R * 0.999):
            st = _stats(env_conv, _R, z, _DX, 2e-7, _N)
            assert st['k4'] < 1.0 and st['kernel'] == 'exact', (z, st['k4'])

    @pytest.mark.parametrize('z_frac', [1.0, 1.0 - 1e-8])
    def test_it_is_dropped_where_it_would_wrap_and_auto_says_so(self, env_conv,
                                                                z_frac):
        """The two cells it does fire on: the leg landing exactly on the
        geometric focus (``A == 0``, ``z_eff`` infinite) and one 4e-7 of the
        focal distance short of it (``A = 1e-08``, group delay 46.8 m against a
        2.048 mm grid half-width).  Measured 2026-09-13: k4 = inf and
        2.3e+04."""
        st = _stats(env_conv, _R, -_R * z_frac, _DX, 2e-7, _N)
        assert st['k4'] > 1.0 and st['kernel'] == 'fresnel', st['k4']

    def test_an_explicit_exact_is_refused_rather_than_downgraded(self,
                                                                 env_conv):
        with pytest.raises(ValueError, match='WRAP'):
            C._collins_transport(env_conv, _R, -_R, _WL, _DX, _DX,
                                 dx_out=2e-7, dy_out=2e-7, N_out_x=_N,
                                 N_out_y=_N, R_ref=np.inf,
                                 gap_kernel='exact',
                                 on_collins_sampling='ignore')

    def test_applying_it_anyway_is_demonstrably_wrong(self, env_conv):
        """FAIL-BEFORE for K4, and it is the defect this WP's own first cut
        shipped: the refinement over a degenerate reduced frame leaves the core
        right and destroys the halo.  Measured 2026-09-13 on the P2 battery's
        exit field: ratio 1.00 at the peak, 3.9x at 28 um, 37x at 40 um and
        520x at 100 um against the direct sum."""
        z = -_R * (1.0 - 1e-8)
        dxo, nout = 2e-7, 128
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            good = np.asarray(C._collins_transport(
                env_conv, _R, z, _WL, _DX, _DX, dx_out=dxo, dy_out=dxo,
                N_out_x=nout, N_out_y=nout, R_ref=np.inf,
                on_collins_sampling='ignore'))
            forced = C._collins_exact_kernel_correction(
                np.fft.fft2(np.ascontiguousarray(env_conv)),
                z / (1.0 + z / _R), _WL, _DX, _DX, (0.0, 0.0))
            bad = np.asarray(C._collins_transport(
                forced, _R, z, _WL, _DX, _DX, dx_out=dxo, dy_out=dxo,
                N_out_x=nout, N_out_y=nout, R_ref=np.inf,
                gap_kernel='fresnel', on_collins_sampling='ignore'))
        js = [nout // 2 + j for j in (0, 40, 60)]
        xs = [(j - nout / 2) * dxo for j in js]
        D = _collins_direct(env_conv, _R, z, _WL, _DX, xs, [0.0])
        g = np.array([good[nout // 2, j] for j in js])
        b = np.array([bad[nout // 2, j] for j in js])
        rg = np.abs(g - D[0]) / np.abs(D[0])
        rb = np.abs(b - D[0]) / np.abs(D[0])
        assert float(rg.max()) < 1e-9, rg
        assert float(rb.max()) > 100.0 * float(rg.max()), (rg, rb)


# ===========================================================================
# 10.  The quadrature selection is complementary, not tuned
# ===========================================================================
class TestQuadratureComplementarity:
    def test_the_two_forms_have_exactly_opposite_conditions(self, env_conv):
        """``K1 = 2 dx (|A| r/|B| + theta)/lambda``.  With ``r`` at the grid
        half-width that is ``N dx^2 / (lambda |z_eff|)`` -- so ``K1 <= 1`` is
        the chirp-Z's condition and ``K1 >= 1`` is, term for term, the
        transfer-function form's.  Checked as an identity on the ratio."""
        for z in (2e-3, 8e-3, 20e-3, 39e-3):
            st = _stats(env_conv, _R, z, _DX, 1e-6, 64)
            A = 1.0 + z / _R
            z_eff = z / A
            geom = _N * _DX ** 2 / (_WL * abs(z_eff))
            k1_geom = 2.0 * _DX * (abs(A) * (0.5 * _N * _DX) / abs(z)) / _WL
            assert k1_geom == pytest.approx(geom, rel=1e-12), (z, k1_geom,
                                                               geom)
            # ... and the MEASURED K1 is that same expression evaluated at the
            # measured support instead of the grid edge, so it is the geometric
            # reading scaled by r/(N dx/2) plus the envelope's own theta term.
            want = (k1_geom * st['r_x'] / (0.5 * _N * _DX)
                    + 2.0 * _DX * st['theta_x'] / _WL)
            assert st['k1'][0] == pytest.approx(want, rel=1e-12), (z, st['k1'])

    @pytest.mark.parametrize('z', [12e-3, 20e-3])
    def test_where_both_hold_the_two_forms_agree(self, env_conv, z):
        """Where both quadratures are sampled the selection cannot introduce a
        step, so the rule needs no smoothing at its boundary.  Measured
        2026-09-13: 1.1e-11 of peak at z = 12 mm (K1 = 0.34) and 3.3e-12 at
        z = 20 mm (K1 = 0.16)."""
        m = (_R + z) / _R
        sz = C._carrier_step_fast(env_conv, _R, z, _WL, _DX, _DX,
                                  gap_kernel='auto')
        co = C._collins_transport(env_conv, _R, z, _WL, _DX, _DX,
                                  dx_out=m * _DX, dy_out=m * _DX,
                                  N_out_x=_N, N_out_y=_N, R_ref=_R + z,
                                  on_collins_sampling='ignore')
        d = float(np.abs(co - np.asarray(sz.env)).max()
                  / np.abs(np.asarray(sz.env)).max())
        assert d < 1e-6, (z, d)

    def test_the_selected_form_is_published(self, env_conv):
        diag = {}
        C._collins_carrier_leg(env_conv, _R, 2e-3, _WL, _DX, _DX,
                               on_collins_sampling='ignore', diag=diag)
        assert diag['collins_form'] == 'tf'
        diag2 = {}
        C._collins_carrier_leg(env_conv, _R, 39e-3, _WL, _DX, _DX,
                               on_collins_sampling='ignore', diag=diag2)
        assert diag2['collins_form'] == 'chirp-z'


# ===========================================================================
# 11.  The near-focus apparatus is never entered
# ===========================================================================
class TestNoNearFocusApparatus:
    """WP-A6 sec. 6.1's central claim: ``m -> 0`` stops being a singularity.
    Proved by POISONING every entry point of the focus machinery -- the same
    instrument WP-A24 used to disprove an attribution -- rather than by reading
    the call graph."""

    _POISON = ('_propagate_carrier_focus_crossing', '_axis_bridge',
               '_default_focus_standoff', '_small_extent_focus_standoff_f',
               '_beam_containment_standoff')

    def test_a_near_focus_leg_runs_with_the_whole_apparatus_poisoned(
            self, env_conv, monkeypatch):
        def boom(*a, **k):
            raise AssertionError('near-focus apparatus entered')
        for name in self._POISON:
            monkeypatch.setattr(C, name, boom)
        for z in (39e-3, 39.9e-3, -_R, 41e-3):
            cr = C.propagate_carrier_referenced(
                env_conv, _R, z, _WL, _DX, transport='collins',
                on_collins_sampling='ignore')
            assert np.isfinite(np.abs(np.asarray(cr.env)).max())
        C._collins_focus_readout(env_conv, _R, -_R, _WL, _DX, _DX,
                                 dx_out=2e-6, N_out=64, on_replica='ignore')

    def test_the_same_poison_fires_on_the_sziklas_transport(self, env_conv,
                                                            monkeypatch):
        """The falsifier: without it the test above could pass because the
        poison never had a chance to fire.

        ``transport='sziklas'`` NAMED (WP-C3).  This arm's whole content is
        that the focus machinery IS entered by the transport that needs it,
        and since 5.49.0 that transport is reached by naming it rather than by
        passing nothing.  The claim is unchanged; what moved is which spelling
        selects the machinery.
        """
        def boom(*a, **k):
            raise AssertionError('near-focus apparatus entered')
        monkeypatch.setattr(C, '_propagate_carrier_focus_crossing', boom)
        with pytest.raises(AssertionError, match='near-focus'):
            C.propagate_carrier_referenced(env_conv, _R, -_R, _WL, _DX,
                                           transport='sziklas')

    def test_the_default_transport_does_not_enter_the_apparatus_at_all(
            self, env_conv, monkeypatch):
        """And the 5.49.0 default runs the same leg with the whole apparatus
        poisoned -- the statement the flip actually makes, on the spelling a
        caller now gets by default."""
        def boom(*a, **k):
            raise AssertionError('near-focus apparatus entered')
        for name in self._POISON:
            monkeypatch.setattr(C, name, boom)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            cr = C.propagate_carrier_referenced(env_conv, _R, -_R, _WL, _DX)
        assert np.isfinite(np.abs(np.asarray(cr.env)).max())

    def test_the_collapsing_pitch_is_floored_by_the_measured_box(self,
                                                                 env_conv):
        """What replaces the apparatus: the output pitch cannot follow ``|A|``
        to zero, because the floor carries the leg's own diffraction
        ``2|B| theta/N``.  Measured 2026-09-13 at 0.1 mm before the focus:
        co-moving 0.0100 um against a resolved 0.2777 um, 28x."""
        z = 39.9e-3
        A = 1.0 + z / _R
        diag = {}
        cr = C._collins_carrier_leg(env_conv, _R, z, _WL, _DX, _DX,
                                    on_collins_sampling='ignore', diag=diag)
        assert diag['collins_dx_floor_hit'] is True
        assert cr.dx > 5.0 * abs(A) * _DX, (cr.dx, abs(A) * _DX)

    def test_the_reference_goes_flat_exactly_at_the_geometric_focus(
            self, env_conv):
        """And the other half: at ``R + z == 0`` the RAY carrier is degenerate
        while the true wavefront is flat, so the transport references to
        infinity there and says so."""
        diag = {}
        cr = C._collins_carrier_leg(env_conv, _R, -_R, _WL, _DX, _DX,
                                    on_collins_sampling='ignore', diag=diag)
        assert diag['collins_flat_reference'] is True
        assert np.isinf(cr.R)


# ===========================================================================
# 12.  The readout period stops being a function of the resolved leg
# ===========================================================================
class TestReadoutPeriodDecoupling:
    def test_the_period_follows_the_input_grid_and_the_leg_only(self,
                                                                env_conv):
        """WP-A25's coupling, removed: the Sziklas period is ``N dx_stop`` and
        ``dx_stop`` is proportional to a standoff resolved from the BEAM, so
        changing the beam changes the faithful window.  Here the period is
        ``lambda |z| / dx``.  Checked by changing the beam and holding the grid:
        the Collins period does not move at all, the Sziklas one does."""
        z = -_R
        per = {}
        for w in (_W, 0.6 * _W):
            env = _gauss_env(_N, _DX, w)
            for tr in ('sziklas', 'collins'):
                pd = {}
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    if tr == 'collins':
                        C._collins_focus_readout(
                            env, _R, z, _WL, _DX, _DX, dx_out=2e-6, N_out=16,
                            on_replica='ignore', _period_out=pd)
                    else:
                        C.carrier_referenced_focus_readout(
                            env, _R, z, _WL, _DX, dx_out=2e-6, N_out=16,
                            on_replica='ignore',
                            on_focus_containment='ignore', _period_out=pd)
                per.setdefault(tr, []).append(float(min(pd['period'])))
        assert per['collins'][0] == per['collins'][1]
        assert per['sziklas'][0] != per['sziklas'][1]

    def test_replica_fill_is_inert_on_a_window_inside_one_period(self,
                                                                 env_conv):
        """The WP-A25 demonstration, at the level this transport changes it:
        with the window at 6.4 % of one period the two fills are the SAME
        ARRAY, so the knob that repaired the P2 battery cell has nothing left
        to repair.  Measured on the battery cell 2026-09-13: 2.0626 periods and
        FWHM 18.500 -> 20.500 um / EE2w 0.9970 -> 0.4953 under 'sziklas', and
        18.500 um / 0.9970 under BOTH fills under 'collins'."""
        z = -_R
        kw = dict(dx_out=2e-6, N_out=64, on_replica='ignore')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            a = np.asarray(C._collins_focus_readout(
                env_conv, _R, z, _WL, _DX, _DX, replica_fill='repeat', **kw))
            b = np.asarray(C._collins_focus_readout(
                env_conv, _R, z, _WL, _DX, _DX, replica_fill='zero', **kw))
        assert np.array_equal(a, b)


# ===========================================================================
# 13.  VERIFY-WP-B4 -- the chirp-Z's OUTPUT PERIOD on a leg
#
# A leg has no ``on_replica``, so K3 -- the condition that the returned window
# fit inside one chirp-Z period -- is the leg's own to dispose of, and it is
# the condition that actually complements the transfer-function form:
# ``K3 * K_tf = 2 dx theta / lambda <= 1`` because theta is read from the
# envelope's own SAMPLED spectrum.  ``K1`` is a weaker statement (``K3 >= K1``
# on every leg lattice), so selecting on K1 alone left a band where the chirp-Z
# ran on a window several periods wide.
# ===========================================================================
def _leg_k3(N, dx_out, wavelength, B, dx):
    return N * float(dx_out) / (wavelength * abs(float(B)) / float(dx))


def _leg(env, R, z, dx, **kw):
    d = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        cr = C._collins_carrier_leg(env, R, z, _WL, dx, dx,
                                    on_collins_sampling='ignore', diag=d,
                                    **kw)
    return cr, d


class TestLegPeriodCondition:
    @pytest.mark.parametrize('z', [-30e-3, -20e-3, -12e-3, 12e-3, 20e-3,
                                   30e-3, 39e-3])
    def test_a_leg_evaluates_the_form_that_is_SAMPLED_on_its_own_lattice(
            self, env_conv, z):
        """Decision, not reading: for each leg, measure K3 on the lattice the
        leg itself resolved, then require the published form to be the one
        that condition allows -- ``'tf'`` when the chirp-Z window exceeds one
        period AND the transfer-function form exists, ``'chirp-z'`` otherwise.
        When it is ``'tf'`` the returned envelope must BE
        :func:`_carrier_step_fast`'s, bit for bit, since that is the claim.

        Fail-before on 185d64cd as shipped (2026-09-13, a Gaussian at N = 512 /
        dx = 7.0312 um): the ``z = -20 mm`` leg reads K1 = 0.7600 (so the
        chirp-Z was selected) with K3 = 1.7842, and returns
        ``P_out/P_in = 1.3164`` -- 32 % of the returned power is a wrapped copy
        -- against relL2 0.5625 to the analytic Gaussian, silently.  The same
        leg now reads P_out/P_in = 1.000000 and relL2 1.96e-08."""
        cr, d = _leg(env_conv, _R, z, _DX)
        dxo = cr.dx if np.isscalar(cr.dx) else cr.dx[0]
        A = 1.0 + z / _R
        k3 = _leg_k3(_N, dxo, _WL, z, _DX)
        tf_exists = bool(A > 0.0 and np.isscalar(cr.R)
                         and np.isfinite(cr.R))
        want = 'tf' if (k3 > 1.0 and tf_exists) else 'chirp-z'
        assert d['collins_form'] == want, (z, k3, d['collins_form'])
        assert max(d['collins_k3']) == pytest.approx(k3, rel=1e-12)
        if want == 'tf':
            sz = C._carrier_step_fast(env_conv, _R, z, _WL, _DX, _DX,
                                      gap_kernel='auto')
            assert np.array_equal(np.asarray(cr.env), np.asarray(sz.env))
            assert cr.dx == sz.dx and cr.R == sz.R

    @pytest.mark.parametrize('z', [-20e-3, 12e-3, 20e-3, 39e-3])
    def test_a_leg_conserves_power(self, env_conv, z):
        """Parseval on the lattice the leg chose.  A chirp-Z window wider than
        one period manufactures power (the replicas are real samples), so this
        is the sharpest single statement of the defect above: bar 1 % against a
        measured 0.0002 % here and 31.64 % on 185d64cd at z = -20 mm.  The
        floor is the grid's own truncation of the Gaussian (this fixture spans
        6.83 w, so exp(-2*6.83^2) = 1e-40) plus the transform's rounding."""
        cr, _ = _leg(env_conv, _R, z, _DX)
        dxo = cr.dx if np.isscalar(cr.dx) else cr.dx[0]
        p_in = float((np.abs(env_conv) ** 2).sum()) * _DX * _DX
        p_out = float((np.abs(np.asarray(cr.env)) ** 2).sum()) * dxo * dxo
        assert p_out / p_in == pytest.approx(1.0, abs=1e-2), (z, p_out / p_in)

    @pytest.mark.parametrize('scale,expect_refusal', [(0.5, False),
                                                      (2.0, True)])
    def test_the_period_condition_speaks_on_a_caller_chosen_lattice(
            self, env_conv, scale, expect_refusal):
        """The public single-step entry has no ``on_replica`` either, so a
        ``dx_out`` wide enough to take the window past one period has to be
        refused rather than returned silently.  The pitch is DERIVED from the
        running build's own period, so the two arms sit either side of the bar
        by construction rather than by a remembered number.

        Fail-before (185d64cd, 2026-09-13): ``on_collins_sampling='error'``
        raised nothing at K3 = 2.03, where the returned field reads relL2
        2.2374 against the analytic Gaussian."""
        z = 20e-3
        period = _WL * abs(z) / _DX
        dxo = scale * period / _N
        k3 = _leg_k3(_N, dxo, _WL, z, _DX)
        assert (k3 > 1.0) is expect_refusal, (k3, expect_refusal)
        kw = dict(transport='collins', dx_out=dxo, carrier_out=np.inf,
                  on_collins_sampling='error')
        if expect_refusal:
            with pytest.raises(RuntimeError, match=r'K3 \(period\)'):
                C.propagate_carrier_referenced(env_conv, _R, z, _WL, _DX,
                                               **kw)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter('error')
                C.propagate_carrier_referenced(env_conv, _R, z, _WL, _DX,
                                               **kw)

    def test_the_selection_boundary_is_where_both_forms_hold(self, env_conv):
        """"The selection cannot introduce a step" is a claim about the
        boundary, so it is measured AT the boundary -- found by bisection on
        the running build, not assumed.  Both arms are evaluated on the same
        co-moving lattice with the same output reference, so their difference
        IS the step a crossing would introduce.

        Fail-before (185d64cd, 2026-09-13): the boundary sat at K1 = 1
        (z = 8.0108 mm on this fixture), where the two arms differ by 0.9999 of
        peak.  It now sits at K3 = 1 (z = 14.9177 mm), where they agree to
        2.9e-11 / 1.4e-11.  Bar 1e-8: three decades over the measurement and
        eight under the 1.0 a mis-placed boundary reads."""
        lo, hi = 5e-3, 39e-3
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            A = 1.0 + mid / _R
            if _leg_k3(_N, abs(A) * _DX, _WL, mid, _DX) > 1.0:
                lo = mid
            else:
                hi = mid
        zc = 0.5 * (lo + hi)
        forms = []
        for dz in (-2e-6, +2e-6):
            z = zc + dz
            A = 1.0 + z / _R
            _, d = _leg(env_conv, _R, z, _DX, gap_kernel='fresnel')
            forms.append(d['collins_form'])
            sz = np.asarray(C._carrier_step_fast(
                env_conv, _R, z, _WL, _DX, _DX, gap_kernel='fresnel').env)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                co = C._collins_transport(
                    env_conv, _R, z, _WL, _DX, _DX, dx_out=abs(A) * _DX,
                    dy_out=abs(A) * _DX, N_out_x=_N, N_out_y=_N,
                    R_ref=_R + z, gap_kernel='fresnel',
                    on_collins_sampling='ignore')
            step = float(np.abs(sz - co).max() / np.abs(sz).max())
            assert step < 1e-8, (z, step)
        assert forms == ['tf', 'chirp-z'], forms

    @pytest.mark.parametrize('z', [-30e-3, -6e-3, 2e-3, 8e-3, 20e-3, 39e-3])
    def test_k3_is_never_below_k1_on_a_leg_lattice(self, env_conv, z):
        """Why K3 is the condition to select on: the leg's pitch is
        ``max(|A| dx, 2 r_out/N)``, and at the floor ``K3`` IS ``K1``, so
        ``K3 >= K1`` identically and selecting on K1 alone can only ever be the
        weaker test.  An identity, so the bar is rounding."""
        cr, d = _leg(env_conv, _R, z, _DX)
        dxo = cr.dx if np.isscalar(cr.dx) else cr.dx[0]
        k3 = _leg_k3(_N, dxo, _WL, z, _DX)
        k1 = max(d['collins_k1'])
        assert k3 >= k1 * (1.0 - 64.0 * EPS), (z, k1, k3)
        if d['collins_dx_floor_hit']:
            assert k3 == pytest.approx(k1, rel=1e-9), (z, k1, k3)

    @pytest.mark.parametrize('z', [-30e-3, -6e-3, 2e-3, 8e-3, 20e-3, 39e-3])
    def test_the_two_conditions_are_exact_complements(self, env_conv, z):
        """``K3 * K_tf = 2 dx theta / lambda``, and ``theta`` is measured from
        the envelope's own SAMPLED spectrum so it cannot exceed the grid's
        Nyquist angle ``lambda/(2 dx)``.  The product is therefore at most 1:
        at least one of the two evaluations is always representable, and both
        are at the crossover.  That is what makes the selection a theorem
        rather than a threshold, and it is what ``K1`` does NOT satisfy
        (``K1 * K_tf`` carries an extra ``4 |z_eff| theta^2/(N lambda)`` that
        nothing bounds)."""
        r_x, r_y, th_x, th_y = C._collins_input_box(env_conv, _DX, _DX, _WL,
                                                    1e-6)
        th = max(th_x, th_y)
        A = 1.0 + z / _R
        z_eff = z / A
        k3_geom = _N * _DX ** 2 / (_WL * abs(z_eff))
        k_tf = 2.0 * abs(z_eff) * th / (_N * _DX)
        assert th <= _WL / (2.0 * _DX) * (1.0 + 64.0 * EPS), th
        assert k3_geom * k_tf == pytest.approx(2.0 * _DX * th / _WL, rel=1e-12)
        assert k3_geom * k_tf <= 1.0 + 64.0 * EPS, (z, k3_geom * k_tf)

    def test_the_leg_publishes_its_period_ratio_and_its_kernel(self,
                                                               env_conv):
        """Both quadratures publish the same key set, so a consumer reading a
        stage does not have to know which one ran to find a reading."""
        keys = {'collins_form', 'collins_k1', 'collins_k2', 'collins_k3',
                'collins_kernel', 'collins_flat_reference',
                'collins_dx_floor_hit'}
        _, d_tf = _leg(env_conv, _R, 2e-3, _DX)
        _, d_cz = _leg(env_conv, _R, 39e-3, _DX)
        assert d_tf['collins_form'] == 'tf' and set(d_tf) == keys
        assert d_cz['collins_form'] == 'chirp-z' and set(d_cz) == keys
        assert d_cz['collins_kernel'] in ('exact', 'fresnel')
        assert d_tf['collins_kernel'] is None


# ===========================================================================
# 14.  VERIFY-WP-B4 -- two claims the package states but does not measure
# ===========================================================================
class TestAbsolutePhaseThroughTheFocus:
    @pytest.mark.parametrize('dz', [-1e-3, -1e-6, 0.0, 1e-6, 1e-3])
    def test_the_gouy_phase_is_continuous_through_the_geometric_focus(
            self, env_conv, dz):
        """ORACLE: ``_abcd_gauss``, compared ABSOLUTELY.  ``A = 1 + z/R``
        crosses zero at ``z = -R``; the WP's claim is that this is an ordinary
        value, which is a statement about the PHASE as much as the amplitude.
        A conjugated Gouy phase, or a branch taken from ``sqrt((1 + z/q)^2)``
        instead of ``q/q2``, reads 2.0 here on the ``A < 0`` side and 0 on the
        other -- so the sweep is two-sided by construction.

        ``gap_kernel='fresnel'``, because the ABCD-Fresnel integral IS what
        this oracle is: the exact-kernel refinement is a different (and, this
        close to the focus, a large) statement, measured separately in
        :class:`TestKernelRefinementNearTheFocus`.  Bar 1e-9: measured
        2026-09-13 the worst cell reads 1.7e-14, and both failure modes it
        exists for are O(1)."""
        z = -_R + dz
        w0 = _WL * abs(_R) / (np.pi * _W)
        dxo, nout = w0 / 8.0, 128
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            E = np.asarray(C._collins_transport(
                env_conv, _R, z, _WL, _DX, _DX, dx_out=dxo, dy_out=dxo,
                N_out_x=nout, N_out_y=nout, R_ref=np.inf,
                gap_kernel='fresnel', on_collins_sampling='ignore'))
        T, _ = _abcd_gauss(_grid(nout, dxo), _W, _R, z, _WL)
        assert _rel_l2(E, T) < 1e-9, (dz, _rel_l2(E, T))
        # ... and the piston-free reading must not be better by decades, which
        # is what a global phase error looks like.
        assert _rel_l2(E, T) < 10.0 * max(_piston_free_rel_l2(E, T), 1e-16)


class TestKernelRefinementNearTheFocus:
    @pytest.mark.parametrize('dz', [1e-6, 1e-5, 1e-4, 1e-3, 5e-3])
    def test_the_abcd_fresnel_integral_is_exact_right_up_to_the_focus(
            self, env_conv, dz):
        """The half of the K4 story that is a property of the transport rather
        than of the refinement: with the refinement OFF, the Collins quadrature
        reads the analytic Gaussian at the transform's rounding from 5 mm away
        down to 1 um from the geometric focus -- 8.9e-15 to 1.7e-14, measured
        2026-09-13, and INDEPENDENT of N (identical at N = 512, 1024, 2048 and
        4096 on the same physical extent), which is what says it is the
        quadrature and not the grid.  Bar 1e-9.

        The companion statement, which is why this one is worth pinning
        separately: ``K4`` is below 1e-2 on every cell here -- the guard is
        silent -- while ``gap_kernel='auto'`` departs from the same oracle by
        up to 2.3e-03, growing linearly in ``|z_eff| = |B/A|`` exactly as the
        refinement's own dropped quartic ``k |z_eff| theta^4/8`` does.  K4
        bounds where the refinement WRAPS, not where it helps."""
        z = -_R + dz
        w0 = _WL * abs(_R) / (np.pi * _W)
        dxo, nout = w0 / 8.0, 128
        st = {}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            E = np.asarray(C._collins_transport(
                env_conv, _R, z, _WL, _DX, _DX, dx_out=dxo, dy_out=dxo,
                N_out_x=nout, N_out_y=nout, R_ref=np.inf,
                gap_kernel='fresnel', on_collins_sampling='ignore',
                stats_out=st))
        T, _ = _abcd_gauss(_grid(nout, dxo), _W, _R, z, _WL)
        assert _rel_l2(E, T) < 1e-9, (dz, _rel_l2(E, T))
        assert st['k4'] < 1.0, (dz, st['k4'])


class TestTiltedAndDecentredReadout:
    @pytest.mark.parametrize('L,M,x0,y0', [
        (0.0, 0.0, 0.0, 0.0),
        (20e-3, -12e-3, 0.0, 0.0),
        (46e-3, 0.0, 0.0, 0.0),
        (0.0, 0.0, 500e-6, -300e-6),
        (20e-3, -12e-3, 500e-6, -300e-6)])
    def test_a_tilted_decentred_congruence_through_the_collins_readout(
            self, L, M, x0, y0):
        """WP-B4 sec.5 item 7: "no fixture in this package reads a strongly
        tilted congruence THROUGH the Collins readout against an independent
        oracle".  This is that fixture.

        ORACLE, written here from Fresnel's shift theorem and nothing else.
        The input FIELD is ``G(x-x0, y-y0) exp(i k r^2/2R) exp(i k (Lx+My))``;
        about ``u = x - x0`` that is the ON-AXIS beam times a ramp
        ``alpha = L + x0/R`` and a constant, and ``f(u-u0) exp(i k alpha u) ->
        exp(i k alpha (x-u0)) exp(-i k alpha^2 z/2) F(x - u0 - alpha z)``, with
        ``u0 + alpha z = A x0 + L z``.  Absolute, so the ramp's own piston is in
        the comparison.  Measured 2026-09-13: 9.0e-15 on axis, 3.1e-14 at
        L = 46 mrad, 4.6e-13 with a 500/-300 um decentre.  Bar 1e-9, three
        decades over the worst cell and far under the O(1) a mis-signed
        ``centre_out`` screen reads."""
        z = -_R
        k = 2.0 * np.pi / _WL
        g = _grid(_N, _DX)
        env = (np.exp(-(((g - x0)[None, :] ** 2 + (g - y0)[:, None] ** 2)
                        / _W ** 2))
               * np.exp(1j * k * (L * g[None, :] + M * g[:, None]))
               ).astype(np.complex128)
        A = 1.0 + z / _R
        cen = (A * x0 + L * z, A * y0 + M * z)
        w0 = _WL * abs(_R) / (np.pi * _W)
        dxo, nout = w0 / 8.0, 128
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            E = np.asarray(C._collins_focus_readout(
                env, _R, z, _WL, _DX, _DX, dx_out=dxo, N_out=nout,
                centre_out=cen, on_replica='ignore',
                on_collins_sampling='ignore'))
        q = 1.0 / (1.0 / _R + 1j * _WL / (np.pi * _W ** 2))
        q2 = q + z
        ax, ay = L + x0 / _R, M + y0 / _R
        gx = _grid(nout, dxo) + cen[0]
        gy = _grid(nout, dxo) + cen[1]
        sx, sy = gx - (x0 + ax * z), gy - (y0 + ay * z)
        T = (np.exp(1j * k * (x0 * x0 / (2 * _R) + L * x0
                              + y0 * y0 / (2 * _R) + M * y0))
             * np.exp(1j * k * z) * (q / q2)
             * np.exp(0.5j * k * (sx[None, :] ** 2 + sy[:, None] ** 2) / q2)
             * np.exp(1j * k * (ax * (gx - x0)[None, :]
                                + ay * (gy - y0)[:, None]))
             * np.exp(-0.5j * k * (ax * ax + ay * ay) * z))
        assert _rel_l2(E, T) < 1e-9, (L, M, x0, y0, _rel_l2(E, T))
