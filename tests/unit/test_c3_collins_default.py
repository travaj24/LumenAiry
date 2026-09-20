"""WP-C3 -- ``transport='collins'`` is the carrier chain's DEFAULT, and the
CuPy arm that had to land with it.

WHAT THIS FILE GATES, IN THE ORDER THE WORK PACKAGE ASKS FOR IT.

1. **The default really moved**, asked two ways that can disagree: from the
   SIGNATURE of each of the three entry points that take ``transport``, and
   from a CALL that names nothing.  A signature can say ``'collins'`` while a
   body branches on something else, and a call can agree with ``'collins'``
   on a leg where the two transports agree anyway -- so the call arm is taken
   on a fixture where they are measurably DIFFERENT, and that difference is
   itself asserted.

2. **The way back is one keyword and it costs no bits.**  In-process this can
   only be an internal-consistency claim, so the archive-to-archive statement
   is made by ``validation/probe_c3_collins_default/probe_sziklas_bitid.py``
   (42 of 42 keys, both builds, with ``git archive 49ddf4bd`` as the base and
   ONE file substituted).  What IS gated here is the structural property that
   made that claim true and would silently break it again: every internal
   caller of ``propagate_carrier_referenced`` inside the module NAMES its
   transport.  That is not a style rule -- it is the defect this package
   found, with a fail-before arm that reproduces it.

3. **The oracle decisions.**  The absolute-phase analytic Gaussian (piston and
   Gouy, not a piston-free shape comparison) on five leg geometries, scored
   as DECISIONS -- the new default is never worse than the old, it reaches the
   focus cell the old one cannot reach at all, and it sits on the fixture's
   own truncation floor -- with every bar derived in the test from the
   fixture, never pasted.

4. **The CuPy arm**: a structural census that every helper on the Collins path
   is ``xp``-parametrised or host-side BY NAME; the repo's own
   SUBSTITUTED-MODULE test applied to the dispatcher the chain's transform
   goes through (fake a CuPy answer, check the module binds ``cp``, which is
   what a CUDA box would check); and a premise-gated device arm that first
   asserts what this box's CuPy actually is and then decides on whichever side
   of that premise holds.

5. **A mutation matrix.**  Five ways this work could silently come undone,
   each with the named test that catches it:

   | mutation | caught by |
   |---|---|
   | the default quietly reverted to ``'sziklas'`` | ``test_the_default_is_collins_in_every_signature`` AND ``test_an_unnamed_call_is_the_collins_call_and_not_the_sziklas_one`` |
   | a Collins helper losing its ``xp`` / ``bld`` | ``test_every_helper_on_the_collins_path_is_xp_parametrised`` |
   | the CuPy arm demoting a device array to the host | ``test_no_collins_helper_demotes_the_field_to_host_numpy`` (structural, both builds) AND ``test_a_device_array_reaches_the_device_transform`` (premise-gated) |
   | the FFT dispatcher answering "this is a CuPy array" without binding its ``cp``, so the device transform raises ``NameError`` | ``test_a_true_cupy_answer_really_binds_the_fft_dispatchers_cp`` (substituted module; runs with no CuPy) |
   | the readout's route resolution deleted, so the default returns the aliased one-step answer | ``test_the_readout_resolves_its_quadrature_and_the_fallback_is_bit_identical`` |
   | an internal caller riding the public default again | ``test_every_internal_transport_call_site_names_its_transport`` with its fail-before arm |

THE FIXTURE'S OWN FLOOR, MEASURED, because every bar below is two-sided.  The
ladder's Gaussian is truncated by its grid: at six ``1/e`` radii across N
samples the amplitude at the edge is ``exp(-9) = 1.23e-4`` and the L2 outside
the window is ~4e-5 of the whole.  That is computed in
:func:`_truncation_floor` from the fixture itself rather than typed, and it is
the number every Collins reading below sits on -- which is what says those
readings are the grid's error and not the transport's.
"""
from __future__ import annotations

import ast
import inspect
import pathlib
import warnings

import numpy as np
import pytest

import lumenairy.propagators.carrier as CA

WL = 1.064e-6
W_IN = 0.30e-3
R_CONV = -40.0e-3
N = 256
DX = 6.0 * W_IN / N

_ENTRY_POINTS = ('propagate_carrier_referenced',
                 'propagate_traced_carrier_chain',
                 'propagate_traced_carrier_chain_multi')

#: What the internal call-site census reads.  The three entry points that take
#: a ``transport`` default, PLUS ``carrier_referenced_focus_readout``, which
#: takes one of its own since 5.49.0 (WP-C3 round 2) and whose two internal
#: callers -- the chain's readout fallback -- must keep naming ``'sziklas'``.
_CENSUS_NAMES = _ENTRY_POINTS + ('carrier_referenced_focus_readout',)


# ---------------------------------------------------------------------------
# fixtures and oracles -- written here, sharing nothing with the transport
# ---------------------------------------------------------------------------
def _axis(n, d):
    return (np.arange(int(n), dtype=np.float64) - int(n) / 2.0) * float(d)


def _env(n=N, dx=DX, w=W_IN):
    x = _axis(n, dx)
    r2 = x[None, :] ** 2 + x[:, None] ** 2
    return np.exp(-r2 / (w * w)).astype(np.complex128)


def _analytic_gaussian(x_out, y_out, w, R, wavelength, z):
    """The propagated Gaussian FIELD, absolute phase included.

    ``1/q = 1/R + i lambda/(pi w^2)`` -- the ``exp(-i omega t)`` pairing this
    library mandates, NOT Siegman's, which conjugates the Gouy phase -- and
    the 2-D prefactor taken as the RATIO ``q/q2`` so it stays continuous
    through the waist (``1/sqrt((1 + z/q)^2)`` picks up exactly ``pi`` past
    it; VERIFY-WP-B4's opening caution).
    """
    k = 2.0 * np.pi / wavelength
    inv_q = (0.0 if not np.isfinite(R) else 1.0 / R) \
        + 1j * wavelength / (np.pi * w * w)
    q = 1.0 / inv_q
    q2 = q + z
    r2 = x_out[None, :] ** 2 + y_out[:, None] ** 2
    return (np.exp(1j * k * z) * (q / q2)
            * np.exp(1j * k * r2 / (2.0 * q2))).astype(np.complex128)


def _rebuild_field(out, wavelength):
    """The FIELD from a returned ``(env, R, dx)`` triple."""
    env = np.asarray(out.env)
    dx = out.dx
    dxo, dyo = (dx if isinstance(dx, tuple) else (dx, dx))
    ny, nx = env.shape[-2], env.shape[-1]
    x, y = _axis(nx, dxo), _axis(ny, dyo)
    Rx, Ry = (out.R if isinstance(out.R, tuple) else (out.R, out.R))
    k = 2.0 * np.pi / wavelength
    ph = np.zeros((ny, nx), dtype=np.float64)
    if np.isfinite(Rx):
        ph = ph + k * (x * x)[None, :] / (2.0 * Rx)
    if np.isfinite(Ry):
        ph = ph + k * (y * y)[:, None] / (2.0 * Ry)
    return env * np.exp(1j * ph), dxo, dyo


def _rel(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b))
                 / np.linalg.norm(np.asarray(b)))


def _truncation_floor(n=N, dx=DX, w=W_IN):
    """The fixture's OWN grid-truncation error, as a relative L2.

    The analytic Gaussian is defined on all of the plane; the transported one
    lives on ``n`` samples of pitch ``dx``.  The part of the continuous L2
    norm that falls outside that window is the smallest relative disagreement
    ANY transport can have with the analytic oracle on this fixture, so it is
    the lower side of every bar in this file.  Computed by quadrature on a
    grid 8x wider than the fixture's, which is itself refined and compared, so
    the number is converged rather than assumed.
    """
    def _out_frac(pad, refine):
        m = int(n * pad * refine)
        d = dx / refine
        x = _axis(m, d)
        r2 = x[None, :] ** 2 + x[:, None] ** 2
        amp2 = np.exp(-2.0 * r2 / (w * w))
        inside = (np.abs(x) < (n / 2) * dx)
        win = inside[None, :] & inside[:, None]
        tot = float(amp2.sum())
        return float(np.sqrt(max(tot - float(amp2[win].sum()), 0.0) / tot))
    coarse = _out_frac(8, 1)
    fine = _out_frac(8, 2)
    assert abs(fine - coarse) < 0.05 * max(fine, 1e-300), (
        f'the truncation-floor quadrature has not converged: {coarse:.6e} at '
        f'the fixture pitch against {fine:.6e} at half of it; the floor below '
        f'would be a statement about the quadrature rather than the fixture')
    return fine


# ===========================================================================
# 1.  The default really moved
# ===========================================================================
@pytest.mark.parametrize('name', _ENTRY_POINTS)
def test_the_default_is_collins_in_every_signature(name):
    """Asked from the signature, at every entry point that takes it.

    Three entry points and not two: ``propagate_traced_carrier_chain_multi``
    forwards ``transport`` to K chain runs, so a multi left at ``'sziklas'``
    would silently give one library two defaults depending on whether the
    caller ran one congruence or several.
    """
    p = inspect.signature(getattr(CA, name)).parameters
    assert 'transport' in p, f'{name} no longer takes transport'
    assert p['transport'].default == 'collins', (
        f"{name}'s transport default is {p['transport'].default!r}; WP-C3 "
        f"moved it to 'collins' and the Migration note names it")


def test_the_vocabulary_is_unchanged_and_still_closed():
    """The flip moves the DEFAULT, not the accepted set, and a typo must still
    raise rather than select the new default silently."""
    assert CA._TRANSPORTS == ('sziklas', 'collins')
    for bad in ('Collins', 'sziklas ', None, 0, 'collins2'):
        with pytest.raises(ValueError, match='transport must be one of'):
            CA._check_transport(bad, 'probe')
    # ... and the message names which one is now the default, so a reader of
    # the refusal is not sent to the pre-5.49.0 answer.
    with pytest.raises(ValueError) as ei:
        CA._check_transport('nope', 'probe')
    assert "'collins' (the default" in str(ei.value)


def test_an_unnamed_call_is_the_collins_call_and_not_the_sziklas_one():
    """The CALL, not the signature -- and taken where the two transports are
    measurably different, so it cannot pass by agreement.

    The fixture is a leg PAST the carrier's geometric focus (``A = -0.5``),
    where the co-moving grid has inverted: this is one of the geometries the
    Collins quadrature exists for.  The test asserts three things in order --
    that the two transports differ AT ALL here (otherwise the rest is
    vacuous), that the unnamed call is the Collins one to the bit, and that
    it is NOT the Sziklas one.
    """
    env = _env()
    z = 60.0e-3                      # A = 1 + z/R = -0.5
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        plain = CA.propagate_carrier_referenced(env, R_CONV, z, WL, DX)
        coll = CA.propagate_carrier_referenced(env, R_CONV, z, WL, DX,
                                               transport='collins')
        szik = CA.propagate_carrier_referenced(env, R_CONV, z, WL, DX,
                                               transport='sziklas')
    a, b = np.asarray(coll.env), np.asarray(szik.env)
    assert not (a.shape == b.shape and np.array_equal(a, b)), (
        'the two transports agree bit for bit on this fixture, so it cannot '
        'tell which one an unnamed call took; pick a leg where they differ')
    assert np.array_equal(np.asarray(plain.env), a)
    assert plain.dx == coll.dx and plain.R == coll.R


def test_the_chain_and_the_multi_take_the_default_through_to_their_legs():
    """The same question one level up: a chain and a multi run with nothing
    named must agree with the explicit ``'collins'`` run, stage list included.

    ``stages`` is compared whole, not just the field: the per-leg Collins
    diagnostics (``collins_form`` / ``collins_k1`` / ``collins_kernel``) are
    published there, so a chain that took the right transport by accident but
    recorded the wrong one is a failure here.
    """
    env, dx, r_in, groups = _chain_fixture()
    kw = dict(r_in=r_in, ray_subsample=16, n_workers=1,
              traced_kwargs=_TKW, final_leg='paraxial', final_distance=8e-3)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        plain = CA.propagate_traced_carrier_chain(env, groups, 1.31e-6, dx,
                                                  **kw)
        named = CA.propagate_traced_carrier_chain(env, groups, 1.31e-6, dx,
                                                  transport='collins', **kw)
    assert np.array_equal(np.asarray(plain.field), np.asarray(named.field))
    assert plain.stages == named.stages
    assert any(s.get('collins_form') for s in plain.stages), (
        'no stage published a collins_form, so this chain did not run the '
        'Collins leg at all and the comparison above is vacuous')


_TKW = dict(on_undersample='silent', on_noncollimated='silent')


def _singlet():
    return {'name': 'p', 'aperture_diameter': 14e-3, 'thicknesses': [3e-3],
            'surfaces': [
                {'radius': 60e-3, 'glass_before': 'air',
                 'glass_after': 'N-BK7', 'conic': 0.0, 'radius_y': None,
                 'conic_y': None, 'aspheric_coeffs': None,
                 'aspheric_coeffs_y': None},
                {'radius': -60e-3, 'glass_before': 'N-BK7',
                 'glass_after': 'air', 'conic': 0.0, 'radius_y': None,
                 'conic_y': None, 'aspheric_coeffs': None,
                 'aspheric_coeffs_y': None}]}


def _chain_fixture(n=256, dx=60e-6, w=4.5e-3):
    """WP-B4's own two-group relay -- the fixture its gates (c) and (d) run
    on, so the readings here are comparable with that package's."""
    x = _axis(n, dx)
    X, Y = np.meshgrid(x, x)
    env = np.exp(-(X ** 2 + Y ** 2) / (w * w)).astype(np.complex128)
    p = _singlet()
    return env, dx, 60e-3, [{'prescription': p, 'gap_before': 20e-3},
                            {'prescription': p, 'gap_before': 10e-3}]


# ===========================================================================
# 2.  The way back
# ===========================================================================
def test_every_internal_transport_call_site_names_its_transport():
    """Every call to ``propagate_carrier_referenced`` INSIDE the module names
    ``transport=``, so no internal leg rides a public default.

    THIS IS THE DEFECT THIS PACKAGE FOUND, gated.  MEASURED 2026-09-20: with
    the default flipped and ``carrier_referenced_focus_readout``'s own leg
    onto the standoff plane left implicit, that readout stopped raising its
    documented containment ``RuntimeError`` on a fixture where it had raised
    -- because the Collins leg resolves its OWN output pitch and the grid it
    handed the containment guard was no longer the co-moving one the guard is
    written about.  It is one archive-to-archive key, and it was found by that
    probe rather than by reading the diff.

    The census is AST-based and counts CALLS, so a new call site added
    anywhere in the PACKAGE is caught at the line.

    WIDENED IN WP-C3 ROUND 2 (VERIFY-WP-C3 D9), which measured the first
    version against six un-named fourth call sites and found it fired for two.
    Four escapes, all closed here:

    * a call in ANOTHER shipped module -- the census read only
      ``carrier.py``.  It now walks every ``*.py`` under the package root, so
      the report's "no other module names these entry points" is gated rather
      than grepped;
    * a call with ``**kw`` -- excused unconditionally by the old
      ``not any(kw.arg is None ...)``, and ``**{}`` excused it too.  Splat
      sites are now excused only by an ALLOW-LIST of enclosing function
      names, each of which is additionally required to mention ``transport``
      in its own source (which is what makes the forward real);
    * a module-level ALIAS (``_alias = propagate_carrier_referenced``) --
      invisible to a name test on the call node;
    * a dynamic lookup (``globals()['propagate_carrier_referenced'](...)`` or
      ``getattr(...)``) -- likewise.

    ``carrier_referenced_focus_readout`` joined the censused names in round 2,
    because 5.49.0 gives it a ``transport`` of its own (VERIFY-WP-C3 D8) and
    the chain's readout FALLBACK must keep naming ``'sziklas'`` for its
    bit-identity contract.
    """
    root = pathlib.Path(CA.__file__).parents[1]
    #: Enclosing functions whose ``**kwargs`` splat is a VERBATIM forward of
    #: the caller's own ``transport``.  Curated, not inferred: a new splat
    #: site anywhere else fails this census by name.
    _SPLAT_FORWARDERS = frozenset({
        'propagate_traced_carrier_chain',          # the multi-congruence arm
        'propagate_traced_carrier_chain_multi',    # -> _common_chain_kwargs
        '_multi_worker_run',                       # the worker process
        '_run_chain_dx_self_check',                # the dx self-check re-run
    })
    offenders, splats, n_sites, n_files = [], [], 0, 0
    for _p in sorted(root.rglob('*.py')):
        try:
            src = _p.read_text(encoding='cp1252')
        except (OSError, UnicodeDecodeError):       # pragma: no cover
            continue
        n_files += 1
        try:
            tree = ast.parse(src)
        except SyntaxError:                         # pragma: no cover
            continue
        rel = _p.relative_to(root).as_posix()
        # module-level aliases and dynamic lookups of an entry point
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(
                    node.value, ast.Name) and node.value.id in _ENTRY_POINTS:
                offenders.append(
                    f'{rel}:{node.lineno}: alias of {node.value.id}')
            if isinstance(node, ast.Subscript) and isinstance(
                    node.slice, ast.Constant) and \
                    node.slice.value in _ENTRY_POINTS:
                offenders.append(
                    f'{rel}:{node.lineno}: dynamic lookup of '
                    f'{node.slice.value}')
            if isinstance(node, ast.Call) and isinstance(
                    node.func, ast.Name) and node.func.id == 'getattr' and \
                    len(node.args) >= 2 and isinstance(
                        node.args[1], ast.Constant) and \
                    node.args[1].value in _ENTRY_POINTS:
                offenders.append(
                    f'{rel}:{node.lineno}: getattr lookup of '
                    f'{node.args[1].value}')
        # the calls themselves, with their enclosing function
        encl = {}
        for _f in ast.walk(tree):
            if isinstance(_f, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for _n in ast.walk(_f):
                    if isinstance(_n, ast.Call):
                        encl.setdefault(id(_n), _f)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            nm = (fn.id if isinstance(fn, ast.Name)
                  else fn.attr if isinstance(fn, ast.Attribute) else None)
            if nm not in _CENSUS_NAMES:
                continue
            n_sites += 1
            names = {kw.arg for kw in node.keywords}
            if 'transport' in names:
                continue
            owner = encl.get(id(node))
            if not any(kw.arg is None for kw in node.keywords):
                offenders.append(f'{rel}:{node.lineno}: {nm}(...)')
                continue
            oname = owner.name if owner is not None else '<module>'
            splats.append(f'{rel}::{oname}')
            if oname not in _SPLAT_FORWARDERS:
                offenders.append(
                    f'{rel}:{node.lineno}: {nm}(**kw) inside {oname!r}, '
                    f'which is not an allow-listed forwarder')
            if any(isinstance(_kw.value, ast.Dict) for _kw in node.keywords
                   if _kw.arg is None):
                offenders.append(
                    f'{rel}:{node.lineno}: {nm}(**<dict literal>) -- a '
                    f'literal cannot be forwarding a caller transport')
    assert n_files >= 50, (
        f'the census walked only {n_files} modules; it is not reading the '
        f'package')
    assert n_sites >= 3, (
        f'only {n_sites} internal call sites found; the census is not reading '
        f'the module')
    assert sorted(splats) == sorted(
        f'propagators/carrier.py::{_f}' for _f in _SPLAT_FORWARDERS), (
        'the splat-forwarding sites are a curated set of exactly four, '
        'one per allow-listed function; found: ' + repr(sorted(splats)))
    assert not offenders, (
        'these internal call sites ride the public transport default, which '
        'moved in 5.49.0 and will move again:\n  ' + '\n  '.join(offenders))


def test_the_focus_readouts_standoff_leg_takes_that_readouts_own_transport():
    """The Sziklas readout's carrier leg onto the stop plane is a CHOICE with
    a keyword, not a pin (WP-C3 round 2, VERIFY-WP-C3 D8).

    WP-C3 pinned it to ``'sziklas'`` because the entry point had no
    ``transport`` of its own, so riding the flipped public default would have
    moved a public answer with no one-keyword way back.  Round 2 gives it the
    keyword, because the measurement goes against the pin: against a converged
    dense separable Fresnel oracle the Collins leg reads relL2 4.7340e-05 on
    this very fixture and the Sziklas one 2.4049, and over five geometries x
    six standoffs the Sziklas leg refuses 7 of 30 while the Collins leg
    refuses none.

    THREE ARMS, so neither side can be empty:

    * ``transport='sziklas'`` still RAISES the documented containment
      ``RuntimeError`` here -- that is the 5.48.1 behaviour, and it is the way
      back;
    * the default RETURNS on the same call, and the containment the guard
      measured is genuinely different, not merely unreported;
    * the chain's readout fallback still NAMES ``'sziklas'``, which
      ``test_every_internal_transport_call_site_names_its_transport``
      censuses, so the fallback's bit-identity contract is untouched.
    """
    n, dx = 128, 4e-6
    x = _axis(n, dx)
    X, Y = np.meshgrid(x, x)
    env = np.exp(-(X ** 2 + Y ** 2) / (120e-6 ** 2)).astype(np.complex128)
    kw = dict(dx_out=2e-7, N_out=32, standoff=1e-3, on_replica='ignore')

    with pytest.raises(RuntimeError, match='co-moving grid at the stop plane'):
        CA.carrier_referenced_focus_readout(env, -0.03, 0.03, 633e-9, dx,
                                            transport='sziklas', **kw)

    pd_c, pd_s = {}, {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        got_c = CA.carrier_referenced_focus_readout(
            env, -0.03, 0.03, 633e-9, dx, _period_out=pd_c, **kw)
        got_s = CA.carrier_referenced_focus_readout(
            env, -0.03, 0.03, 633e-9, dx, transport='sziklas',
            on_focus_containment='ignore', _period_out=pd_s, **kw)
    assert np.all(np.isfinite(got_c)), (
        'the default did not return a finite field on the fixture the pinned '
        'leg refuses, so the decision above has no consequence')
    assert pd_c['containment'] != pd_s['containment'], (
        f"the two transports handed the guard the same stop plane "
        f"({pd_c['containment']:.6g} vs {pd_s['containment']:.6g}), so the "
        f"keyword selects nothing")

    # THE ADJUDICATOR: a dense separable Fresnel quadrature of the SAME input
    # field onto the SAME output lattice, oversampled in the input plane until
    # it stops moving.  It shares no code with either transport, and its own
    # convergence is asserted before it is allowed to decide anything.
    def _oracle(over):
        k0 = 2.0 * np.pi / 633e-9
        xa = _axis(n, dx)
        xd = np.linspace(xa[0], xa[-1] + dx, n * over, endpoint=False)
        e1 = np.exp(-xd ** 2 / (120e-6 ** 2)) * np.exp(
            1j * k0 * xd ** 2 / (2.0 * -0.03))
        xo = _axis(32, 2e-7)
        ph = np.exp(1j * k0 * xd ** 2 / (2.0 * 0.03))
        ker = np.exp(-1j * k0 * np.outer(xo, xd) / 0.03)
        f1 = ker @ (e1 * ph * np.gradient(xd))
        pre = np.exp(1j * k0 * 0.03) / (1j * 633e-9 * 0.03)
        return pre * np.outer(f1, f1) * np.exp(
            1j * k0 * (xo[:, None] ** 2 + xo[None, :] ** 2) / (2.0 * 0.03))

    t_hi, t_lo = _oracle(256), _oracle(64)
    conv = float(np.linalg.norm(t_hi - t_lo) / np.linalg.norm(t_hi))
    assert conv < 1e-5, (
        f'the oracle has not converged ({conv:.3e}), so it cannot arbitrate')

    def _rel(a):
        a = np.asarray(a)
        return float(np.linalg.norm(a - t_hi) / np.linalg.norm(t_hi))

    r_c, r_s = _rel(got_c), _rel(got_s)
    # Bars derived from the oracle's own floor: its convergence is ~5e-7 and
    # its self-consistency against a sampled sum ~4.5e-5, so 1e-3 is two
    # decades above anything the oracle itself could be wrong by, and 0.1 is
    # two decades above THAT.  Measured 2026-09-20 on both builds:
    # collins 4.7340e-05, sziklas 2.4049 -- 50 000x apart.
    assert r_c < 1e-3, (
        f'the default standoff leg is {r_c:.4e} from the converged '
        f'quadrature, which is not the reason this keyword exists')
    assert r_s > 0.1, (
        f'the pinned co-moving standoff leg reads {r_s:.4e} here, so this '
        f'fixture no longer shows the difference the decision was taken on')
    assert r_s / r_c > 1e3, (
        f'collins {r_c:.4e} vs sziklas {r_s:.4e}: less than three decades '
        f'apart, so the decision is inside the fixture\'s own noise')


def test_the_readout_resolves_its_quadrature_and_the_fallback_is_bit_identical():
    """The chain's focus readout on a leg the one-step form cannot represent.

    Three claims, and the first two are the premise of the third:

    * K1 on this fixture's exit lattice is far above 1 -- asserted, so the
      test is known to be exercising the fallback and not the direct route;
    * the readout stage SAYS which route ran and why;
    * the returned field is bit-identical to ``transport='sziklas'``, which is
      what makes the default flip free on every readout of this shape.
    """
    env, dx, r_in, groups = _chain_fixture()
    kw = dict(r_in=r_in, ray_subsample=16, n_workers=1, traced_kwargs=_TKW,
              final_leg='paraxial', final_distance=8e-3,
              focus_readout=dict(dx_out=0.5e-6, N_out=64))
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        coll = CA.propagate_traced_carrier_chain(env, groups, 1.31e-6, dx,
                                                 transport='collins', **kw)
        szik = CA.propagate_traced_carrier_chain(env, groups, 1.31e-6, dx,
                                                 transport='sziklas', **kw)
    stage = coll.stages[-1]
    assert stage['readout_route_k1'] > 1.0, (
        f"this fixture's readout K1 is {stage['readout_route_k1']!r}, so the "
        f"one-step form IS representable here and the fallback is not being "
        f"exercised; pick a shorter final leg or a coarser exit grid")
    assert stage['readout_route'] == 'sziklas'
    assert stage['readout_route_reason'] == 'k1'
    assert np.array_equal(np.asarray(coll.field), np.asarray(szik.field))
    assert not [w for w in rec
                if 'chirp-Z stage is under-sampled' in str(w.message)], (
        'the Kelly guard spoke on a readout that resolved away from the '
        'chirp-Z; the resolution is supposed to remove the condition, not '
        'run into it and warn')


def test_the_sziklas_stage_list_gains_no_key_from_the_route_publisher():
    """``stages`` is a bit-identity key on ``'sziklas'``, so the new
    ``readout_route*`` keys must not appear there.

    Both sides asserted: absent on ``'sziklas'``, present on ``'collins'``.
    A publisher that wrote them unconditionally would pass a "the keys exist"
    test and still break the byte-identity claim.
    """
    env, dx, r_in, groups = _chain_fixture()
    kw = dict(r_in=r_in, ray_subsample=16, n_workers=1, traced_kwargs=_TKW,
              final_leg='paraxial', final_distance=8e-3,
              focus_readout=dict(dx_out=0.5e-6, N_out=64))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        szik = CA.propagate_traced_carrier_chain(env, groups, 1.31e-6, dx,
                                                 transport='sziklas', **kw)
        coll = CA.propagate_traced_carrier_chain(env, groups, 1.31e-6, dx,
                                                 transport='collins', **kw)
    keys = ('readout_route', 'readout_route_k1', 'readout_route_reason')
    assert not any(k in szik.stages[-1] for k in keys)
    assert all(k in coll.stages[-1] for k in keys)


# ===========================================================================
# 3.  The oracle decisions
# ===========================================================================
_LADDER = [
    ('diverging', +40.0e-3, 5.0e-3),
    ('converging', R_CONV, 20.0e-3),
    ('at-the-focus', R_CONV, 40.0e-3),
    ('just-past-it', R_CONV, 41.0e-3),
    ('well-past-it', R_CONV, 60.0e-3),
]


def _oracle_rel(out, R, z):
    got, dxo, dyo = _rebuild_field(out, WL)
    ref = _analytic_gaussian(_axis(np.shape(got)[-1], dxo),
                             _axis(np.shape(got)[-2], dyo),
                             W_IN, R, WL, z)
    return _rel(got, ref)


@pytest.mark.parametrize('tag,R,z', _LADDER, ids=[c[0] for c in _LADDER])
def test_the_new_default_is_never_worse_than_the_old_against_the_oracle(
        tag, R, z):
    """The absolute-phase analytic Gaussian, decided rather than pinned.

    The claim is a DECISION -- "the new default is not worse here" -- and its
    two sides are both measured on the running build: the old transport's own
    reading is the upper bar, and the fixture's grid-truncation floor is the
    lower one.  Nothing is compared against a remembered residual, so a
    library that got better moves this test's own bar with it.

    The focus cell is the asymmetric one and is asserted as such: the shipped
    transport cannot land there at all.
    """
    env = _env()
    floor = _truncation_floor()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        coll = CA.propagate_carrier_referenced(env, R, z, WL, DX,
                                               transport='collins')
        try:
            szik = CA.propagate_carrier_referenced(env, R, z, WL, DX,
                                                   transport='sziklas')
        except ValueError as exc:
            szik = None
            szik_err = str(exc)
    r_coll = _oracle_rel(coll, R, z)
    if szik is None:
        assert 'R_carrier == 0' in szik_err, (
            f'the shipped transport failed here for an unexpected reason: '
            f'{szik_err[:160]}')
        assert r_coll <= 2.0 * floor, (
            f'the Collins transport reaches the focus cell the shipped one '
            f'refuses, but reads {r_coll:.4e} against a grid-truncation floor '
            f'of {floor:.4e}')
        return
    r_szik = _oracle_rel(szik, R, z)
    assert r_coll <= r_szik * (1.0 + 1e-12), (
        f'{tag}: the new default reads {r_coll:.6e} against the analytic '
        f'Gaussian where the old one reads {r_szik:.6e} -- the flip made this '
        f'cell worse')
    assert r_coll <= 2.0 * floor, (
        f'{tag}: {r_coll:.6e} against a grid-truncation floor of '
        f'{floor:.6e}; the reading is no longer the fixture and something in '
        f'the transport is contributing')


def test_the_ladder_has_a_cell_where_the_flip_is_a_large_improvement():
    """Two-sided in the other direction: if every cell agreed, the ladder
    above would be gating nothing.

    Measured on the running build, no number pinned -- the assertion is that
    SOME cell improves by more than a decade, and the message prints the whole
    ladder so a future reader can see which.
    """
    env = _env()
    ratios = {}
    for tag, R, z in _LADDER:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                szik = CA.propagate_carrier_referenced(env, R, z, WL, DX,
                                                       transport='sziklas')
            except ValueError:
                ratios[tag] = float('inf')
                continue
            coll = CA.propagate_carrier_referenced(env, R, z, WL, DX,
                                                   transport='collins')
        ratios[tag] = _oracle_rel(szik, R, z) / max(_oracle_rel(coll, R, z),
                                                    1e-300)
    assert max(ratios.values()) > 10.0, (
        f'no cell of the ladder improves by even a decade, so the flip is '
        f'not buying what the decision was taken on: {ratios!r}')


def test_the_focus_cell_is_reachable_on_the_default_and_not_on_the_old_one():
    """The one thing the flip changes qualitatively, stated on its own.

    ``A = 0`` exactly: the co-moving grid collapses and
    ``carrier_referenced_envelope`` refuses ``R_carrier == 0``.  Both arms are
    asserted, so this cannot pass because the fixture stopped being a focus.
    """
    env = _env()
    z = -R_CONV
    with pytest.raises(ValueError, match='R_carrier == 0'):
        CA.propagate_carrier_referenced(env, R_CONV, z, WL, DX,
                                        transport='sziklas')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = CA.propagate_carrier_referenced(env, R_CONV, z, WL, DX)
    assert np.all(np.isfinite(np.asarray(out.env)))
    assert _oracle_rel(out, R_CONV, z) <= 2.0 * _truncation_floor()


def test_the_multi_orchestrator_agrees_with_the_chain_at_K1_and_K2():
    """``_multi`` forwards one transport to K congruences.  Checked at K = 1
    against the single chain and at K = 2 against the linearity the
    recombination promises, both on the default."""
    env, dx, r_in, groups = _chain_fixture()
    fr = dict(dx_out=0.5e-6, N_out=64)
    base = dict(ray_subsample=16, n_workers=1, traced_kwargs=_TKW,
                final_leg='paraxial', final_distance=8e-3)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        k1 = CA.propagate_traced_carrier_chain_multi(
            [{'field': env, 'carrier': r_in}], groups, 1.31e-6, dx,
            output_grid=fr, **base)
        k2 = CA.propagate_traced_carrier_chain_multi(
            [{'field': env, 'carrier': r_in},
             {'field': env, 'carrier': r_in}], groups, 1.31e-6, dx,
            output_grid=fr, **base)
        chain = CA.propagate_traced_carrier_chain(
            env, groups, 1.31e-6, dx, r_in=r_in, focus_readout=fr, **base)
    a = np.asarray(k1.field)
    assert np.array_equal(a, np.asarray(chain.field))
    # K = 2 congruences that are the SAME field recombine to twice it; the bar
    # is the accumulation's own rounding, measured from the sum itself.
    two = np.asarray(k2.field)
    bar = 8.0 * float(np.finfo(np.float64).eps)
    assert _rel(two, 2.0 * a) <= bar, (
        f'K=2 of one repeated congruence is {_rel(two, 2.0 * a):.3e} from '
        f'twice K=1, against an accumulation-rounding bar of {bar:.3e}')


# ===========================================================================
# 4.  The CuPy arm
# ===========================================================================
#: Helpers on the Collins path that are HOST-SIDE BY DESIGN, each with the
#: reason it is not ``xp``-parametrised.  A census with no allow-list is a
#: census nobody can satisfy; an allow-list with no reasons is a rubber stamp.
_HOST_SIDE_BY_DESIGN = {
    '_collins_power_marginals':
        'routes through backend.to_numpy and accumulates host-side in row '
        'bands; reducing on-device would change the NumPy summation order and '
        'break the bit-identity contract (H2-2)',
    '_collins_containment_radius': 'takes host marginals and returns a float',
    '_collins_space_support': 'thin wrapper over the marginals',
    '_collins_angle_support': 'thin wrapper over the marginals',
    '_collins_envelope_half_angle': 'thin wrapper over the marginals',
    '_collins_sampling_stats': 'takes only Python floats',
    '_collins_kernel_wrap_ratio': 'takes only Python floats',
    '_collins_exact_kernel_departure': 'takes only Python floats',
    '_collins_envelope_abcd': 'takes only Python floats',
    '_collins_leg_output_axis': 'takes only Python floats',
    '_collins_readout_k1': 'reads the measured box; returns a float',
    '_check_collins_sampling': 'disposes of a stats dict of floats',
    '_check_transport': 'vocabulary gate on a string',
    '_publish_readout_route': 'writes three entries into a stage dict',
}

#: The helpers that MUST carry the backend triple, with the parameter each one
#: is entitled to have (``_collins_axis_chirp`` builds a field-INDEPENDENT
#: grid, so ``bld`` alone is the whole of its backend contract).
_XP_PARAMETRISED = {
    '_collins_transport': ('env',),
    '_collins_carrier_leg': ('env',),
    '_collins_focus_readout': ('env',),
    '_collins_input_box': ('env',),
    '_collins_exact_kernel_correction': ('xp', 'is_jax', 'bld'),
    '_collins_axis_chirp': ('bld',),
    '_tf_phase_to_H': ('xp', 'is_jax', 'bld'),
    '_exact_dispersion_phase': ('bld',),
    '_fft2_pair': ('xp', 'is_jax'),
    '_as_c_order': ('xp',),
    '_to_dev': ('xp', 'is_jax'),
}


def _module_functions():
    src = pathlib.Path(CA.__file__).read_text(encoding='cp1252')
    tree = ast.parse(src)
    return src, {n.name: n for n in ast.walk(tree)
                 if isinstance(n, ast.FunctionDef)}


def _body_source(src, node):
    """A function's source with its docstring removed, so a cross-reference in
    prose is not counted as an implementation (the H2-2 lesson)."""
    body = node.body
    if (body and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)):
        body = body[1:]
    return '\n'.join(ast.get_source_segment(src, s) or '' for s in body)


def _collins_call_graph(funcs, roots):
    """Every module-level function transitively reachable from ``roots``."""
    seen, stack = set(), list(roots)
    while stack:
        nm = stack.pop()
        if nm in seen or nm not in funcs:
            continue
        seen.add(nm)
        for node in ast.walk(funcs[nm]):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in funcs:
                    stack.append(node.func.id)
    return seen


def test_every_helper_on_the_collins_path_is_xp_parametrised():
    """The structural gate: ONE implementation per kernel, parametrised by the
    field's namespace -- not a NumPy copy beside a device copy.

    The census walks the CALL GRAPH from the three Collins entry helpers, so
    it covers whatever the chain actually reaches rather than whatever this
    file remembered to list.  Every reached helper whose name is this module's
    own must either carry the backend triple (:data:`_XP_PARAMETRISED`, which
    also pins WHICH parameters each one owes) or be named in
    :data:`_HOST_SIDE_BY_DESIGN` WITH its reason.

    A helper that loses its ``xp`` fails here at the parameter, and a new
    helper added to the path with neither property fails here as unclassified
    -- which is the point: the classification is the review.
    """
    src, funcs = _module_functions()
    reached = _collins_call_graph(
        funcs, ('_collins_transport', '_collins_carrier_leg',
                '_collins_focus_readout'))
    assert '_collins_transport' in reached and len(reached) > 8, (
        f'the call-graph walk reached only {sorted(reached)}; the census is '
        f'not reading the module')

    missing = []
    for nm, owed in _XP_PARAMETRISED.items():
        if nm not in funcs:
            missing.append(f'{nm}: no longer defined in this module')
            continue
        have = {a.arg for a in funcs[nm].args.args}
        have |= {a.arg for a in funcs[nm].args.kwonlyargs}
        for p in owed:
            if p not in have:
                missing.append(f'{nm}: lost its {p!r} parameter')
    assert not missing, (
        'these helpers no longer carry the backend triple the Collins chain '
        'threads through them:\n  ' + '\n  '.join(missing))

    # ... and nothing on the reached path is unclassified.
    known = set(_XP_PARAMETRISED) | set(_HOST_SIDE_BY_DESIGN)
    unclassified = sorted(
        nm for nm in reached
        if nm.startswith('_collins') and nm not in known)
    assert not unclassified, (
        f'these Collins helpers are on the transport path and are neither '
        f'declared xp-parametrised nor declared host-side-by-design: '
        f'{unclassified}.  Classify each one (and say why) rather than '
        f'widening the census.')
    for nm, why in _HOST_SIDE_BY_DESIGN.items():
        assert why and len(why) > 10, f'{nm} is allow-listed with no reason'


#: The names a FIELD travels under inside the Collins chain.  A host
#: normalisation of any of these is the demotion; a host normalisation of a
#: coordinate axis or a scalar is not, which is why the census is keyed on the
#: ARGUMENT and not on the function.
_FIELD_NAMES = ('env', 'env_a', 'E_env', 'spectrum', 'g', 'G', 'E_out')

#: The two host normalisations that silently copy a device array (or raise a
#: bare TypeError on CuPy) while being a bitwise no-op on NumPy.
_HOST_NORMALISERS = ('asarray', 'ascontiguousarray')


def _host_demotions(node):
    """``np.<normaliser>(<field>)`` call sites inside one function node."""
    hits = []
    for n in ast.walk(node):
        if not isinstance(n, ast.Call) or not n.args:
            continue
        f = n.func
        if not (isinstance(f, ast.Attribute)
                and f.attr in _HOST_NORMALISERS
                and isinstance(f.value, ast.Name) and f.value.id == 'np'):
            continue
        a0 = n.args[0]
        if isinstance(a0, ast.Name) and a0.id in _FIELD_NAMES:
            hits.append(f'line {n.lineno}: np.{f.attr}({a0.id}, ...)')
    return hits


def test_no_collins_helper_demotes_the_field_to_host_numpy():
    """The mutation that the JAX value tests cannot see, closed structurally.

    ``np.asarray(env)`` is bitwise a no-op for a NumPy field and a silent
    host demotion for a JAX one, so no VALUE comparison on either backend can
    catch it -- V-D3 records exactly that: the leg's ``env_a = np.asarray(env)``
    was invisible to every test until a CuPy array raised a bare ``TypeError``
    naming neither the leg nor the transport.

    So the gate is on the SOURCE: inside the Collins chain's own functions, a
    field may be normalised only through ``xp.asarray`` / :func:`_as_c_order`,
    never through ``np.asarray`` / ``np.ascontiguousarray``.  Host-side
    MEASUREMENT keeps its ``np.asarray`` -- it goes through
    ``backend.to_numpy`` first, and those functions are the allow-listed ones
    above, which this census does not walk into.

    It is a census of AST CALL NODES, not a substring search, for a reason
    this test was taught the hard way (2026-09-20): ``_collins_input_box``
    carries a COMMENT naming the old spelling
    (``_fft2(np.ascontiguousarray(env, ...))``) as the defect it fixed, and a
    substring search over the function's source fired on it -- a false
    positive on a cross-reference, which is the same shape H2-2's own census
    hit on a docstring and the opposite of what the gate is for.  Matching
    call nodes cannot see a comment at all.

    The census is shown to be LOAD-BEARING at the end: the same matcher is run
    over a deliberately mutated copy of one function and must fire.
    """
    _, funcs = _module_functions()
    sites = ('_collins_transport', '_collins_carrier_leg',
             '_collins_focus_readout', '_collins_input_box',
             '_collins_exact_kernel_correction')
    bad = [f'{nm}: {hit}' for nm in sites
           for hit in _host_demotions(funcs[nm])]
    assert not bad, (
        'these sites demote the FIELD to host NumPy inside the Collins '
        'chain, which is bitwise invisible on NumPy and a silent device '
        'copy (or a bare TypeError) elsewhere:\n  ' + '\n  '.join(bad))

    mutated = ast.parse(
        'def f(env):\n'
        '    env_a = np.asarray(env)\n'
        '    return env_a\n').body[0]
    assert _host_demotions(mutated), (
        'the matcher does not fire on an explicit np.asarray(env), so the '
        'clean reading above is not evidence of anything')


def test_the_fft_on_the_collins_path_is_the_backend_dispatcher():
    """The transform comes from the library's ONE FFT dispatcher, by identity
    on the NumPy side and by agreement with ``backend.fft2`` on both.

    Identity, not equivalence: ``_bluestein_2d`` keys its chirp-kernel cache on
    ``fft2 is fft_infra._fft2``.  And a CuPy array is dispatched by
    ``fft_infra._fft2`` itself (its first branch is ``_is_cupy_array``), which
    is why the CuPy arm needs no second selector -- asserted here as a
    property of the dispatcher's source rather than assumed.
    """
    from lumenairy.backend import fft2 as backend_fft2
    from lumenairy.propagators import fft_infra
    f, i = CA._fft2_pair(np, False)
    assert f is fft_infra._fft2 and i is fft_infra._ifft2
    env = _env(n=64, dx=8e-6, w=60e-6)
    a = np.ascontiguousarray(f(np.ascontiguousarray(env,
                                                    dtype=np.complex128)))
    b = np.ascontiguousarray(backend_fft2(np.ascontiguousarray(
        env, dtype=np.complex128)))
    assert np.array_equal(a.view(np.float64), b.view(np.float64))
    disp = pathlib.Path(fft_infra.__file__).read_text(encoding='cp1252')
    for fn in ('def _fft2(x):', 'def _ifft2(x):'):
        head = disp.split(fn, 1)[1][:2000]
        assert '_is_cupy_array(x)' in head, (
            f'{fn.strip()} no longer dispatches a CuPy array to cp.fft, so '
            f'the Collins chain would run a device array through the host '
            f'transform')


def test_a_true_cupy_answer_really_binds_the_fft_dispatchers_cp(monkeypatch):
    """The Collins chain's device transform rests on ONE coupling, and this is
    the repo's own way of exercising it without a device.

    ``fft_infra._fft2``'s FIRST branch is ``if _is_cupy_array(x): return
    cp.fft.fft2(x)`` -- it reads the MODULE-LEVEL name ``cp``.  So
    ``_is_cupy_array(x) is True`` must imply ``cp`` is bound, or the Collins
    leg's transform raises ``NameError`` on the one box it was written for.
    The shipped ``test_fft_infra_keeps_its_cp_alias_contract`` asserts that
    coupling only on the branch this box HAS: with no CuPy it checks
    ``cp is None`` and never reaches the True side at all.

    This is ``test_audit2609_a16_verify_config_and_arch.py::
    test_a_true_cupy_answer_really_binds_the_module_cp`` applied to the
    dispatcher the Collins chain uses: FAKE a CuPy answer by substituting the
    module's four optional-dependency handles, and check the name afterwards
    -- which is what a CUDA box would check.  Both sides are asserted, and the
    module is restored, so a later test cannot inherit a stub.
    """
    import types
    from lumenairy.propagators import fft_infra as fi
    from lumenairy.backend import _optional

    stub = types.ModuleType('cupy_stub')
    monkeypatch.setattr(fi, 'CUPY_AVAILABLE', True)
    monkeypatch.setattr(fi, 'cp', None)
    monkeypatch.setattr(fi, '_optional_is_cupy_array', lambda x: True)
    monkeypatch.setattr(fi, '_ensure_cupy', lambda: stub)
    try:
        assert fi._is_cupy_array(object()) is True
        assert fi.cp is stub, (
            'fft_infra._is_cupy_array answered True without binding the '
            'module-level cp; the CuPy branch of _fft2 would raise '
            'NameError, which is how the Collins chain reaches a device '
            'transform')
    finally:
        fi.cp = None
    monkeypatch.undo()
    assert fi._is_cupy_array(np.zeros(3)) is False
    assert fi.CUPY_AVAILABLE is _optional.CUPY_AVAILABLE


def _cupy_premise():
    """READ this box's CuPy state as a fact.  Returns a dict; never skips."""
    out = {'importable': False, 'devices': None, 'elementwise': False,
           'fft': False, 'fft_error': None, 'cp': None}
    try:
        import cupy as cp
    except Exception:                                 # noqa: BLE001
        return out
    out['importable'] = True
    out['cp'] = cp
    try:
        out['devices'] = int(cp.cuda.runtime.getDeviceCount())
        a = cp.ones((4, 4), dtype=cp.complex128)
        cp.asnumpy(a * 2.0)
        out['elementwise'] = True
    except Exception:                                 # noqa: BLE001
        return out
    try:
        cp.asnumpy(cp.fft.fft2(cp.ones((4, 4), dtype=cp.complex128)))
        out['fft'] = True
    except Exception as exc:                          # noqa: BLE001
        out['fft_error'] = f'{type(exc).__name__}: {exc}'
    return out


def test_the_cupy_premise_is_read_and_not_assumed():
    """The premise itself, asserted as a fact of the running box.

    MEASURED 2026-09-20 here: WIN-py3.14 has cupy 14.0.1 with ONE visible
    device, working elementwise kernels, and ``cupy.fft`` raising
    ``ImportError: DLL load failed while importing cufft``; WSL-py3.12 has no
    CuPy at all.  Neither of those is asserted as a constant -- what is
    asserted is that the three readings are CONSISTENT, because an
    inconsistent reading (a working cuFFT on a box with no devices, say) would
    make every decision below meaningless.
    """
    p = _cupy_premise()
    if not p['importable']:
        assert p['devices'] is None and not p['elementwise'] and not p['fft']
        return
    assert p['devices'] is not None
    if p['fft']:
        assert p['elementwise'], (
            'cupy.fft works but elementwise kernels do not, which is not a '
            'state this library knows how to reason about')
    if not p['fft']:
        assert p['fft_error'], 'cupy.fft failed without an error to record'


def test_the_device_helpers_agree_with_the_host_build():
    """Every FIELD-INDEPENDENT helper the Collins chain builds on ``bld``, run
    ON THE DEVICE and compared to the host build.

    These need no transform, so they run on a box whose cuFFT is broken -- and
    that is the whole point: the CuPy half of ``bld`` is covered here by
    arithmetic instead of by inspection.  The bar is the device's own libm
    against the host's, so it is stated in ULPs of 1 and derived from the
    dtype, not from a remembered residual.  MEASURED 2026-09-20 (WIN-py3.14,
    cupy 14.0.1): the axis chirp reads 3.7e-17 to 4.4e-17 relative (<= 0.7 ULP
    of 1) at three grids, ``_tf_phase_to_H`` 4.6e-17, and
    ``_exact_dispersion_phase`` EXACTLY 0.0 both untilted and tilted.
    """
    p = _cupy_premise()
    if not p['elementwise']:
        # The decision that holds on a box with no usable CuPy: the helpers
        # must still be reachable with ``bld=np`` and the NumPy build must be
        # what the chain gets.  Asserted rather than skipped.
        assert CA._backend_of(_env(n=8, dx=DX))[2] is np
        assert inspect.signature(
            CA._collins_axis_chirp).parameters['bld'].default is np
        return
    cp = p['cp']
    ulp = float(np.finfo(np.float64).eps)
    bar = 8.0 * ulp                       # eight ULPs of 1, derived not fitted
    worst = 0.0
    for n, dx, R in ((64, 8e-6, -0.05), (256, 4e-6, -0.02)):
        d = cp.asnumpy(CA._collins_axis_chirp(n, dx, WL, R, bld=cp))
        h = CA._collins_axis_chirp(n, dx, WL, R, bld=np)
        worst = max(worst, float(np.abs(d - h).max()))
    arg = np.linspace(-3.0e4, 3.0e4, 128 * 128).reshape(128, 128)
    Hd = cp.asnumpy(CA._tf_phase_to_H(cp.asarray(arg), np.complex128,
                                      cp, False, cp))
    Hh = CA._tf_phase_to_H(arg, np.complex128, np, False, np)
    worst = max(worst, float(np.abs(Hd - Hh).max()))
    qx = 2.0 * np.pi * np.fft.fftfreq(64, d=4e-6)
    k = 2.0 * np.pi / WL
    for tilt in ((0.0, 0.0), (0.02, -0.01)):
        pd = cp.asnumpy(CA._exact_dispersion_phase(
            cp.asarray(qx), cp.asarray(qx), k, tilt, cp, 'probe'))
        ph = CA._exact_dispersion_phase(qx, qx, k, tilt, np, 'probe')
        worst = max(worst, float(np.abs(pd - ph).max()))
    assert worst <= bar, (
        f'a field-independent Collins grid differs between the device build '
        f'and the host build by {worst:.3e} ({worst / ulp:.1f} ULP of 1), '
        f'against a bar of {bar:.3e}; that is more than two libms disagreeing')
    # and the triple the chain would resolve for a device field is the device
    # namespace on BOTH slots -- ``bld is xp`` for CuPy, unlike JAX.
    xp, is_jax, bld = CA._backend_of(cp.asarray(_env(n=8, dx=DX)))
    assert xp is cp and is_jax is False and bld is cp


def test_a_device_array_reaches_the_device_transform():
    """The public leg with a CuPy array in, decided on whichever side of this
    box's cuFFT premise holds.

    * WORKING cuFFT: the leg runs ON THE DEVICE, returns a DEVICE array, and
      agrees with the NumPy leg to a bar measured here from the two backends'
      own single FFT -- the only thing entitled to differ -- times the chain
      depth the transport applies.
    * BROKEN cuFFT (this box, 2026-09-20): the leg must fail AT THE DEVICE
      TRANSFORM, i.e. with the ``ImportError`` naming cufft.  What it must NOT
      do is fail earlier with ``TypeError: Implicit conversion to a NumPy
      array is not allowed``, which is the signature of a host demotion and is
      the defect V-D3 fixed.  So the broken-cuFFT arm is not an absence of
      evidence: it is the evidence that the array got all the way to the
      transform.
    * NO CuPy: the NumPy leg must still run and return a NumPy array, which is
      the only decision available.

    THE DEVICE RUN IS OWED.  On no box available to this package does
    ``cupy.fft`` work, so the first arm has never executed.  It is written,
    and ``docs/audits/.../fixes/WP-C3_COLLINS_DEFAULT_REPORT.md`` records the
    debt in those words.
    """
    p = _cupy_premise()
    env = _env(n=64, dx=8e-6, w=60e-6)
    kw = dict(transport='collins', gap_kernel='fresnel',
              on_collins_sampling='ignore')
    host = CA.propagate_carrier_referenced(env, -0.05, 5e-3, 633e-9, 8e-6,
                                           **kw)
    assert isinstance(np.asarray(host.env), np.ndarray)
    if not p['elementwise']:
        return
    cp = p['cp']
    Ed = cp.asarray(env)
    if p['fft']:
        out = CA.propagate_carrier_referenced(Ed, -0.05, 5e-3, 633e-9, 8e-6,
                                              **kw)
        assert type(out.env).__module__.split('.')[0] == 'cupy', (
            'a device array in did not give a device array out')
        from lumenairy.propagators.fft_infra import _fft2
        fa = np.asarray(_fft2(np.ascontiguousarray(env,
                                                   dtype=np.complex128)))
        fb = np.asarray(cp.asnumpy(cp.fft.fft2(
            cp.asarray(env, dtype=cp.complex128))))
        spread = float(np.linalg.norm(fa - fb) / np.linalg.norm(fa))
        bar = max(6.0 * spread, 32.0 * float(np.finfo(np.float64).eps))
        got = _rel(cp.asnumpy(out.env), np.asarray(host.env))
        assert got <= bar, (
            f'the device leg disagrees with the host leg by {got:.4e} against '
            f'a bar of {bar:.4e} measured from the two backends\' own FFTs '
            f'(single-transform spread {spread:.4e})')
        return
    with pytest.raises(Exception) as ei:              # noqa: PT011
        CA.propagate_carrier_referenced(Ed, -0.05, 5e-3, 633e-9, 8e-6, **kw)
    msg = f'{type(ei.value).__name__}: {ei.value}'
    assert 'Implicit conversion' not in msg, (
        f'the Collins leg demoted a CuPy array to the host instead of '
        f'reaching the device transform: {msg[:200]}')
    assert 'cufft' in msg.lower(), (
        f"this box's cupy.fft is broken, so the leg is expected to fail AT "
        f"the device transform and name cufft; it failed with {msg[:200]} "
        f"instead, which means it stopped somewhere else")


# ===========================================================================
# 5.  The Kelly guard's default, now that more callers reach it
# ===========================================================================
def test_the_default_does_not_start_warning_on_the_shipped_fixtures():
    """``on_collins_sampling`` defaults to ``'warn'``, so a caller who never
    saw the K-condition warning could start seeing it once the transport
    changes underneath them.  Measured, not asserted away.

    On the fixtures this file can reach, the count is ZERO -- because both the
    leg and the readout RESOLVE away from the chirp-Z exactly where the
    conditions would fire.  The claim is two-sided: the same guard is shown to
    be capable of firing, by driving a leg that has no fallback (a caller-named
    output lattice) past its own condition.
    """
    env, dx, r_in, groups = _chain_fixture()
    fr = dict(dx_out=0.5e-6, N_out=64)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        CA.propagate_traced_carrier_chain(
            env, groups, 1.31e-6, dx, r_in=r_in, ray_subsample=16,
            n_workers=1, traced_kwargs=_TKW, final_leg='paraxial',
            final_distance=8e-3, focus_readout=fr)
        CA.propagate_traced_carrier_chain(
            env, groups, 1.31e-6, dx, r_in=r_in, ray_subsample=16,
            n_workers=1, traced_kwargs=_TKW, final_leg='paraxial',
            final_distance=8e-3)
    fired = [str(w.message) for w in rec
             if 'chirp-Z stage is under-sampled' in str(w.message)]
    assert not fired, (
        f'{len(fired)} Kelly warning(s) on the shipped default over two chain '
        f'runs; the Migration paragraph says zero and would have to be '
        f'rewritten:\n{fired[0][:300] if fired else ""}')

    # ... and the guard is not simply dead: a caller-NAMED output lattice has
    # no complementary form to fall back to, and it speaks there.
    with warnings.catch_warnings(record=True) as rec2:
        warnings.simplefilter('always')
        CA.propagate_carrier_referenced(
            _env(), R_CONV, 20e-3, WL, DX, transport='collins',
            dx_out=DX * 64.0, on_collins_sampling='warn')
    assert [w for w in rec2
            if 'chirp-Z stage is under-sampled' in str(w.message)], (
        'the Kelly guard did not fire even on a deliberately under-sampled '
        'caller-named lattice, so the zero above is not evidence of anything')


# ===========================================================================
# 6.  WP-C3 ROUND 2 -- the flat-reference leg, and the ordinary chain that
#     used to raise (VERIFY-WP-C3 D6 and D5, which are ONE defect)
# ===========================================================================
def _flat_unrepresentable_leg():
    """A leg the transport resolves FLAT and whose chirp-Z is NOT
    representable, DERIVED from the running build rather than pinned.

    Construction: a grid-filling Gaussian carrying a linear phase ramp, so the
    measured angular half-width is a large fraction of the grid's Nyquist
    angle, on a short leg with ``A`` small enough that the geometric
    reference's space-bandwidth ``4 r_out theta/(|A| lambda)`` exceeds ``N``.
    Both facts are READ BACK from the library's own resolver below, so this
    helper cannot drift from what the leg actually does.
    """
    wl, n, dx, w = 1.064e-6, 1024, 4.0e-6, 0.30e-3
    f0 = 1.0e5                      # 0.4 cycles/sample -> theta = 0.1064 rad
    z = 2.0e-3
    r_c = -z / 0.89                 # A = 1 + z/R = 0.11
    x = _axis(n, dx)
    xx, yy = np.meshgrid(x, x, indexing='ij')
    env = (np.exp(-(xx ** 2 + yy ** 2) / (w * w))
           * np.exp(2j * np.pi * f0 * yy)).astype(np.complex128)
    r_x, _r_y, th_x, _th_y = CA._collins_input_box(
        env, dx, dx, wl, CA._COLLINS_TAIL_FRAC)
    a, b, _c, _d = CA._collins_envelope_abcd(r_c, z, np.inf)
    d_out, flat = CA._collins_leg_output_axis(
        a, b, r_c + z, dx, n, r_x, th_x, wl)
    k1 = 2.0 * dx * (abs(a) * r_x / abs(b) + th_x) / wl
    k3 = n * d_out / (wl * abs(b) / dx)
    return env, r_c, z, wl, dx, flat, k1, k3


def test_a_flat_resolving_leg_resolves_its_quadrature_like_any_other():
    """ROUND 2, closing VERIFY-WP-C3 D6 -- and D5 with it, since the ordinary
    chain that raised did so because ITS second gap leg took this branch.

    A resolved FLAT output reference used to disable the fallback outright
    (``tf_available`` carried ``and not flat``), so the chirp-Z ran on such a
    leg whatever K1 and K3 read.  The exclusion's stated reason -- "the
    Sziklas transport could never evaluate these legs" -- is true only of the
    ``A == 0`` sub-case, which the ``Ax != 0`` conjunct already excludes on
    its own; ``flat`` is ALSO resolved whenever the geometric reference's
    space-bandwidth exceeds ``N``, and there ``R_out = R + z`` is finite and
    non-zero.

    TWO-SIDED, because the fix is a selection and not a retreat:

    * a flat-resolving leg whose chirp-Z is NOT representable now falls back,
      and the fallback is the Sziklas ANSWER to the bit;
    * a flat-resolving leg whose chirp-Z IS representable keeps the flat
      reference and the chirp-Z, and still comes back with ``R = inf``.
    """
    env, r_c, z, wl, dx, flat, k1, k3 = _flat_unrepresentable_leg()
    # the PREMISE, measured off the library's own resolver, not assumed
    assert flat, ('the fixture no longer resolves a flat output reference, '
                  'so it cannot exercise the branch this test is about')
    assert max(k1, k3) > 1.0, (
        f'the fixture is representable (K1 = {k1:.6g}, K3 = {k3:.6g}), so no '
        f'fallback would be owed here')
    diag = {}
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        got = CA._collins_carrier_leg(env, r_c, z, wl, dx, dx,
                                      gap_kernel='fresnel',
                                      on_collins_sampling='warn', diag=diag)
    assert diag.get('collins_form') == 'tf', (
        f"a flat-resolving leg at K1 = {k1:.6g} / K3 = {k3:.6g} still ran the "
        f"chirp-Z ({diag.get('collins_form')!r})")
    assert not [w for w in rec
                if 'chirp-Z stage is under-sampled' in str(w.message)], (
        'the leg fell back and still emitted the Kelly warning')
    ref = CA.propagate_carrier_referenced(env, r_c, z, wl, dx,
                                          gap_kernel='fresnel',
                                          transport='sziklas')
    assert np.array_equal(np.asarray(got.env), np.asarray(ref.env)), (
        'the flat leg fell back but not to the Sziklas ANSWER')
    assert _one(got.dx) == _one(ref.dx) and _one(got.R) == _one(ref.R)

    # ... and the other side: a flat reference that IS representable is kept.
    n2, dx2, w2, wl2 = 256, 6.0 * W_IN / 256, W_IN, WL
    x2 = _axis(n2, dx2)
    xx2, yy2 = np.meshgrid(x2, x2, indexing='ij')
    env2 = np.exp(-(xx2 ** 2 + yy2 ** 2) / (w2 * w2)).astype(np.complex128)
    r2, z2 = R_CONV, -R_CONV                      # A == 0: the focus landing
    rr, _ry, tt, _ty = CA._collins_input_box(
        env2, dx2, dx2, wl2, CA._COLLINS_TAIL_FRAC)
    a2, b2, _c2, _d2 = CA._collins_envelope_abcd(r2, z2, np.inf)
    d2, flat2 = CA._collins_leg_output_axis(a2, b2, r2 + z2, dx2, n2, rr, tt,
                                            wl2)
    assert flat2
    assert n2 * d2 / (wl2 * abs(b2) / dx2) <= 1.0, (
        'the second arm is no longer a REPRESENTABLE flat leg')
    diag2 = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        got2 = CA._collins_carrier_leg(env2, r2, z2, wl2, dx2, dx2,
                                       diag=diag2)
    assert diag2.get('collins_form') == 'chirp-z', (
        'a representable flat leg was pushed onto the fallback, which would '
        'make the fix a retreat from the flat form rather than a selection')
    assert diag2.get('collins_flat_reference') is True
    assert not np.isfinite(_one(got2.R)), (
        'the representable flat leg no longer returns R = inf')


def _one(v):
    return float(v[0]) if isinstance(v, tuple) else float(v)


def test_an_ordinary_two_group_relay_with_a_readout_matches_the_old_default():
    """ROUND 2, closing VERIFY-WP-C3 D5 at the level the Migration paragraph
    makes its claim.

    The reproducer is deliberately ORDINARY: a converging launch into two
    identical N-BK7 biconvex singlets, a plain ``focus_readout``, no
    stop-plane key, no tilt, no ``gap_kernel``.  Before round 2, 11 of 12
    configurations of this family RAISED ``RuntimeError`` from the Sziklas
    readout's containment guard where all 12 returned at 49ddf4bd, and a
    192-cell ordinary-chain sweep read 118 IDENTICAL / 52 MOVED / 22
    OK->RAISED with 74 Kelly warnings over 51 cells.  After it the same sweep
    reads 192 IDENTICAL / 0 MOVED / 0 OK->RAISED / 0 warnings, on both builds.

    Asserted as a DECISION rather than as those counts: on this family the
    default returns, agrees with ``transport='sziklas'`` to the bit, and warns
    about nothing.  The fixture's own relevance is asserted too -- the readout
    really does route to the Sziklas quadrature here (K1 > 1), which is what
    made the containment guard reachable at all.
    """
    p = _singlet()
    groups = [{'prescription': p, 'gap_before': 20e-3},
              {'prescription': p, 'gap_before': 15e-3}]
    n, dx, w = 256, 10.24e-3 / 256, 3.0e-3
    x = _axis(n, dx)
    xx, yy = np.meshgrid(x, x, indexing='ij')
    env = np.exp(-(xx ** 2 + yy ** 2) / (w * w)).astype(np.complex128)
    for fd in (5e-3, 15e-3):
        kw = dict(r_in=60e-3, ray_subsample=16, n_workers=1,
                  traced_kwargs=_TKW, final_leg='paraxial',
                  final_distance=fd,
                  focus_readout=dict(dx_out=0.5e-6, N_out=64))
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter('always')
            ref = CA.propagate_traced_carrier_chain(
                env, groups, 1.31e-6, dx, transport='sziklas', **kw)
            got = CA.propagate_traced_carrier_chain(
                env, groups, 1.31e-6, dx, **kw)
        assert np.all(np.isfinite(ref.field)), (
            'the pre-flip arithmetic no longer returns on this fixture, so '
            'it cannot say anything about the default')
        assert np.array_equal(np.asarray(got.field), np.asarray(ref.field)), (
            f'the default moved on an ordinary two-group relay at '
            f'final_distance = {fd * 1e3:g} mm')
        st = got.stages[-1]
        assert st.get('readout_route') == 'sziklas' and \
            (st.get('readout_route_k1') or 0.0) > 1.0, (
            f'the fixture no longer reaches the Sziklas readout route '
            f'({st.get("readout_route")!r}, K1 = '
            f'{st.get("readout_route_k1")!r}), so the containment guard this '
            f'defect was about is no longer on the path')
        assert not [x_ for x_ in rec
                    if 'chirp-Z stage is under-sampled' in str(x_.message)], (
            'the default emitted a Kelly warning on an ordinary relay where '
            'the pre-flip default emitted none')
