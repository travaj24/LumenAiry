# validation/pipeline/sources.py -- the two PLUG-IN POINTS of the pipeline:
# what decomposes a system into beams, and what carries one beam to the
# summable plane.
#
# Both are registries keyed by STRING, because a spec file has to be a
# complete description of a run.  A config that named a Python import path
# would be a config that executes arbitrary code, and a config that hard-coded
# design 121 would be a pipeline with one customer.
#
# WHAT SHIPS HERE
# ---------------
#   decomposers
#     'design121_doe'  -- the design-121 Dammann DOE order table (LOCAL-ONLY:
#                         needs the .zmx and the design-study runner).  The
#                         first implementation, and the one the acceptance is
#                         scored on.
#     'sumap_cache'    -- beams read from the ARCHIVED arm-A metadata of
#                         PROBE_SUM_AT_APERTURE.  No .zmx, no chain: this is
#                         the route that makes a reproduction claim STRONG,
#                         because nothing upstream of the seam can differ.
#     'explicit'       -- the beam list written out in the spec.  Generic.
#
#   chain runners
#     'traced'           -- propagate_traced_carrier_chain to the last group's
#                           back aperture ON THE EXACT LEG'S FINE RETRACE GRID
#                           (the only summable plane), captured with a
#                           read-only spy on _fine_trace_group_exit.
#     'cached_aperture'  -- the byte-identical archived aperture field.
#
# cp1252-safe ASCII only.
from __future__ import annotations

import json
import os
import time
import warnings
from dataclasses import dataclass, field as _dcfield
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

import numpy as np

from .spec import PipelineSpec, SpecError

# ---------------------------------------------------------------------------
# values
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Beam:
    """One member of a decomposition: a weight, a frame, and whatever the
    chain runner needs to carry it.

    ``key`` is the artifact name -- it appears in file paths, in the ledger
    and in every report -- so it is required to be short and filesystem-safe.
    ``payload`` is the runner's own, and is JSON-only so it can be
    checkpointed."""

    key: str
    index: int
    weight: complex
    frame_centre: Tuple[float, float]
    label: str = ''
    payload: Dict[str, Any] = _dcfield(default_factory=dict)

    def __post_init__(self):
        k = str(self.key)
        if not k or any(c not in
                        'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
                        '0123456789_-.' for c in k):
            raise SpecError(
                f"Beam.key {self.key!r} must be a non-empty string of "
                f"[A-Za-z0-9_-.] -- it names files.")
        object.__setattr__(self, 'key', k)
        object.__setattr__(self, 'weight', complex(self.weight))
        object.__setattr__(self, 'frame_centre',
                           (float(self.frame_centre[0]),
                            float(self.frame_centre[1])))
        object.__setattr__(self, 'label', str(self.label or k))

    def to_dict(self):
        return {'key': self.key, 'index': int(self.index),
                'weight': [self.weight.real, self.weight.imag],
                'frame_centre': [self.frame_centre[0], self.frame_centre[1]],
                'label': self.label, 'payload': self.payload}

    @classmethod
    def from_dict(cls, d):
        return cls(key=d['key'], index=int(d['index']),
                   weight=complex(d['weight'][0], d['weight'][1]),
                   frame_centre=(d['frame_centre'][0], d['frame_centre'][1]),
                   label=d.get('label', ''), payload=dict(d.get('payload',
                                                                {})))


class DecomposeResult(NamedTuple):
    """``beams`` is checkpointed; ``context`` is not.

    The split is the point: ``context`` holds the launch field and the group
    list -- arrays and prescriptions that are expensive but REPRODUCIBLE from
    their own caches -- while ``beams`` is the small, exact, human-readable
    thing every later stage indexes by.  On resume the context is rebuilt and
    the beam list is READ, so a beam table can never drift from the artifacts
    named after it."""

    beams: List[Beam]
    context: Dict[str, Any]


class ChainOutput(NamedTuple):
    field: Any                      #: CarrierField at the summable plane
    meta: Dict[str, Any]            #: JSON-safe: geometry, energy, timings
    reference_tile: Optional[np.ndarray]   #: the SHIPPED path's own frame


# ---------------------------------------------------------------------------
# registries
# ---------------------------------------------------------------------------
_DECOMPOSERS: Dict[str, Callable[..., DecomposeResult]] = {}
_CHAIN_RUNNERS: Dict[str, Callable[..., ChainOutput]] = {}


def register_decomposer(name, fn):
    """``fn(params: dict, spec: PipelineSpec) -> DecomposeResult``."""
    _DECOMPOSERS[str(name)] = fn


def register_chain_runner(name, fn):
    """``fn(beam, context, spec) -> ChainOutput``."""
    _CHAIN_RUNNERS[str(name)] = fn


def get_decomposer(name):
    try:
        return _DECOMPOSERS[str(name)]
    except KeyError:
        raise SpecError(
            f"decompose.kind={name!r} is not registered.  Known: "
            f"{', '.join(sorted(_DECOMPOSERS)) or '(none)'}.")


def get_chain_runner(name):
    try:
        return _CHAIN_RUNNERS[str(name)]
    except KeyError:
        raise SpecError(
            f"chain.kind={name!r} is not registered.  Known: "
            f"{', '.join(sorted(_CHAIN_RUNNERS)) or '(none)'}.")


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------
def order_tag(m, n):
    """The probe's own order tag, kept BYTE-COMPATIBLE with
    ``sumap_probe_121._tag`` so a pipeline artifact and an archived probe
    artifact for the same order have the same name."""
    return f"{int(m):+d}_{int(n):+d}".replace('+', 'p').replace('-', 'm')


def _resolve_ram_budget(v):
    s = str(v).strip().lower()
    if s == 'inf':
        return float('inf')
    if s in ('auto', 'box', 'none', ''):
        return None
    return float(s) * 1e9


def _readout_grid_from_meta(meta, wavelength, window_factor, n_fine_cap):
    """Reproduce ``carrier_referenced_exact_focus_readout``'s OWN sizing
    arithmetic for a crop leg, from an aperture field's recorded geometry.

    Verbatim in behaviour from ``sumap_probe_121.arm_a_readout_grid`` (and
    from the readout itself), which is what makes a 'crop' leg LIKE FOR LIKE
    with the shipped per-order path rather than merely similar: the probe
    checked the numbers this returns against the published
    ``ro_win_mm`` / ``ro_n_fine`` / ``ro_dx_fine_um`` columns of the 32-order
    adjudication and they agree to every digit."""
    dx = float(meta['dx_ap'])
    n = int(meta['n_ap'])
    cen = (meta['chief_exit'][0] - meta['grid_origin'][0],
           meta['chief_exit'][1] - meta['grid_origin'][1])
    w_amp = float(meta['w_amp'])
    dec = bool(cen[0] or cen[1] or meta['tilt_exit'][0] or meta['tilt_exit'][1])
    avail = n * dx - 2.0 * max(abs(cen[0]), abs(cen[1])) if dec else n * dx
    win = min(float(window_factor) * w_amp, avail)
    n_crop = int(min(max(int(2 * round((win / dx) / 2)), 2), n))
    win = n_crop * dx
    na = min(max(w_amp / abs(float(meta['R_out'])), 0.02), 0.95)
    dxf = float(wavelength) / (3.0 * na)
    n_fine = int(2 ** int(np.ceil(np.log2(max(win / dxf, n_crop)))))
    n_fine = min(n_fine, int(n_fine_cap))
    return {'w_amp': w_amp, 'win': win, 'n_crop': n_crop, 'n_fine': n_fine,
            'dx_fine': win / n_fine}


def _aperture_meta(E, dx_ap, grid_origin, R_out, chief_exit, tilt_exit,
                   power_exit):
    """The geometry + energy record every chain runner must produce.

    Measured, not asserted: ``w_amp`` comes from the library's own decentred
    ``_envelope_amp_radius`` on the field itself (|full field| == |envelope|,
    so it does not matter which is passed), and ``power_field`` is integrated
    on the returned grid rather than copied from the chain's own stage."""
    from lumenairy.propagators.carrier import _envelope_amp_radius
    cen = (float(chief_exit[0]) - float(grid_origin[0]),
           float(chief_exit[1]) - float(grid_origin[1]))
    w_amp = float(_envelope_amp_radius(np.asarray(E), float(dx_ap),
                                       float(dx_ap), centre=cen))
    p_field = float((np.abs(E) ** 2).sum()) * float(dx_ap) ** 2
    return {'dx_ap': float(dx_ap), 'n_ap': int(np.shape(E)[-1]),
            'grid_origin': [float(grid_origin[0]), float(grid_origin[1])],
            'R_out': float(R_out),
            'chief_exit': [float(chief_exit[0]), float(chief_exit[1])],
            'tilt_exit': [float(tilt_exit[0]), float(tilt_exit[1])],
            'power_exit': (None if power_exit is None else float(power_exit)),
            'power_field': p_field, 'w_amp': w_amp}


def wrap_aperture_field(E, meta, wavelength, provenance):
    """Divide the beam's OWN congruence out and wrap it as a CarrierField.

    This is step (1) of the probe's arm B, and it is the only place in the
    pipeline where a raw array becomes a first-class field.  The carrier's
    ``piston`` is 0.0 -- NOT because the piston does not matter, but because
    the chain does not return one on a library without
    ``FIX_TILT_QUADRATIC_OPL_2026_08_11``.  Recording it explicitly as zero is
    the honest encoding of that state; a primitive that omitted the field
    would make the pre- and post-fix cases indistinguishable."""
    from lumenairy.propagators.carrier_field import (
        CarrierField, CarrierSpec, FieldGrid)
    n = int(meta['n_ap'])
    grid = FieldGrid((n, n), float(meta['dx_ap']),
                     origin=(meta['grid_origin'][0], meta['grid_origin'][1]))
    car = CarrierSpec(R=float(meta['R_out']),
                      centre=(meta['chief_exit'][0], meta['chief_exit'][1]),
                      tilt=(meta['tilt_exit'][0], meta['tilt_exit'][1]),
                      piston=0.0)
    return CarrierField.from_full_field(E, grid, car, float(wavelength),
                                        provenance=provenance)


# ===========================================================================
# DECOMPOSER 1 -- design-121 Dammann DOE order table
# ===========================================================================
def decompose_design121_doe(params, spec: PipelineSpec) -> DecomposeResult:
    """The DOE's own order decomposition, as ``fan_multi_121.py`` performs it.

    The Dammann cell's DFT gives the complex amplitude ``a(m,n)`` of order
    ``(m,n)``, whose tilt is ``(L, M) = (m, n) * lambda / period``.  Many
    periods illuminate the beam, so that decomposition is the DOE's EXACT
    action rather than a model of it -- which is what makes the beams a
    linear-superposition basis and the whole architecture admissible.

    LOCAL-ONLY: needs the 121 ``.zmx`` and the design-study runner (whose
    ``_NEW_GLASSES`` Sellmeier table sets the dispersion the trace uses).
    That is the single strongest reason this package is runner-side; see the
    package docstring.

    params
    ------
    orders : 'all' | [[m, n], ...]
        Which orders to carry.  'all' is the design's own 4x8 table.
    cell_pixels : int
        Dammann cell resolution (default 128, the design's own).
    chain_a : {'n': int, 'dx0': float|None, 'ray_subsample': int,
               'n_workers': int, 'final_leg': str}
        The launch chain to the DOE plane.  Defaults follow
        ``_d121_common.chain_a`` except that ``ray_subsample`` follows the
        spec's chain stage, so a run cannot silently launch on one lattice and
        relay on another.
    """
    import sys
    repro = os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..',
        'repro_traced_carrier_121'))
    if repro not in sys.path:
        sys.path.insert(0, repro)
    import _d121_common as C                                 # noqa: E402
    import lumenairy as la                                   # noqa: E402
    import lumenairy.propagators.carrier as CAR              # noqa: E402

    if abs(float(spec.wavelength) - C.LAM) > 1e-18:
        raise SpecError(
            f"decompose.design121_doe: the spec is at wavelength "
            f"{spec.wavelength!r} m but design 121 is a {C.LAM!r} m design "
            f"and its prescription, Dammann period and glass table are all "
            f"evaluated there.  Refusing to decompose one design at another "
            f"wavelength.")

    ca = dict(params.get('chain_a') or {})
    n_a = int(ca.get('n', 1024))
    dx0 = ca.get('dx0')
    rs = int(ca.get('ray_subsample', spec.chain.ray_subsample))
    nw = int(ca.get('n_workers', spec.chain.n_workers))
    leg_a = str(ca.get('final_leg', 'exact'))

    _pre, groups_post, _gap, period = C.geometry()
    mx, my, amps = C.order_table(period,
                                 n_per=int(params.get('cell_pixels', 128)))
    env_doe, R_doe, dx_doe, P_in = C.chain_a(n=n_a, dx0=dx0, rs=rs, nw=nw,
                                             final_leg=leg_a)

    want = params.get('orders', 'all')
    table = [(int(mx[i]), int(my[i]), complex(amps[i]))
             for i in range(len(mx))]
    if want != 'all':
        wset = [tuple(int(v) for v in o) for o in want]
        have = {(m, n) for m, n, _a in table}
        miss = [o for o in wset if o not in have]
        if miss:
            raise SpecError(
                f"decompose.design121_doe: order(s) {miss} are not in the "
                f"design's own {len(table)}-order table "
                f"{sorted((m, n) for m, n, _ in table)}.")
        keep = set(wset)
        table = [t for t in table if (t[0], t[1]) in keep]

    # Frame centres: the LIBRARY's own paraxial chief-ray predictor, snapped
    # to the output lattice -- exactly what
    # ``propagate_traced_carrier_chain_multi._window`` does, so the pipeline
    # places its windows where the shipped fan places them and a per-frame
    # comparison is like for like.  The independent EXACT SKEW TRACE is
    # recorded alongside as a diagnostic (fan_multi_121 scores its acceptance
    # on that one, precisely so the partition does not inherit the placement
    # it is checking); the two agree to sub-micron on this design.
    from lumenairy.raytrace import make_ray, trace                # noqa: E402
    surfs = C.post_surfaces(groups_post)
    dxo = float(spec.readout.dx_out)
    beams: List[Beam] = []
    for i, (m, n, a) in enumerate(table):
        L = float(m) * C.LAM / period
        M = float(n) * C.LAM / period
        car = la.TiltedCarrier(float(R_doe), L, M)
        x, y, _L, _M = CAR._chain_chief_ray_at_target(
            groups_post, C.LAM, car, C.TRAILING, 'pipeline.design121_doe')
        tr = trace(make_ray(0.0, 0.0, L, M, wavelength=C.LAM), surfs, C.LAM)
        beams.append(Beam(
            key=order_tag(m, n), index=i, weight=a,
            frame_centre=(round(x / dxo) * dxo, round(y / dxo) * dxo),
            label=f"({m:+d},{n:+d})",
            payload={'order': [m, n], 'tilt': [L, M], 'R_doe': float(R_doe),
                     'chief_pred': [float(x), float(y)],
                     'chief_exact': [float(tr.image_rays.x[0]),
                                     float(tr.image_rays.y[0])]}))
    ctx = {'groups_post': groups_post, 'env_doe': env_doe,
           'dx_doe': float(dx_doe), 'R_doe': float(R_doe), 'P_in': float(P_in),
           'period': float(period), 'trailing': float(C.TRAILING),
           'design': 'd121', 'chain_a': {'n': n_a, 'ray_subsample': rs,
                                         'n_workers': nw, 'final_leg': leg_a}}
    return DecomposeResult(beams=beams, context=ctx)


# ===========================================================================
# DECOMPOSER 2 -- the archived probe metadata
# ===========================================================================
def decompose_sumap_cache(params, spec: PipelineSpec) -> DecomposeResult:
    """Beams read from PROBE_SUM_AT_APERTURE's ARCHIVED arm-A metadata.

    WHY THIS EXISTS AND WHY IT IS THE STRONG FORM.  A reproduction claim that
    re-runs the chain is a claim about the chain as well as about the
    pipeline.  This decomposer (with the ``cached_aperture`` runner) consumes
    the probe's own ``_sumap_A_<tag>_nfc<NFC>.npz`` -- the weight, the snapped
    window centre, the exit congruence and the arm-A tile -- and its
    ``_sumap_ap_<tag>_nfc<NFC>.npy`` back-aperture field, byte for byte.  So
    NOTHING upstream of the seam can differ, and the S1.2 hazard (a library
    edit moving a chain's absolute phase while every intensity metric holds)
    cannot contaminate the comparison: the aperture field is a file, not a
    computation.

    params: ``cache_dir`` (read-only), ``orders``, ``nfc``.
    """
    cache = str(params['cache_dir'])
    nfc = int(params.get('nfc', 8192))
    orders = params.get('orders')
    if not orders:
        raise SpecError("decompose.sumap_cache: 'orders' is required -- this "
                        "decomposer reads named archived artifacts and will "
                        "not glob a directory into a beam list.")
    beams: List[Beam] = []
    for i, o in enumerate(orders):
        m, n = (int(v) for v in o)
        tag = order_tag(m, n)
        mp = os.path.join(cache, f'_sumap_A_{tag}_nfc{nfc}.npz')
        ap = os.path.join(cache, f'_sumap_ap_{tag}_nfc{nfc}.npy')
        for p in (mp, ap):
            if not os.path.exists(p):
                raise SpecError(
                    f"decompose.sumap_cache: {p} is missing.  Rebuild it with "
                    f"`python sumap_probe_121.py arma` in the tree that owns "
                    f"it, or point decompose.params.cache_dir at the tree "
                    f"that has it.")
        d = np.load(mp, allow_pickle=False)
        M_ = json.loads(str(d['meta']))
        beams.append(Beam(
            key=tag, index=i,
            weight=complex(M_['weight'][0], M_['weight'][1]),
            frame_centre=(M_['centre_out'][0], M_['centre_out'][1]),
            label=f"({m:+d},{n:+d})",
            payload={'order': [m, n], 'meta_path': mp, 'ap_path': ap,
                     'nfc': nfc}))
    return DecomposeResult(beams=beams, context={'design': 'sumap_cache',
                                                 'cache_dir': cache})


# ===========================================================================
# DECOMPOSER 3 -- explicit
# ===========================================================================
def decompose_explicit(params, spec: PipelineSpec) -> DecomposeResult:
    """The beam list written out in the spec, verbatim.  Generic."""
    raw = params.get('beams')
    if not raw:
        raise SpecError("decompose.explicit: params.beams is required and "
                        "must be a non-empty list.")
    beams = [Beam.from_dict({'index': i, **b}) for i, b in enumerate(raw)]
    keys = [b.key for b in beams]
    if len(set(keys)) != len(keys):
        raise SpecError(f"decompose.explicit: duplicate beam key(s) in "
                        f"{keys} -- keys name files.")
    return DecomposeResult(beams=beams,
                           context=dict(params.get('context') or {}))


# ===========================================================================
# CHAIN RUNNER 1 -- the real traced chain, spied at the fine-retrace exit
# ===========================================================================
class _RetraceOnly(Exception):
    """Internal sentinel: the readout was skipped on purpose."""


def run_chain_traced(beam: Beam, context, spec: PipelineSpec) -> ChainOutput:
    """One congruence to the last group's back aperture, ON THE FINE RETRACE
    GRID.

    THE SEAM, restated because it is the whole design.  The chain CAN be
    stopped at the last group's exit vertex (``final_distance=0``, no
    ``focus_readout``), and that plane is useless for aggregation: design 121
    lands there at dx = 33.2 um where the exit sphere needs 4.26 um -- 7.8x
    under-sampled -- and ``carrier_field.re_reference``'s Nyquist guard
    REFUSES it, naming the ``reconstruct`` term.  The plane that works is
    internal to the chain, produced by ``_fine_trace_group_exit``, so it is
    reached here the way the probe reached it: a READ-ONLY spy that records
    the array the chain then propagates.  Nothing is monkey-patched into the
    physics; the wrapper calls through and returns the original tuple.

    THE GRID OF RECORD IS ASSERTED, not inferred from the absence of a
    warning (ADJUDICATION_NFC_8192 S2.1: a 137 GB box silently ran a 16384
    request at 8192 and printed a passing acceptance banner).  Two independent
    checks: ``_grid_intent.preflight`` PROVES beforehand that the RAM clamp
    cannot bind, and the returned aperture field's own N is compared with the
    request afterwards.
    """
    import sys
    repro = os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..',
        'repro_traced_carrier_121'))
    if repro not in sys.path:
        sys.path.insert(0, repro)
    import _grid_intent as gi                                # noqa: E402
    import lumenairy as la                                   # noqa: E402
    import lumenairy.propagators.carrier as CAR              # noqa: E402

    cs, rs_ = spec.chain, spec.readout
    L, M = (float(v) for v in beam.payload['tilt'])
    car = la.TiltedCarrier(float(beam.payload['R_doe']), L, M)
    ramb = _resolve_ram_budget(cs.ram_budget)
    fr = dict(dx_out=float(rs_.dx_out), N_out=int(rs_.n_out),
              centre_out=tuple(beam.frame_centre),
              n_fine_cap=int(cs.n_fine_cap),
              window_factor=float(cs.window_factor))
    if ramb is not None:
        fr['ram_budget'] = ramb

    gi.preflight(cs.n_fine_cap, cs.ram_budget, workers=1,
                 n_px=int(np.asarray(context['env_doe']).size),
                 n_out=int(rs_.n_out), paraxial=(cs.final_leg == 'paraxial'),
                 on_box_budget=cs.on_box_budget,
                 label=f"{spec.name}: beam {beam.label}, "
                       f"NFC={cs.n_fine_cap} rs={cs.ray_subsample}")

    cap: Dict[str, Any] = {}
    o_ftge = CAR._fine_trace_group_exit
    o_ro = CAR.carrier_referenced_exact_focus_readout

    def _spy_ftge(*a, **kw):
        t0 = time.perf_counter()
        out = o_ftge(*a, **kw)
        cap['t_retrace'] = time.perf_counter() - t0
        cap['E_ap'] = out[0]
        cap['dx_ap'] = float(out[1])
        cap['R_out'] = float(a[8])
        cap['grid_origin'] = tuple(
            float(v) for v in kw['grid_origin_out'].get('origin', (0.0, 0.0)))
        return out

    def _spy_ro(E_full, R, z, wl, dx, **kw):
        if not cs.capture_reference_tile:
            # The stage's PRODUCT is the aperture field; the per-order tile is
            # a by-product.  When it is not wanted, the readout it costs
            # (11.6 % of an order, PROBE_SUM_AT_APERTURE S7.1) is not spent.
            # Returning a zero tile rather than raising keeps the chain's own
            # return path -- and its stage bookkeeping -- intact.
            cap['t_readout'] = 0.0
            cap['readout_skipped'] = True
            return np.zeros((int(kw['N_out']), int(kw['N_out'])),
                            dtype=np.complex128)
        t0 = time.perf_counter()
        out = o_ro(E_full, R, z, wl, dx, **kw)
        cap['t_readout'] = time.perf_counter() - t0
        return out

    t0 = time.perf_counter()
    with gi.record_warnings() as wrec:
        CAR._fine_trace_group_exit = _spy_ftge
        CAR.carrier_referenced_exact_focus_readout = _spy_ro
        try:
            res = CAR.propagate_traced_carrier_chain(
                context['env_doe'], context['groups_post'],
                float(spec.wavelength), float(context['dx_doe']), r_in=car,
                ray_subsample=int(cs.ray_subsample),
                n_workers=int(cs.n_workers),
                final_distance=float(context['trailing']),
                focus_readout=fr, final_leg=str(cs.final_leg))
        finally:
            CAR._fine_trace_group_exit = o_ftge
            CAR.carrier_referenced_exact_focus_readout = o_ro
    wall = time.perf_counter() - t0
    if cs.final_leg != 'paraxial':
        gi.assert_no_grid_degradation(wrec, cs.n_fine_cap,
                                      label=f"{spec.name}: {beam.label}")
    if 'E_ap' not in cap:
        raise RuntimeError(
            f"chain 'traced' ({beam.label}): the chain never called "
            f"_fine_trace_group_exit, so it did not run the EXACT final leg "
            f"and there is no fine-retrace exit plane to sum on.  "
            f"chain.final_leg={cs.final_leg!r} resolved to the paraxial "
            f"route; the summable plane of PROBE_SUM_AT_APERTURE S2 does not "
            f"exist on it.")
    E = np.asarray(cap['E_ap'])
    if E.shape[-1] != int(cs.n_fine_cap):
        raise RuntimeError(
            f"chain 'traced' ({beam.label}): the retrace ran at n_fine="
            f"{E.shape[-1]}, not the requested {cs.n_fine_cap}.  The RAM "
            f"clamp degraded the grid of record after the pre-flight proved "
            f"it could not; refusing to write an artifact that would wear the "
            f"wrong label.")

    st = [s for s in res.stages if s.get('exact_final') and 'na_exit' in s]
    st0 = st[0] if st else {}
    meta = _aperture_meta(
        E, cap['dx_ap'], cap['grid_origin'], cap['R_out'],
        (st0.get('x_c_out', 0.0), st0.get('y_c_out', 0.0)),
        (st0.get('L_out', 0.0), st0.get('M_out', 0.0)),
        st0.get('power'))
    meta.update(
        runner='traced', order=beam.payload.get('order'),
        na_exit=(None if st0.get('na_exit') is None
                 else float(st0['na_exit'])),
        na_exit_measured=(None if st0.get('na_exit_measured') is None
                          else float(st0['na_exit_measured'])),
        ray_subsample=int(cs.ray_subsample), n_fine_cap=int(cs.n_fine_cap),
        window_factor=float(cs.window_factor),
        t_retrace=float(cap.get('t_retrace', float('nan'))),
        t_readout=float(cap.get('t_readout', float('nan'))),
        readout_skipped=bool(cap.get('readout_skipped', False)),
        wall=float(wall),
        warnings=[r['msg'][:200] for r in wrec.records][:20])
    tile = (np.asarray(res.field) * beam.weight
            if cs.capture_reference_tile else None)
    fld = wrap_aperture_field(
        E, meta, spec.wavelength,
        provenance={'beam': beam.key, 'label': beam.label,
                    'runner': 'traced', 'order': beam.payload.get('order')})
    return ChainOutput(field=fld, meta=meta, reference_tile=tile)


# ===========================================================================
# CHAIN RUNNER 2 -- the archived aperture field
# ===========================================================================
def run_chain_cached_aperture(beam: Beam, context,
                              spec: PipelineSpec) -> ChainOutput:
    """The byte-identical archived back-aperture field, wrapped.

    No chain runs.  Used with the ``sumap_cache`` decomposer this is the
    strongest available isolation for a reproduction claim -- see that
    decomposer's docstring."""
    mp = beam.payload['meta_path']
    ap = beam.payload['ap_path']
    d = np.load(mp, allow_pickle=False)
    M_ = json.loads(str(d['meta']))
    tileA = np.asarray(d['tile'])
    t0 = time.perf_counter()
    E = np.load(ap)
    t_load = time.perf_counter() - t0
    meta = _aperture_meta(E, M_['dx_ap'], M_['grid_origin'], M_['R_out'],
                          M_['chief_exit'], M_['tilt_exit'],
                          M_.get('power_exit'))
    if int(meta['n_ap']) != int(M_['n_ap']):
        raise RuntimeError(
            f"chain 'cached_aperture' ({beam.label}): {ap} is "
            f"{meta['n_ap']}-square but its metadata says {M_['n_ap']}.  The "
            f"pair is inconsistent; refusing.")
    meta.update(runner='cached_aperture', order=beam.payload.get('order'),
                na_exit=M_.get('na_exit'),
                na_exit_measured=M_.get('na_exit_measured'),
                ray_subsample=None, n_fine_cap=int(M_['n_ap']),
                window_factor=float(M_['ro'].get('window_factor', float('nan'))),
                source_meta=mp, source_field=ap,
                t_retrace=float(M_.get('t_retrace', float('nan'))),
                t_readout=float(M_.get('t_readout', float('nan'))),
                readout_skipped=False, wall=float(t_load), warnings=[])
    fld = wrap_aperture_field(
        E, meta, spec.wavelength,
        provenance={'beam': beam.key, 'label': beam.label,
                    'runner': 'cached_aperture',
                    'order': beam.payload.get('order'),
                    'source': os.path.basename(ap)})
    return ChainOutput(field=fld, meta=meta,
                       reference_tile=(tileA if spec.chain.capture_reference_tile
                                       else None))


# ---------------------------------------------------------------------------
register_decomposer('design121_doe', decompose_design121_doe)
register_decomposer('sumap_cache', decompose_sumap_cache)
register_decomposer('explicit', decompose_explicit)
register_chain_runner('traced', run_chain_traced)
register_chain_runner('cached_aperture', run_chain_cached_aperture)

# 'design121_doe_rcwa' -- the same decomposition with RIGOROUS (RCWA) order
# amplitudes instead of the thin-element DFT.  Imported HERE, at the bottom,
# for its registration side effect: that module imports Beam /
# DecomposeResult / order_tag from this one, so it can only be imported once
# this module is fully defined.  BUILD_RCWA_DOE_TABLE_2026_08_12.
from . import doe_rcwa as _doe_rcwa      # noqa: E402,F401
