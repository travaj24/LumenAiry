# validation/pipeline/driver.py -- the five stages, the resume logic, and the
# admissibility banner.
#
# THE STAGES, AND WHY THE BOUNDARIES ARE WHERE THEY ARE
# ----------------------------------------------------
#   decompose  a system -> a BEAM LIST (initial carriers + complex weights).
#   chains     each beam -> the ONE summable plane, independently.  This is
#              the expensive stage (76 % of an order is the fine re-trace,
#              PROBE_SUM_AT_APERTURE S7.1) and it is the one that parallelises
#              trivially, so it is also the one that must be per-beam
#              resumable: a 32-beam run that dies on beam 29 must not re-run
#              28 chains.
#   aggregate  K fields -> ONE field on ONE common carrier, with the Nyquist
#              guard LIVE and an energy ledger per beam.
#   leg        the summed field -> a resolved plan for the exact final leg.
#   readout    that plan -> one Bluestein zoom per frame, plus metrics.
#
# A CHECKPOINT IS A STAGE BOUNDARY, and each artifact carries the full key of
# everything it depends on (see ``artifacts``).  So resume is not "the file
# exists": it is "the file exists AND was produced by this configuration, this
# library and this driver".
#
# COHERENT-SUM ADMISSIBILITY -- READ THIS BEFORE TRUSTING A PISTON COLUMN
# ----------------------------------------------------------------------
# Summing FIELDS across chains needs the per-chain absolute optical path to be
# right.  On a library without ``FIX_TILT_QUADRATIC_OPL_2026_08_11`` it is
# not: PROBE_CHAIN_LADDER_PISTON S3.6 scored the inter-order piston against an
# exact skew ray trace of the same chief ray and found it wrong by 0.050 to
# 0.416 WAVES -- 5x to 42x outside lambda/100 -- reproducibly, and invariant
# under grid, ray lattice and every element-fit lever.  The mechanism (S3.7)
# is a tilt-quadratic obliquity piston the traced element hand-off drops per
# group.
#
# What that does and does not invalidate:
#
#   * ENERGY and ENVELOPE SHAPE per beam are sound, and the beams land on
#     separate frames, so every INTENSITY column of this pipeline (EE, FWHM,
#     window power, crosstalk) is unaffected -- which is exactly why the
#     defect survived every shipped acceptance metric;
#   * the PISTON columns are a measurement of the aggregation only when the
#     chains feeding it carry an absolute path.  Without the fix they measure
#     the aggregation's own contribution ON TOP OF a wrong per-chain constant,
#     so they are recorded and FLAGGED rather than scored.
#
# This driver detects the fix and says so, loudly, once per run.
#
# cp1252-safe ASCII only.
from __future__ import annotations

import os
import time
import warnings
from dataclasses import dataclass, field as _dcfield
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from . import artifacts as A
from .metrics import compare, spot
from .sources import (Beam, get_chain_runner, get_decomposer,
                      _readout_grid_from_meta, _resolve_ram_budget)
from .spec import STAGES, PipelineSpec, SpecError, resolve_selection


class PipelineError(RuntimeError):
    """A run that cannot proceed (as opposed to a config that cannot describe
    one -- that is :class:`~.spec.SpecError`)."""


# ---------------------------------------------------------------------------
# coherent-sum admissibility
# ---------------------------------------------------------------------------
_PISTON_FIX_MARK = 'FIX_TILT_QUADRATIC_OPL'


def piston_fix_status():
    """``('present'|'absent'|'unknown', detail)`` for the absolute-OPL fix.

    Detected by scanning the SOURCE of the ``_lens_traced`` module that is
    actually imported for the fix's own marker.  A source probe is a blunt
    instrument and is named as such: the fix adds no public symbol, and a
    BEHAVIOURAL probe would cost a ray trace through a tilted group on every
    run.  The real protection is not this function -- it is that every
    artifact this pipeline writes is keyed on a content hash of
    ``lumenairy/**`` (``artifacts.lumenairy_source_sha``), so a library that
    gains the fix mid-campaign orphans every field produced without it instead
    of mixing the two."""
    try:
        from lumenairy.elements import _lens_traced
        path = os.path.abspath(_lens_traced.__file__)
        with open(path, 'r', encoding='utf-8', errors='replace') as fh:
            src = fh.read()
    except Exception as exc:                                  # pragma: no cover
        return 'unknown', f"could not read _lens_traced source: {exc}"
    if _PISTON_FIX_MARK in src:
        return 'present', path
    return 'absent', path


def admissibility_banner(spec: PipelineSpec, log=print) -> Dict[str, Any]:
    """Print, warn, and return the coherent-sum admissibility record."""
    status, detail = piston_fix_status()
    rec = {'piston_fix': status, 'detail': detail,
           'coherent_sum_declared': bool(spec.coherent_sum),
           'piston_columns_valid': status == 'present'}
    bar = '=' * 74
    log(bar)
    log(f"COHERENT-SUM ADMISSIBILITY -- {spec.name}")
    log(bar)
    if status == 'present':
        log("  FIX_TILT_QUADRATIC_OPL is PRESENT in this library: the traced")
        log("  element returns an ABSOLUTE optical path, so the inter-chain")
        log("  piston is a real quantity and a coherent sum is admissible.")
    else:
        msg = (
            "COHERENT-SUM NOT ADMISSIBLE ON THIS LIBRARY.  "
            "FIX_TILT_QUADRATIC_OPL_2026_08_11 (branch fix/tilt-quadratic-opl) "
            "is %s in %s.  Without it the traced element hand-off drops a "
            "tilt-quadratic obliquity piston per group, and the inter-chain "
            "piston at the summable plane is REPRODUCIBLE AND WRONG by "
            "0.050-0.416 waves (5x-42x outside lambda/100), measured against "
            "an exact skew-ray oracle in PROBE_CHAIN_LADDER_PISTON S3.6.  "
            "Every INTENSITY column of this run (EE, FWHM, window power, "
            "crosstalk) is unaffected -- the beams land on separate frames and "
            "the defect is a per-beam constant -- but every PISTON column is "
            "recorded PRE-FIX and must not be scored.  Merge that branch "
            "before using this pipeline for anything phase-sensitive."
            % ('ABSENT' if status == 'absent' else 'UNDETERMINED', detail))
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
        for line in ('  !! FIX_TILT_QUADRATIC_OPL IS %s.' % status.upper(),
                     '  !! Inter-chain piston is reproducible and WRONG by',
                     '  !!   0.050-0.416 waves (PROBE_CHAIN_LADDER_PISTON '
                     'S3.6).',
                     '  !! INTENSITY columns are unaffected and are scored.',
                     '  !! PISTON columns are recorded PRE-FIX and FLAGGED.'):
            log(line)
        if spec.coherent_sum:
            log('  !! spec.coherent_sum=True was DECLARED on a library that')
            log('  !!   cannot support it.  The run proceeds -- the sum is')
            log('  !!   still exactly linear and the intensity answer is')
            log('  !!   still right -- but the relative phases between beams')
            log('  !!   are not usable as they stand.')
    log(bar)
    return rec


# ---------------------------------------------------------------------------
# run state
# ---------------------------------------------------------------------------
@dataclass
class RunState:
    spec: PipelineSpec
    workdir: str
    manifest: A.Manifest
    log: Any = print
    beams: List[Beam] = _dcfield(default_factory=list)
    context: Dict[str, Any] = _dcfield(default_factory=dict)
    keys: Dict[str, Any] = _dcfield(default_factory=dict)
    digests: Dict[str, str] = _dcfield(default_factory=dict)
    cache: Dict[str, Any] = _dcfield(default_factory=dict)
    report: Dict[str, Any] = _dcfield(default_factory=dict)

    def path(self, *parts):
        return os.path.join(self.workdir, *parts)

    @property
    def beam_keys(self):
        return [b.key for b in self.beams]

    def beam(self, key):
        for b in self.beams:
            if b.key == key:
                return b
        raise KeyError(key)


def _tag_for(state: RunState) -> str:
    inc = resolve_selection(state.spec.aggregate.include, state.beam_keys,
                            'aggregate.include')
    return A.selection_tag(inc)


# ---------------------------------------------------------------------------
# STAGE 1 -- decompose
# ---------------------------------------------------------------------------
def stage_decompose(state: RunState, force=False):
    spec = state.spec
    key, digest = A.stage_key(spec, 'decompose')
    state.keys['decompose'], state.digests['decompose'] = key, digest
    path = state.path('decompose', 'beams.json')
    hit = None if force else A.read_json(path, key)
    fn = get_decomposer(spec.decompose.kind)
    if hit is not None:
        state.beams = [Beam.from_dict(b) for b in hit['beams']]
        state.log(f"  [decompose] RESUMED {len(state.beams)} beam(s) from "
                  f"{os.path.relpath(path, state.workdir)}")
        state.report['decompose'] = {'resumed': True, **hit['summary']}
        # The CONTEXT is not checkpointed (it holds arrays); it is rebuilt
        # only when a later stage actually needs it.  See ``ensure_context``.
        return
    t0 = time.perf_counter()
    res = fn(dict(spec.decompose.params), spec)
    wall = time.perf_counter() - t0
    if not res.beams:
        raise PipelineError(f"decompose {spec.decompose.kind!r} produced no "
                            f"beams.")
    keys = [b.key for b in res.beams]
    if len(set(keys)) != len(keys):
        raise PipelineError(f"decompose {spec.decompose.kind!r} produced "
                            f"duplicate beam keys: {keys}.")
    state.beams = list(res.beams)
    state.context = dict(res.context)
    p_tot = float(sum(abs(b.weight) ** 2 for b in state.beams))
    summary = {'n_beams': len(state.beams), 'kind': spec.decompose.kind,
               'sum_abs_weight_sq': p_tot, 'wall': wall}
    A.write_json(path, {'beams': [b.to_dict() for b in state.beams],
                        'summary': summary}, key)
    state.manifest.record('decompose', digest, wall=wall,
                          n_beams=len(state.beams))
    state.log(f"  [decompose] {len(state.beams)} beam(s) via "
              f"{spec.decompose.kind!r} in {wall:.1f}s; "
              f"sum|a|^2 = {p_tot:.6f}")
    state.report['decompose'] = {'resumed': False, **summary}


def ensure_context(state: RunState):
    """Rebuild the decomposer's context on demand (resume path).

    Cheap by construction: every expensive input a decomposer builds is
    itself cached (``_d121_common.chain_a``), and the beam LIST is read from
    the checkpoint rather than recomputed, so the two can never drift."""
    if state.context:
        return state.context
    fn = get_decomposer(state.spec.decompose.kind)
    t0 = time.perf_counter()
    res = fn(dict(state.spec.decompose.params), state.spec)
    have = {b.key for b in state.beams}
    got = {b.key for b in res.beams}
    if have and have != got:
        raise PipelineError(
            f"ensure_context: the decomposer now yields beams {sorted(got)} "
            f"but the checkpoint holds {sorted(have)}.  The key check passed, "
            f"so the spec and the code are unchanged and the DESIGN INPUTS "
            f"moved under the run (a .zmx, a Dammann cache, an order table).  "
            f"Refusing to mix them.")
    state.context = dict(res.context)
    state.log(f"  [context] rebuilt in {time.perf_counter() - t0:.1f}s")
    return state.context


# ---------------------------------------------------------------------------
# STAGE 2 -- chains
# ---------------------------------------------------------------------------
def _chain_paths(state: RunState, key: str):
    return (state.path('chains', f'{key}.zarr'),
            state.path('chains', f'{key}.json'),
            state.path('chains', f'{key}_ref.npy'))


def stage_chains(state: RunState, force=False):
    spec = state.spec
    # PER-BEAM artifacts hang off the decompose LINEAGE, not off the decompose
    # stage digest, so adding or dropping a sibling beam does not orphan the
    # ones that did not change.  See artifacts.decompose_lineage_digest.
    up = A.decompose_lineage_digest(spec)
    runner = get_chain_runner(spec.chain.kind)
    done, ran = 0, 0
    t_stage = time.perf_counter()
    per_beam: Dict[str, Any] = {}
    for beam in state.beams:
        key, digest = A.stage_key(spec, 'chains', upstream=up,
                                  extra={'beam': beam.key,
                                         'weight': [beam.weight.real,
                                                    beam.weight.imag],
                                         'frame_centre':
                                             list(beam.frame_centre),
                                         'payload': beam.payload})
        fz, fj, fr = _chain_paths(state, beam.key)
        meta = None if force else A.read_json(fj, key)
        if meta is not None:
            # CONTENT, not presence.  The stored power is compared against
            # the power re-integrated off the checkpoint; a partial store
            # reads as zeros in the missing block and would otherwise resume
            # as if it were whole (VERIFY_ARCHITECTURE P0-3).
            ok, why = A.field_is_complete(fz, meta.get('power_field'))
            if ok:
                per_beam[beam.key] = {**meta, 'resumed': True,
                                      'key': digest,
                                      'resume_check': why}
                done += 1
                continue
            if A.field_exists(fz):
                state.log(f"  [chains] {beam.label} ({beam.key}) checkpoint "
                          f"REJECTED and will be recomputed: {why}")
        state.log(f"  [chains] {beam.label} ({beam.key}) ...")
        t0 = time.perf_counter()
        out = runner(beam, ensure_context(state), spec)
        A.save_field(fz, out.field, key,
                     extra_provenance={'beam': beam.key,
                                       'stage': 'chains',
                                       'pipeline': spec.name})
        m = dict(out.meta)
        m['stage_wall'] = time.perf_counter() - t0
        m['peak_wset_gb'], m['peak_commit_gb'] = A.peak_rss_gb()
        m['has_reference_tile'] = out.reference_tile is not None
        if out.reference_tile is not None:
            A.save_npy_atomic(fr, out.reference_tile)
            m['reference_tile_power'] = float(
                (np.abs(out.reference_tile) ** 2).sum()) * \
                float(spec.readout.dx_out) ** 2
        # ENERGY LEDGER, per beam and at the plane the sum is taken on: the
        # power the chain reports at its own exit stage against the power
        # actually integrated on the returned grid.  The probe measured this
        # ratio at 1.000000000 for every order; a run whose beams do not is a
        # run whose seam moved.
        if m.get('power_exit'):
            m['power_ratio_field_over_exit'] = (m['power_field']
                                                / m['power_exit'])
        A.write_json(fj, m, key)
        per_beam[beam.key] = {**m, 'resumed': False, 'key': digest}
        ran += 1
        state.log(f"  [chains] {beam.label} done in "
                  f"{m['stage_wall']:.1f}s  dx_ap "
                  f"{m['dx_ap'] * 1e6:.4f} um  N {m['n_ap']}  R_out "
                  f"{m['R_out'] * 1e3:.6f} mm  origin "
                  f"({m['grid_origin'][0] * 1e3:+.4f},"
                  f"{m['grid_origin'][1] * 1e3:+.4f}) mm"
                  + (f"  P/P_exit "
                     f"{m['power_ratio_field_over_exit']:.9f}"
                     if 'power_ratio_field_over_exit' in m else ''))
    wall = time.perf_counter() - t_stage
    # The stage's own digest is the digest of the SET, so a downstream stage
    # is orphaned when any beam changes -- including a beam being added.
    _skey, sdig = A.stage_key(spec, 'chains', upstream=up,
                              extra={'beams': sorted(
                                  (k, v['key']) for k, v in per_beam.items())})
    state.digests['chains'] = sdig
    state.cache['chain_meta'] = per_beam
    state.manifest.record('chains', sdig, wall=wall, ran=ran, resumed=done,
                          n_beams=len(state.beams))
    state.log(f"  [chains] {ran} run, {done} resumed, {wall:.1f}s")
    state.report['chains'] = {'ran': ran, 'resumed': done, 'wall': wall,
                              'beams': per_beam}


def chain_meta(state: RunState, key: str) -> Dict[str, Any]:
    cm = state.cache.get('chain_meta')
    if cm and key in cm:
        return cm[key]
    _k, _d = None, None
    fj = _chain_paths(state, key)[1]
    if not os.path.exists(fj):
        raise PipelineError(f"chain artifact for beam {key!r} is missing "
                            f"({fj}); run the 'chains' stage.")
    import json as _json
    with open(fj, 'r', encoding='cp1252') as fh:
        return _json.load(fh)['payload']


# ---------------------------------------------------------------------------
# STAGE 3 -- aggregate
# ---------------------------------------------------------------------------
def stage_aggregate(state: RunState, force=False):
    from lumenairy.propagators.carrier_field import (CarrierSpec, FieldGrid,
                                                     aggregate)
    spec = state.spec
    ags = spec.aggregate
    inc = resolve_selection(ags.include, state.beam_keys, 'aggregate.include')
    tag = A.selection_tag(inc)
    up = state.digests['chains']
    key, digest = A.stage_key(spec, 'aggregate', upstream=up,
                              extra={'include': inc, 'tag': tag})
    state.keys['aggregate'], state.digests['aggregate'] = key, digest
    fz = state.path('aggregate', tag, 'summed.zarr')
    fj = state.path('aggregate', tag, 'ledger.json')
    state.cache['agg_tag'] = tag
    state.cache['agg_include'] = inc
    hit = None if force else A.read_json(fj, key)
    if hit is not None:
        # Content, not presence -- see stage_chains (P0-3).  The ledger
        # already records the summed field's own power.
        ok, why = A.field_is_complete(fz, hit.get('summed_power'))
        if ok:
            state.log(f"  [aggregate] RESUMED {tag} ({len(inc)} beam(s))")
            state.report['aggregate'] = {'resumed': True, **hit}
            return
        if A.field_exists(fz):
            state.log(f"  [aggregate] checkpoint REJECTED and will be "
                      f"recomputed: {why}")

    # THE COMMON CARRIER.  One sphere for the whole fan, taken from a NAMED
    # beam rather than averaged, because at a shared back aperture the
    # paraxial closure is order-independent (design 121 measures R_out =
    # -7.712425 mm for every order) and a mean would hide a design where it is
    # not.  The pipeline CHECKS that rather than assuming it.
    ref_key = ags.reference_beam or inc[0]
    if ref_key not in state.beam_keys:
        raise SpecError(f"aggregate.reference_beam={ref_key!r} is not a beam "
                        f"key ({state.beam_keys}).")
    R_c = float(chain_meta(state, ref_key)['R_out'])
    spread = [(k, float(chain_meta(state, k)['R_out'])) for k in inc]
    worst = max(abs(r / R_c - 1.0) for _k, r in spread)
    if worst > 1e-6:
        warnings.warn(
            f"aggregate: the beams do NOT share one exit sphere -- R_out "
            f"spreads by {worst:.3e} relative across {len(inc)} beam(s) "
            f"(reference {ref_key!r} at {R_c * 1e3:.6f} mm; extremes "
            f"{min(r for _k, r in spread) * 1e3:.6f} .. "
            f"{max(r for _k, r in spread) * 1e3:.6f} mm).  A single common "
            f"carrier is then an APPROXIMATION: the residual quadratic each "
            f"beam carries against it is real curvature, not a re-reference, "
            f"and PROBE_SUM_AT_APERTURE's S9 names that case as untested "
            f"(the mean sphere plus a per-beam residual quadratic is what it "
            f"needs).  The Nyquist guard will bind harder; the answer is not "
            f"guaranteed.", RuntimeWarning, stacklevel=2)
    car_c = CarrierSpec(R=R_c, centre=(0.0, 0.0), tilt=(0.0, 0.0), piston=0.0)
    grid_c = FieldGrid((ags.n_common, ags.n_common), ags.dx_common,
                       origin=ags.origin)

    bs = max(1, int(ags.batch_size))
    nbatch = (len(inc) + bs - 1) // bs
    state.log(f"  [aggregate] {tag}: {len(inc)} beam(s) onto "
              f"{ags.n_common}^2 at dx {ags.dx_common * 1e6:.4f} um "
              f"(window {ags.n_common * ags.dx_common * 1e3:.4f} mm), "
              f"R_c {R_c * 1e3:.6f} mm, {nbatch} batch(es) of <= {bs}"
              + ("  [ONE aggregate() call -- bit-identical to the "
                 "un-batched primitive]" if nbatch == 1 else ""))
    t0 = time.perf_counter()
    acc = None
    rows: List[Dict[str, Any]] = []
    for b0 in range(0, len(inc), bs):
        chunk = inc[b0:b0 + bs]
        fields, ws = [], []
        for k in chunk:
            bkey, _bd = A.stage_key(
                spec, 'chains', upstream=A.decompose_lineage_digest(spec),
                extra={'beam': k,
                       'weight': [state.beam(k).weight.real,
                                  state.beam(k).weight.imag],
                       'frame_centre': list(state.beam(k).frame_centre),
                       'payload': state.beam(k).payload})
            f = A.load_field(_chain_paths(state, k)[0], bkey)
            if f is None:
                raise PipelineError(
                    f"aggregate: the chain checkpoint for beam {k!r} is "
                    f"missing or was produced by a different configuration.  "
                    f"Re-run the 'chains' stage.")
            fields.append(f)
            ws.append(state.beam(k).weight if ags.weights == 'decompose'
                      else 1.0 + 0.0j)
        res = aggregate(fields, car_c, grid_c, weights=ws,
                        support_frac=ags.support_frac,
                        nyquist_margin=ags.nyquist_margin,
                        on_nyquist=ags.on_nyquist, on_window=ags.on_window,
                        bandlimit=ags.bandlimit)
        del fields
        for k, r in zip(chunk, res.ledger.rows):
            rows.append({'beam': k, **{f: getattr(r, f) for f in r._fields}})
        if acc is None:
            acc = res.field                     # ONE batch: take it whole, so
        else:                                   # the single-batch case is
            acc.envelope += res.field.envelope  # bit-identical to aggregate()
        del res
    wall = time.perf_counter() - t0
    p_src = float(sum(r['power_source'] for r in rows))
    p_com = float(sum(r['power_on_common'] for r in rows))
    ledger = {
        'tag': tag, 'include': inc, 'reference_beam': ref_key, 'R_c': R_c,
        'dx_common': ags.dx_common, 'n_common': ags.n_common,
        'origin': list(ags.origin), 'n_batches': nbatch, 'batch_size': bs,
        'rows': rows, 'power_source_total': p_src,
        'power_on_common_total': p_com,
        'power_out_of_window_total': p_src - p_com,
        'frac_out_of_window_total': ((p_src - p_com) / p_src if p_src else 0.0),
        'worst_containment_margin': float(min(r['containment_margin']
                                              for r in rows)),
        'worst_nyquist_margin': float(min(r['nyquist_margin'] for r in rows)),
        'R_out_spread_rel': worst, 'wall': wall,
        'peak_wset_gb': A.peak_rss_gb()[0],
        'summed_power': acc.power()}
    A.save_field(fz, acc, key, extra_provenance={
        'stage': 'aggregate', 'tag': tag, 'include': inc,
        'pipeline': spec.name})
    A.write_json(fj, ledger, key)
    state.manifest.record('aggregate', digest, wall=wall, tag=tag,
                          n_included=len(inc))
    state.cache['summed'] = acc
    for r in rows:
        state.log(f"    {r['beam']:>8s}  P_src {r['power_source']:.6e}  "
                  f"P_common {r['power_on_common']:.6e}  out-of-window "
                  f"{r['frac_out_of_window']:.3e}  support "
                  f"{r['support_radius'] * 1e3:.4f} mm  nyquist "
                  f"{r['nyquist_margin']:.3f}x ({r['binding_term']})  "
                  f"containment {r['containment_margin'] * 1e3:+.4f} mm")
    state.log(f"  [aggregate] {wall:.1f}s; worst Nyquist margin "
              f"{ledger['worst_nyquist_margin']:.3f}x, worst containment "
              f"{ledger['worst_containment_margin'] * 1e3:+.4f} mm")
    state.report['aggregate'] = {'resumed': False, **ledger}


def load_summed(state: RunState):
    if 'summed' in state.cache:
        return state.cache['summed']
    tag = state.cache.get('agg_tag') or _tag_for(state)
    f = A.load_field(state.path('aggregate', tag, 'summed.zarr'),
                     state.keys['aggregate'])
    if f is None:
        raise PipelineError(f"the aggregate checkpoint for {tag!r} is missing "
                            f"or stale; run the 'aggregate' stage.")
    state.cache['summed'] = f
    return f


# ---------------------------------------------------------------------------
# STAGE 4 -- the single leg (plan)
# ---------------------------------------------------------------------------
def stage_leg(state: RunState, force=False):
    from lumenairy.propagators.carrier import _envelope_amp_radius
    spec = state.spec
    lg = spec.leg
    tag = state.cache.get('agg_tag') or _tag_for(state)
    frames = resolve_selection(spec.readout.frames, state.beam_keys,
                               'readout.frames')
    key, digest = A.stage_key(spec, 'leg', upstream=state.digests['aggregate'],
                              extra={'tag': tag, 'frames': frames})
    state.keys['leg'], state.digests['leg'] = key, digest
    fj = state.path('aggregate', tag, f'leg_{lg.mode}.json')
    hit = None if force else A.read_json(fj, key)
    if hit is not None:
        state.cache['leg_plan'] = hit
        state.log(f"  [leg] RESUMED plan ({lg.mode}) for {len(frames)} "
                  f"frame(s)")
        state.report['leg'] = {'resumed': True, **hit}
        return
    t0 = time.perf_counter()
    fsum = load_summed(state)
    ox, oy = fsum.grid.origin
    dx_c = fsum.grid.dx
    # The full field is what the exact leg takes (it references the sphere
    # itself); build it ONCE here and hand it to the readout in-process.
    full = fsum.full_field()
    state.cache['summed_full'] = full
    plan = {'mode': lg.mode, 'tag': tag, 'R_c': float(fsum.carrier.R),
            'dx_c': float(dx_c), 'n_c': int(fsum.grid.shape[-1]),
            'origin': [float(ox), float(oy)],
            'distance': float(lg.distance), 'frames': {}}
    if (lg.mode == 'crop'
            and (float(lg.window_factor) != float(spec.chain.window_factor)
                 or int(lg.n_fine_cap) != int(spec.chain.n_fine_cap))):
        warnings.warn(
            f"leg.mode='crop' is the LIKE-FOR-LIKE variant: its window and "
            f"internal fine grid are meant to be the ones the shipped "
            f"per-order readout used, which are sized from the CHAIN's "
            f"window_factor ({spec.chain.window_factor}) and n_fine_cap "
            f"({spec.chain.n_fine_cap}).  This spec sizes the leg from "
            f"{lg.window_factor} / {lg.n_fine_cap} instead, so the per-frame "
            f"comparison against the shipped path is no longer like for like "
            f"-- a different window carries a different halo, which moves the "
            f"field relative L2 by more than the aggregation does "
            f"(PROBE_SUM_AT_APERTURE S5: 2.8e-05 crop vs 7.3e-04 full).",
            RuntimeWarning, stacklevel=2)
    for k in frames:
        m = chain_meta(state, k)
        ch = (float(m['chief_exit'][0]), float(m['chief_exit'][1]))
        cen = (ch[0] - ox, ch[1] - oy)
        beam = state.beam(k)
        co = (float(beam.frame_centre[0]) - ox,
              float(beam.frame_centre[1]) - oy)
        if lg.mode == 'crop':
            g = _readout_grid_from_meta(m, spec.wavelength, lg.window_factor,
                                        lg.n_fine_cap)
            w_sum = float(_envelope_amp_radius(full, dx_c, dx_c, centre=cen))
            plan['frames'][k] = {
                'centre': list(cen), 'centre_out': list(co),
                'N_fine': int(g['n_fine']),
                'window_factor': float(g['win'] / w_sum),
                'win': float(g['win']), 'dx_fine': float(g['dx_fine']),
                'n_crop': int(g['n_crop']), 'w_armA': float(g['w_amp']),
                'w_sum': w_sum}
        else:
            # 'full': one leg on the WHOLE summed aperture at the common
            # pitch.  window_factor = 1e6 is the probe's own spelling of "the
            # crop is the entire grid" (the readout clamps n_crop to N), and
            # it is kept literally so this mode reproduces the probe's `full`
            # tables rather than merely matching their intent.
            plan['frames'][k] = {
                'centre': [0.0, 0.0], 'centre_out': list(co),
                'N_fine': int(fsum.grid.shape[-1]), 'window_factor': 1e6,
                'win': float(fsum.grid.shape[-1] * dx_c),
                'dx_fine': float(dx_c),
                'n_crop': int(fsum.grid.shape[-1]),
                'w_armA': float(m['w_amp']),
                'w_sum': float(_envelope_amp_radius(full, dx_c, dx_c,
                                                    centre=cen))}
    plan['wall'] = time.perf_counter() - t0
    A.write_json(fj, plan, key)
    state.cache['leg_plan'] = plan
    state.manifest.record('leg', digest, wall=plan['wall'], mode=lg.mode,
                          n_frames=len(frames))
    for k, p in plan['frames'].items():
        state.log(f"    {k:>8s}  win {p['win'] * 1e3:.6f} mm  n_fine "
                  f"{p['N_fine']}  dx_fine {p['dx_fine'] * 1e6:.10f} um  "
                  f"centre ({p['centre'][0] * 1e3:+.4f},"
                  f"{p['centre'][1] * 1e3:+.4f}) mm")
    state.log(f"  [leg] plan ({lg.mode}) for {len(frames)} frame(s) in "
              f"{plan['wall']:.1f}s")
    state.report['leg'] = {'resumed': False, **plan}


# ---------------------------------------------------------------------------
# STAGE 5 -- the per-frame readout plan
# ---------------------------------------------------------------------------
def stage_readout(state: RunState, force=False):
    from lumenairy.propagators.carrier import (
        carrier_referenced_exact_focus_readout)
    spec = state.spec
    ro, lg = spec.readout, spec.leg
    tag = state.cache.get('agg_tag') or _tag_for(state)
    frames = resolve_selection(ro.frames, state.beam_keys, 'readout.frames')
    key, digest = A.stage_key(spec, 'readout', upstream=state.digests['leg'],
                              extra={'tag': tag, 'frames': frames})
    state.keys['readout'], state.digests['readout'] = key, digest
    out_dir = state.path('readout', f'{tag}__{lg.mode}')
    fj = os.path.join(out_dir, 'metrics.json')
    fn = os.path.join(out_dir, 'frames.npz')
    hit = None if force else A.read_json(fj, key)
    if hit is not None and (os.path.exists(fn) or not ro.save_frames):
        state.log(f"  [readout] RESUMED {len(frames)} frame(s) ({lg.mode})")
        state.report['readout'] = {'resumed': True, **hit}
        return
    plan = state.cache.get('leg_plan')
    if plan is None:
        raise PipelineError("readout: no leg plan in this process and none "
                            "cached; run the 'leg' stage.")
    full = state.cache.get('summed_full')
    if full is None:
        full = load_summed(state).full_field()
        state.cache['summed_full'] = full
    ramb = _resolve_ram_budget(lg.ram_budget)
    t_stage = time.perf_counter()
    out: Dict[str, np.ndarray] = {}
    rows: Dict[str, Any] = {}
    for k in frames:
        p = plan['frames'][k]
        t0 = time.perf_counter()
        kw = dict(dx_out=float(ro.dx_out), N_out=int(ro.n_out),
                  tilt=(0.0, 0.0), centre=tuple(p['centre']),
                  centre_out=tuple(p['centre_out']),
                  N_fine=int(p['N_fine']),
                  window_factor=float(p['window_factor']),
                  on_ram_cap=lg.on_ram_cap, on_replica=lg.on_replica,
                  on_readout_window=lg.on_readout_window,
                  on_n_fine_cap=lg.on_n_fine_cap)
        if ramb is not None:
            kw['ram_budget'] = ramb
        F = carrier_referenced_exact_focus_readout(
            full, float(plan['R_c']), float(lg.distance),
            float(spec.wavelength), float(plan['dx_c']), **kw)
        t_leg = time.perf_counter() - t0
        row: Dict[str, Any] = {'t_leg': t_leg}
        if ro.metrics:
            row['spot'] = spot(F, ro.dx_out, ro.ee_radii)
            fr = _chain_paths(state, k)[2]
            if os.path.exists(fr):
                ref = np.load(fr)
                row['vs_reference'] = compare(ref, F)
                pr = spot(ref, ro.dx_out, ro.ee_radii)
                row['reference_spot'] = pr
                row['power_ratio'] = (row['spot']['power'] / pr['power']
                                      if pr['power'] else float('nan'))
                del ref
        rows[k] = row
        if ro.save_frames:
            out[f'f_{k}'] = F
        s = row.get('spot', {})
        state.log(f"    {k:>8s}  {t_leg:6.1f}s  FWHM "
                  f"{s.get('fwhm', float('nan')) * 1e6:6.3f} um  EE3 "
                  f"{s.get('ee3', float('nan')) * 100:8.4f}  power "
                  f"{s.get('power', float('nan')):.7e}"
                  + (f"  P/P_ref {row['power_ratio']:.7f}  relL2 "
                     f"{row['vs_reference']['rel_l2']:.4e}  piston "
                     f"{row['vs_reference']['piston']:+.3e}"
                     if 'vs_reference' in row else ''))
        del F
    wall = time.perf_counter() - t_stage
    os.makedirs(out_dir, exist_ok=True)
    if ro.save_frames:
        np.savez_compressed(fn, **out)
    payload = {'tag': tag, 'mode': lg.mode, 'frames': frames,
               'included': state.cache.get('agg_include', []),
               'rows': rows, 'wall': wall,
               'peak_wset_gb': A.peak_rss_gb()[0],
               'peak_commit_gb': A.peak_rss_gb()[1],
               'admissibility': state.report.get('admissibility', {})}
    A.write_json(fj, payload, key)
    state.manifest.record('readout', digest, wall=wall, mode=lg.mode,
                          n_frames=len(frames))
    state.log(f"  [readout] {len(frames)} frame(s) in {wall:.1f}s")
    state.report['readout'] = {'resumed': False, **payload}


_STAGE_FNS = {'decompose': stage_decompose, 'chains': stage_chains,
              'aggregate': stage_aggregate, 'leg': stage_leg,
              'readout': stage_readout}


# ---------------------------------------------------------------------------
# the engine
# ---------------------------------------------------------------------------
def run(spec: PipelineSpec, *, from_stage: Optional[str] = None,
        through: Optional[str] = None, only: Optional[Sequence[str]] = None,
        log=print) -> RunState:
    """Run the pipeline, resuming every stage whose checkpoint still matches.

    ``from_stage`` forces a RECOMPUTE of that stage and everything after it
    (upstream stages are still resumed -- that is the point of a stage
    boundary).  ``through`` stops after a stage.  ``only`` runs exactly the
    named stages, which is how a long chains stage is farmed out separately
    from the aggregation that consumes it.

    Stages before the first one to be RUN are still visited, because a
    downstream stage's key depends on the upstream digest: skipping a stage
    entirely would mean not knowing what its output was keyed on, and a
    resume that cannot verify its inputs is a resume that can serve someone
    else's field.
    """
    if from_stage is not None and from_stage not in STAGES:
        raise SpecError(f"from_stage={from_stage!r} is not one of {STAGES}.")
    if through is not None and through not in STAGES:
        raise SpecError(f"through={through!r} is not one of {STAGES}.")
    if only is not None:
        bad = [s for s in only if s not in STAGES]
        if bad:
            raise SpecError(f"only={bad!r} are not stages; known {STAGES}.")
    workdir = os.path.abspath(spec.workdir)
    os.makedirs(workdir, exist_ok=True)
    state = RunState(spec=spec, workdir=workdir,
                     manifest=A.Manifest(os.path.join(workdir,
                                                      '_manifest.json')),
                     log=log)
    log('')
    log('=' * 74)
    log(f"PIPELINE {spec.name}")
    log('=' * 74)
    import lumenairy
    log(f"  library   {lumenairy.__version__} @ "
        f"{os.path.dirname(os.path.abspath(lumenairy.__file__))}")
    log(f"  lib sha   {A.lumenairy_source_sha()[:16]}   pipeline sha "
        f"{A.pipeline_source_sha()[:16]}")
    log(f"  workdir   {workdir}")
    log(f"  decompose {spec.decompose.kind!r}   chain {spec.chain.kind!r} "
        f"-> {spec.chain.plane!r}   leg {spec.leg.mode!r}")
    state.report['admissibility'] = admissibility_banner(spec, log=log)
    A.write_json(os.path.join(workdir, 'spec.json'), spec.to_dict(),
                 {'kind': 'spec-of-record'})

    i_from = STAGES.index(from_stage) if from_stage else len(STAGES)
    i_through = STAGES.index(through) if through else len(STAGES) - 1
    t_run = time.perf_counter()
    for i, stage in enumerate(STAGES):
        if i > i_through:
            break
        if only is not None and stage not in only:
            # Still VISIT it (resume-only) so downstream keys are known.
            _STAGE_FNS[stage](state, force=False)
            continue
        _STAGE_FNS[stage](state, force=(i >= i_from))
    state.report['wall'] = time.perf_counter() - t_run
    pk, pc = A.peak_rss_gb()
    state.report['peak_wset_gb'], state.report['peak_commit_gb'] = pk, pc
    state.manifest.doc['run'] = {'wall': state.report['wall'],
                                 'peak_wset_gb': pk, 'peak_commit_gb': pc,
                                 'name': spec.name}
    state.manifest.save()
    log(f"  TOTAL {state.report['wall']:.1f}s   peak working set {pk:.2f} GB "
        f"(peak commit {pc:.2f} GB)")
    return state
