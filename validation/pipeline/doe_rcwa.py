# validation/pipeline/doe_rcwa.py -- the RIGOROUS design-121 DOE decomposer.
#
# One plug-in: ``design121_doe_rcwa``.  It is ``design121_doe`` with ONE thing
# changed -- where the 32 complex order amplitudes come from.  The scalar
# decomposer takes them from the DFT of the Dammann cell's ideal complex
# transmittance (the thin-element assumption: no material, no relief, no angle,
# no polarization).  This one takes them from a cached ANGLE x POLARIZATION
# RCWA sweep of the reconstructed etched cell, built by
# ``validation/repro_traced_carrier_121/doe_rcwa_table.py``.
#
# Everything else -- the launch chain to the DOE plane, the order tilts, the
# frame centres, the context handed to the chain runner -- is IDENTICAL and is
# deliberately computed the same way, because the point of this decomposer is
# to be A/B-able against the scalar one with exactly one variable moving.
#
# WHY THE TABLE BUILDER IS NOT IN THIS PACKAGE.  Every ``*.py`` under
# ``validation/pipeline`` is content-hashed into every artifact key
# (``artifacts.pipeline_source_sha``), so a module that is EDITED during a
# campaign orphans every checkpoint in every workdir.  The table builder is an
# instrument under active development and its natural neighbour is the SCALAR
# table it replaces -- ``_d121_common.order_table`` -- which lives in
# ``validation/repro_traced_carrier_121/``.  It is therefore there, and this
# module is the thin, stable adapter.  (This file's own existence re-keys the
# package once, which is unavoidable for any new plug-in and is why it is
# small.)
#
# cp1252-safe ASCII only.
"""The rigorous (RCWA) design-121 DOE decomposer.

THE AVERAGING CHOICE -- READ THIS BEFORE USING THE WEIGHTS
==========================================================
The pipeline's beam basis carries ONE complex weight per order.  The RCWA
table carries an amplitude per order PER INCIDENT ANGLE and PER POLARIZATION.
Something has to collapse, and the collapse is a modelling decision:

* The weight used here is the BEAM-WEIGHTED COHERENT MEAN over the incident
  angular spectrum, then the unpolarized mean over the two incident linear
  polarizations:

      w_m = < a_m(theta, phi) >_beam, averaged over pol

  COHERENT because the pipeline's beams are summed COHERENTLY: the quantity
  that must survive the collapse is the FIELD each order contributes, not its
  power.  Where the phase varies across the cone the coherent mean loses power
  -- and that loss is real decoherence, not a modelling artifact.
* The incoherent (power-preserving) alternative
  ``sqrt(<|a|^2>)`` is computed alongside and both are recorded in the beam
  payload, together with ``coherence = |coherent|^2 / incoherent^2``, which
  MEASURES how much the choice mattered.  A coherence of 1 means the two
  collapses agree and the choice is immaterial.
* FULL ANGLE-RESOLVED DECOMPOSITION -- one beam per (order, angle node), so
  nothing collapses at all -- is a STATED FUTURE REFINEMENT.  It is not
  implementable against today's beam basis, which has no angle: ``Beam``
  carries a weight, a frame centre and a payload, and the ``traced`` runner
  launches ONE congruence per beam from ``payload['tilt']``.  Adding it means
  adding an angle to the basis and K x A chains, which is a pipeline change,
  not a decomposer change.

The beam angular weight defaults to the MEASURED angular content of design
121's own field at the DOE plane (rotationally symmetric, RMS half-angle
4.75e-05 rad; see the build note).  It is a parameter.

WHAT IS AND IS NOT INHERITED FROM THE SCALAR DECOMPOSER
=======================================================
Inherited verbatim, so an A/B moves one variable: the chain-A launch, the
group split, ``period``, the order tilts ``(m, n) lambda / period``, the
library's own chief-ray frame centres snapped to the output lattice, the exact
skew-trace diagnostic, and every context key the ``traced`` runner consumes.
Replaced: the complex weight, and the beam payload gains an ``rcwa`` record.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, List

import numpy as np

from .sources import Beam, DecomposeResult, order_tag, register_decomposer
from .spec import PipelineSpec, SpecError

#: Measured RMS half-angle of design 121's angular spectrum at the DOE plane
#: (both the envelope and the full field read 4.7e-05 rad; the DOE sits in
#: near-collimated space, R = 703.6 m over a 13.7 mm support).
D121_THETA_RMS = 4.75e-05

#: Default angular half-width of the sweep: 10x the RMS half-angle, which is
#: 1.3x the measured 99.999 %-enclosed half-angle (3.75e-04 rad).
D121_THETA_MAX = 5.0e-04


def _repro_dir():
    return os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..',
        'repro_traced_carrier_121'))


def _import_table():
    """The table builder ONLY.

    Deliberately separate from :func:`_import_d121`: importing
    ``_d121_common`` READS THE DESIGN-STUDY RUNNER AT IMPORT TIME (it parses
    ``_NEW_GLASSES`` and registers the Sellmeier coefficients), so it raises
    ``FileNotFoundError`` on any machine without the LOCAL-ONLY design tree.
    ``rcwa_weights`` needs none of that -- it needs a structure and a period --
    so it must not drag it in.  MEASURED: the WSL parity run of
    ``tests/unit/test_doe_rcwa.py`` failed 4 tests on exactly this before the
    split, while Windows -- where the design tree exists -- passed all 41.
    That is the whole reason the parity mount is run."""
    repro = _repro_dir()
    if repro not in sys.path:
        sys.path.insert(0, repro)
    import doe_rcwa_table as DT  # noqa: E402
    return DT


def _import_d121():
    """The design-121 module.  LOCAL-ONLY: needs the ``.zmx`` and the runner."""
    repro = _repro_dir()
    if repro not in sys.path:
        sys.path.insert(0, repro)
    import _d121_common as C  # noqa: E402
    return C


def rcwa_weights(params, wavelength, period, *, log=None, structure=None):
    """The 32 (or selected) RCWA order weights, from the cached table.

    Returns ``(orders, weights, record, per_order)``.  ``record`` and
    ``per_order`` are JSON-safe and are what the beams carry as provenance.
    Split out from the decomposer so a study can read the same weights without
    building a spec, and so ``structure`` can be INJECTED -- the tests drive a
    small cell through this function, which is the only way to exercise it
    without the design's 128-pixel Dammann cell.
    """
    DT = _import_table()
    p = dict(params or {})
    struct = structure if structure is not None else DT.design121_structure(
        cell_pixels=int(p.get('cell_pixels', 128)),
        n_doe=float(p.get('n_doe', DT.N_FUSED_SILICA_1310)),
        relief_sign=int(p.get('relief_sign', 1)),
        wavelength=float(wavelength), period=float(period),
        cache_dir=_repro_dir())
    orders = DT.design_order_set(struct)

    # DEFAULT 6 IS THE CEILING, NOT A CHOICE.  The design-121 cell is a
    # 113.76 um (86.8 wavelength) period with sub-wavelength features, and the
    # solver's own energy guard -- whose message names "very large period, low
    # index contrast" -- refuses every truncation from 7 upward on BOTH
    # mountings and in every formulation / truncation set tried.  6 is the
    # highest rung that solves.  It is NOT converged; the ladder is in
    # docs/audits/BUILD_RCWA_DOE_TABLE_2026_08_12.md and the residual
    # rung-to-rung movement is the dominant uncertainty on every weight here.
    n_orders = int(p.get('n_orders', 6))
    # THE DEFAULT ANGLE GRID IS ONE POINT, AND THAT IS A MEASUREMENT.
    # The angle-resolved sweep of record (n_orders = 4, the highest truncation
    # that solves CLEANLY at every node of a 3 x 8 Chebyshev x uniform grid over
    # the beam's cone) measured the per-order efficiency varying by 4.5e-04
    # RELATIVE and the per-order phase by 2.4e-04 rad across the whole cone --
    # four decades below the truncation uncertainty on the same numbers.  The
    # DOE sits in near-collimated space (R = 703.6 m at the DOE plane), so its
    # cone is 1.25e-04 rad at 99.9 % enclosed; there is simply no angular
    # content for the table to resolve.
    # Collapsing the grid is therefore not an approximation being tolerated,
    # it is the removal of a confound: at n_orders = 6 (the highest rung that
    # solves at all) 5 of 17 angle nodes fall back to a LOWER truncation under
    # the solver's stability guard, so an angle average there mixes
    # truncations and reports 2.6 of apparent angular spread that is entirely
    # numerical.  One clean solve at the highest truncation beats a
    # contaminated average.  Raise n_theta / n_phi to reproduce the sweep.
    theta_max = float(p.get('theta_max', D121_THETA_MAX))
    n_theta = int(p.get('n_theta', 1))
    n_phi = int(p.get('n_phi', 1))
    formulation = str(p.get('formulation', 'laurent'))
    truncation = str(p.get('truncation', 'rectangular'))
    mount = str(p.get('mount', 'relief_first'))
    theta_rms = float(p.get('beam_theta_rms', D121_THETA_RMS))

    mode = str(p.get('averaging', 'coherent'))
    if mode not in ('coherent', 'incoherent'):
        # validated BEFORE the sweep: a config typo must not cost an RCWA run
        raise SpecError(
            f"decompose.design121_doe_rcwa: averaging must be 'coherent' or "
            f"'incoherent', got {mode!r}.  See this module's docstring -- the "
            f"pipeline sums beams coherently, so 'coherent' is the default and "
            f"'incoherent' is the power-preserving control.")

    th, ph, quad = DT.angle_grid(theta_max, n_theta, n_phi)
    beam_w = DT.gaussian_beam_angular_weight(th, ph, theta_rms)
    path = DT.table_path(DT.table_key(
        struct, orders, n_orders, th, ph, formulation=formulation,
        truncation=truncation, mount=mount,
        cell_upsample=p.get('cell_upsample'))[1], _repro_dir())
    if not os.path.exists(path) and not bool(p.get('build_if_missing', True)):
        raise SpecError(
            f"decompose.design121_doe_rcwa: the RCWA order table "
            f"{os.path.basename(path)} has not been built and "
            f"build_if_missing is false.  Build it with\n"
            f"    python {os.path.join(_repro_dir(), 'doe_rcwa_table.py')} "
            f"sweep\nor set decompose.params.build_if_missing = true and "
            f"accept the sweep cost inside the run.")
    table = DT.build_table(
        struct, orders, n_orders, th, ph, formulation=formulation,
        truncation=truncation, cell_upsample=p.get('cell_upsample'),
        mount=mount, quad_weight=quad, cache=True, cache_dir=_repro_dir(),
        max_workers=p.get('max_workers'),
        blas_per_worker=int(p.get('blas_per_worker', 1)),
        log=(log or (lambda *a, **k: None)))

    avg = DT.beam_weighted_amplitudes(table, beam_weight=beam_w)
    w = (avg['amp_coherent'] if mode == 'coherent'
         else avg['amp_incoherent'].astype(complex))

    eta = np.abs(np.asarray(table['eta_T'])).mean(axis=(0, 1, 2))
    xpol = np.asarray(table['xpol']).max(axis=(0, 1, 2))
    rec = {'digest': str(table['digest']),
           'table': os.path.basename(str(table['path'])),
           'n_orders': n_orders, 'formulation': formulation,
           'truncation': truncation, 'mount': mount, 'averaging': mode,
           'theta_max': theta_max, 'n_theta': n_theta, 'n_phi': n_phi,
           'beam_theta_rms': theta_rms,
           'n_doe': float(struct.n_doe), 'relief_sign': int(struct.relief_sign),
           'relief_total_m': float(struct.relief_total),
           'dz_m': float(struct.dz), 'cell_pixels': int(struct.n_cell),
           'worst_closure': float(table['meta']['worst_closure']),
           'n_stabilized': int(table['meta'].get('n_stabilized', 0)),
           'flat_face_fresnel': list(table['meta']['flat_face_fresnel']),
           'sum_abs_w_sq': float(np.sum(np.abs(w) ** 2))}
    per_order = [{'eta_mean': float(eta[i]), 'xpol_max': float(xpol[i]),
                  'coherence': float(avg['coherence'][i]),
                  'amp_incoherent': float(avg['amp_incoherent'][i])}
                 for i in range(len(w))]
    return orders, np.asarray(w), rec, per_order


# ===========================================================================
# the decomposer
# ===========================================================================
def decompose_design121_doe_rcwa(params, spec: PipelineSpec) -> DecomposeResult:
    """Design 121's DOE decomposition with RIGOROUS (RCWA) order amplitudes.

    params
    ------
    orders : 'all' | [[m, n], ...]
        Which orders to carry.  'all' is the design's own 4x8 table.
    cell_pixels : int
        Dammann cell resolution (default 128 -- the design's own, and the one
        the SCALAR decomposer uses, so the two are on one geometry).
    n_doe : float
        Refractive index of the DOE relief material.  DEFAULT 1.446804 (fused
        silica at 1.31 um) is an ASSUMPTION -- the design tree records no
        material.  See ``doe_rcwa_table``'s STRUCTURE ASSUMPTIONS.
    relief_sign : {+1, -1}
        Pillars (+1, default) or pits (-1).
    n_orders : int
        RCWA truncation per side per axis.  DEFAULT 6, AND IT IS A CEILING
        RATHER THAN A CHOICE: on the design-121 cell the solver's own energy
        guard refuses every truncation from 7 upward, on both mountings and in
        every formulation tried, so 6 is the highest rung that solves and it is
        NOT converged.  The ladder, and the rung-to-rung movement that is the
        dominant uncertainty on every weight this module returns, are in
        ``docs/audits/BUILD_RCWA_DOE_TABLE_2026_08_12.md``; the same statement
        sits at the default's own site (:154).
    theta_max, n_theta, n_phi : float, int, int
        The incident-angle grid: Chebyshev in polar angle over
        ``[0, theta_max]``, uniform in azimuth.
    beam_theta_rms : float
        RMS half-angle of the Gaussian angular weight used for the average.
    averaging : {'coherent', 'incoherent'}
        How the angle x polarization table collapses to one weight per order.
    mount : {'relief_first', 'substrate_first'}
        Which face of the plate the beam meets first.  NOT RECORDED in the
        design tree and NOT determined by the order tilts -- see assumption A4
        in ``doe_rcwa_table``.  Keyed.
    formulation, truncation, cell_upsample, max_workers, blas_per_worker,
    build_if_missing
        Passed to the table builder.
    chain_a : dict
        The launch chain to the DOE plane -- IDENTICAL to ``design121_doe``.
    """
    C = _import_d121()
    import lumenairy as la  # noqa: E402
    import lumenairy.propagators.carrier as CAR  # noqa: E402

    if abs(float(spec.wavelength) - C.LAM) > 1e-18:
        raise SpecError(
            f"decompose.design121_doe_rcwa: the spec is at wavelength "
            f"{spec.wavelength!r} m but design 121 is a {C.LAM!r} m design -- "
            f"its prescription, Dammann period, glass table AND the RCWA relief "
            f"depth (which is lambda/(8 (n-1)) per level) are all evaluated "
            f"there.  Refusing to decompose one design at another wavelength.")

    ca = dict(params.get('chain_a') or {})
    n_a = int(ca.get('n', 1024))
    dx0 = ca.get('dx0')
    rs = int(ca.get('ray_subsample', spec.chain.ray_subsample))
    nw = int(ca.get('n_workers', spec.chain.n_workers))
    leg_a = str(ca.get('final_leg', 'exact'))

    _pre, groups_post, _gap, period = C.geometry()
    orders, weights, rec, per_order = rcwa_weights(
        params, float(spec.wavelength), float(period))
    env_doe, R_doe, dx_doe, P_in = C.chain_a(n=n_a, dx0=dx0, rs=rs, nw=nw,
                                             final_leg=leg_a)

    table = [(int(orders[i, 0]), int(orders[i, 1]), complex(weights[i]),
              per_order[i]) for i in range(len(weights))]
    want = params.get('orders', 'all')
    if want != 'all':
        wset = [tuple(int(v) for v in o) for o in want]
        have = {(m, n) for m, n, _a, _d in table}
        miss = [o for o in wset if o not in have]
        if miss:
            raise SpecError(
                f"decompose.design121_doe_rcwa: order(s) {miss} are not in the "
                f"design's own {len(table)}-order table "
                f"{sorted((m, n) for m, n, _a, _d in table)}.")
        keep = set(wset)
        table = [t for t in table if (t[0], t[1]) in keep]

    # Frame centres: the LIBRARY's own paraxial chief-ray predictor snapped to
    # the output lattice, with the independent exact skew trace recorded
    # alongside -- VERBATIM from ``decompose_design121_doe``, deliberately, so
    # an A/B between the two decomposers moves the WEIGHTS and nothing else.
    from lumenairy.raytrace import make_ray, trace  # noqa: E402
    surfs = C.post_surfaces(groups_post)
    dxo = float(spec.readout.dx_out)
    beams: List[Beam] = []
    for i, (m, n, a, diag) in enumerate(table):
        L = float(m) * C.LAM / period
        M = float(n) * C.LAM / period
        car = la.TiltedCarrier(float(R_doe), L, M)
        x, y, _L, _M = CAR._chain_chief_ray_at_target(
            groups_post, C.LAM, car, C.TRAILING, 'pipeline.design121_doe_rcwa')
        tr = trace(make_ray(0.0, 0.0, L, M, wavelength=C.LAM), surfs, C.LAM)
        beams.append(Beam(
            key=order_tag(m, n), index=i, weight=a,
            frame_centre=(round(x / dxo) * dxo, round(y / dxo) * dxo),
            label=f"({m:+d},{n:+d})",
            payload={'order': [m, n], 'tilt': [L, M], 'R_doe': float(R_doe),
                     'chief_pred': [float(x), float(y)],
                     'chief_exact': [float(tr.image_rays.x[0]),
                                     float(tr.image_rays.y[0])],
                     'rcwa': {'digest': rec['digest'],
                              'n_orders': rec['n_orders'],
                              'averaging': rec['averaging'],
                              **{k: diag[k] for k in sorted(diag)}}}))
    ctx: Dict[str, Any] = {
        'groups_post': groups_post, 'env_doe': env_doe,
        'dx_doe': float(dx_doe), 'R_doe': float(R_doe), 'P_in': float(P_in),
        'period': float(period), 'trailing': float(C.TRAILING),
        'design': 'd121', 'doe_model': 'rcwa', 'rcwa': rec,
        'chain_a': {'n': n_a, 'ray_subsample': rs, 'n_workers': nw,
                    'final_leg': leg_a}}
    return DecomposeResult(beams=beams, context=ctx)


register_decomposer('design121_doe_rcwa', decompose_design121_doe_rcwa)
