"""GAP 2, INDEPENDENT -- what the INDEX CEILING actually reads, and the
false-positive hunt on healthy ABSORBING stacks.

THE STRUCTURAL FACT THIS PROBE IS BUILT ON.  ``bor_solve.solve`` passes the
ceiling ``(layers[0], layers[0]["q"][inc] / k0)`` and
``(layers[-1], layers[-1]["q"][out] / k0)``, and ``inc`` / ``out`` come from
``_physical_propagating(layers[0] or layers[-1], k0)``.  Both arguments are
therefore functions of ONE HALF-SPACE ALONE -- its ``m``, ``N``, ``Rbig``,
``k0`` and its own ``eps`` -- and of nothing in the cascade.  So the ceiling's
excess can be measured with a single ``build_layer`` call (part ``halfspace``,
thousands of rows for the price of a few cascades), and a refusal it issues is
a statement about the half-space's discretisation, not about the answer the
cascade returned.  Part ``fp`` is the consequence test: a stack whose middle
layer is the SAME uniform medium as its half-spaces scatters nothing (the exact
answer is ``R = 0``, ``T = 1`` per channel), so if the ceiling refuses it the
refusal cannot be about the cascade.

Parts: ``halfspace`` (the cheap large grid), ``census`` (the set-right /
set-wrong populations, full cascades in both bases), ``fp`` (healthy stacks).
"""
from __future__ import annotations

import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _vb3  # noqa: E402

K0 = _vb3.K0
IMS = (0.0, 1e-12, 1e-9, 1e-6, 1e-3, 1e-1)


def _half(m, N, rbl, eps, im, k0=K0):
    """One half-space: its returned channel set and the ceiling excess over
    it, exactly as ``_check_nodal_passivity`` would read them."""
    import lumenairy.elements.bor.bor_solve as bs
    Rbig = float(rbl) * 2.0 * np.pi / float(k0)
    prof = _vb3._lossy(_vb3.uni(eps), im, im != 0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        L = bs.build_layer(m, Rbig, N, prof, k0, basis="nodal")
        Ls = bs.build_layer(m, Rbig, N, prof, k0, basis="staggered")
    keep = np.where(bs._physical_propagating(L, k0))[0]
    qn = np.asarray(L["q"])[keep] / k0
    exc = bs._channel_index_excess(L, qn)
    n_max = float(np.real(np.sqrt(complex(L["eps_ceiling"]))))
    slack_fn = getattr(bs, "_index_ceiling_slack", None)
    slack = (slack_fn(L, n_max) if slack_fn is not None
             else bs._BOR_INDEX_CEILING_SLACK)
    keeps = np.where(bs._physical_propagating(Ls, k0))[0]
    return dict(m=m, N=N, rbl=rbl, eps=float(eps), im=im,
                n_channels=int(keep.size), n_channels_staggered=int(keeps.size),
                lossless=bool(bs._layer_is_lossless(L)),
                n_max=n_max, excess=exc, slack=float(slack),
                fires=bool(exc is not None and exc > slack),
                fires_base_slack=bool(exc is not None
                                      and exc > bs._BOR_INDEX_CEILING_SLACK),
                max_abs_im_qn=(float(np.max(np.abs(np.imag(qn))))
                               if qn.size else None))


def part_halfspace():
    t0 = time.time()
    rows = []
    for m in (0, 1, 2, 3):
        for N in (80, 120, 200):
            for rbl in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
                for eps in (2.25, 4.0):
                    for im in IMS:
                        r = _half(m, N, rbl, eps, im)
                        rows.append(r)
    fires = [r for r in rows if r["fires"]]
    lossless = [r for r in rows if r["im"] == 0.0]
    lossy = [r for r in rows if r["im"] != 0.0]
    #: the decisive pair: would this half-space have been refused on the
    #: round-2 tree?  There the ceiling was evaluated ONLY where the medium is
    #: lossless, so every lossy row was silent whatever its excess.
    newly = [r for r in lossy if r["fires"] and not r["lossless"]]
    #: rows where the LOSS ITSELF moved the decision (the lossless twin of the
    #: same geometry does not fire)
    key = lambda r: (r["m"], r["N"], r["rbl"], r["eps"])          # noqa: E731
    zero = {key(r): r for r in rows if r["im"] == 0.0}
    loss_flips = [dict(row=r, lossless_excess=zero[key(r)]["excess"])
                  for r in lossy
                  if r["fires"] and not zero[key(r)]["fires_base_slack"]]
    exc = [r["excess"] for r in rows if r["excess"] is not None]
    summary = dict(
        rows=len(rows), fires=len(fires),
        fires_frac=len(fires) / max(1, len(rows)),
        lossless_rows=len(lossless), lossy_rows=len(lossy),
        newly_armed_rows_that_fire=len(newly),
        rows_where_loss_alone_flips_the_decision=len(loss_flips),
        loss_flip_examples=loss_flips[:10],
        worst_excess=max(exc) if exc else None,
        best_negative_excess=max([e for e in exc if e <= 0] or [float("nan")]),
        closest_approach_from_below=max([e for e in exc if e <= 0]
                                        or [float("nan")]),
        mildest_positive=min([e for e in exc if e > 0] or [float("nan")]),
        seconds=time.time() - t0)
    for k in sorted(summary):
        if k != "loss_flip_examples":
            print(" ", k, summary[k])
    _vb3.dump("v2_halfspace", dict(rows=rows, summary=summary))


def _cascade_row(family, m, N, rbl, where, im):
    lay = _vb3.stack("nodal", family, m, N, rbl, im, where)
    twin = _vb3.stack("staggered", family, m, N, rbl, im, where)
    raw = _vb3.disarmed(lay)
    stg = _vb3.disarmed(twin)
    en = np.asarray(raw["energy"], float)
    et = np.asarray(stg["energy"], float)
    ei, ex = _vb3.ceiling_excess(lay, raw)
    si, sx = _vb3.ceiling_excess(twin, stg)
    v, det, nw = _vb3.armed(lay)
    set_right = (len(raw["inc"]) == len(stg["inc"])
                 and len(raw["out"]) == len(stg["out"]))
    return dict(family=family, m=m, N=N, rbl=rbl, where=where, im=im,
                n_inc=int(len(raw["inc"])), n_out=int(len(raw["out"])),
                t_inc=int(len(stg["inc"])), t_out=int(len(stg["out"])),
                set_right=bool(set_right), verdict=v, detector=det,
                warnings=nw,
                nodal_closure=(float(np.max(np.abs(en - 1.0)))
                               if en.size else None),
                twin_closure=(float(np.max(np.abs(et - 1.0)))
                              if et.size else None),
                exc_nodal=[ei, ex], exc_staggered=[si, sx])


def part_census():
    t0 = time.time()
    rows = []
    for family in ("para", "grate"):
        for m in (0, 1, 2, 3):
            for rbl in (0.5, 1.0, 2.0, 4.0):
                for where in ("inc", "both"):
                    for im in (1e-12, 1e-9, 1e-6, 1e-3, 1e-1):
                        r = _cascade_row(family, m, 120, rbl, where, im)
                        rows.append(r)
                        print("  %-6s m=%d rbl=%-4g %-4s im=%-8.3g "
                              "n=%d/%d t=%d/%d right=%-5s -> %s %s"
                              % (family, m, rbl, where, im, r["n_inc"],
                                 r["n_out"], r["t_inc"], r["t_out"],
                                 r["set_right"], r["verdict"], r["detector"]),
                              flush=True)
    def _exc(r, which):
        return [e for e in r[which] if e is not None]

    right = [r for r in rows if r["set_right"] and r["n_inc"]]
    wrong = [r for r in rows if not r["set_right"] and r["n_inc"]]
    stag = [e for r in rows for e in _exc(r, "exc_staggered")]
    right_exc = [e for r in right for e in _exc(r, "exc_nodal")]
    wrong_exc = [e for r in wrong for e in _exc(r, "exc_nodal")]
    wrong_fire = [r for r in wrong if r["verdict"] == "REFUSED"
                  and r["detector"] == "ceiling"]
    fp = [r for r in right if r["verdict"] == "REFUSED"
          and r["detector"] == "ceiling"]
    summary = dict(
        rows=len(rows), solves=2 * len(rows),
        set_right=len(right), set_wrong=len(wrong),
        nodal_set_right_worst_excess=max(right_exc) if right_exc else None,
        staggered_worst_excess=max(stag) if stag else None,
        set_wrong_fires=len(wrong_fire),
        set_wrong_mildest_positive=min([e for e in wrong_exc if e > 0]
                                       or [float("nan")]),
        false_positives_on_set_right=len(fp),
        undamaged_rows=len(right) + len(rows),
        seconds=time.time() - t0)
    for k in sorted(summary):
        print(" ", k, summary[k])
    _vb3.dump("v2_census", dict(rows=rows, summary=summary))


def part_fp():
    """THE FALSE-POSITIVE HUNT.

    Population A -- SCREENED healthy: a geometry joins it only if, at ZERO
    loss, the nodal cascade returns the same channel counts as its staggered
    twin AND its closure is inside the energy screen's own WARN edge
    (1e-06) AND the ceiling is silent.  A refusal of such a row carrying a loss
    is a false one by the round's own definition of damage.

    Population B -- TRIVIAL: middle layer identical to the half-spaces, so the
    stack scatters nothing and the exact answer is ``R = 0``, ``T = 1`` on
    every channel.  This population exists to separate "the cascade is damaged"
    from "the half-space's own mode list contains an over-ceiling entry", which
    the structural fact in the module docstring says are different claims.
    """
    import lumenairy.elements.bor.bor_solve as bs
    t0 = time.time()
    cands = [(f, m, N, rbl)
             for f in ("para", "grate", "core", "bilayer", "unif35", "unif23")
             for m in (0, 1, 2, 3)
             for N in (120,)
             for rbl in (0.5, 1.0, 2.0, 4.0)]
    healthy, screened = [], []
    for (f, m, N, rbl) in cands:
        lay = _vb3.stack("nodal", f, m, N, rbl, 0.0, "inc")
        twin = _vb3.stack("staggered", f, m, N, rbl, 0.0, "inc")
        raw = _vb3.disarmed(lay)
        stg = _vb3.disarmed(twin)
        en = np.asarray(raw["energy"], float)
        ei, ex = _vb3.ceiling_excess(lay, raw)
        ok = (len(raw["inc"]) == len(stg["inc"])
              and len(raw["out"]) == len(stg["out"])
              and en.size > 0
              and float(np.max(np.abs(en - 1.0)))
              <= bs._BOR_NODAL_SUPERUNITY_WARN
              and (ei is None or ei <= bs._BOR_INDEX_CEILING_SLACK)
              and (ex is None or ex <= bs._BOR_INDEX_CEILING_SLACK))
        screened.append(dict(family=f, m=m, N=N, rbl=rbl, healthy=bool(ok),
                             closure=(float(np.max(np.abs(en - 1.0)))
                                      if en.size else None),
                             n=[int(len(raw["inc"])), int(len(raw["out"]))],
                             t=[int(len(stg["inc"])), int(len(stg["out"]))],
                             exc=[ei, ex]))
        if ok:
            healthy.append((f, m, N, rbl))
    print("  screened %d candidates -> %d healthy" % (len(cands), len(healthy)),
          flush=True)
    rows = []
    for (f, m, N, rbl) in healthy:
        for where in ("inc", "exit", "both"):
            for im in (1e-12, 3e-12, 1e-9, 1e-6, 1e-5, 1e-4):
                lay = _vb3.stack("nodal", f, m, N, rbl, im, where)
                v, det, nw = _vb3.armed(lay)
                raw = _vb3.disarmed(lay)
                en = np.asarray(raw["energy"], float)
                ei, ex = _vb3.ceiling_excess(lay, raw)
                rows.append(dict(family=f, m=m, N=N, rbl=rbl, where=where,
                                 im=im, verdict=v, detector=det, warnings=nw,
                                 n_channels=int(len(raw["inc"])),
                                 closure=(float(np.max(np.abs(en - 1.0)))
                                          if en.size else None),
                                 exc=[ei, ex]))
                if v != "returned" or nw:
                    print("  FP? %-8s m=%d rbl=%-4g %-5s im=%-8.3g -> %s %s "
                          "nw=%d" % (f, m, rbl, where, im, v, det, nw),
                          flush=True)
    # Population B -- the trivial stack
    triv = []
    for m in (0, 1, 2, 3):
        for N in (80, 120, 200):
            for rbl in (0.5, 1.0, 2.0, 4.0, 8.0):
                for im in (0.0, 1e-9, 1e-6):
                    Rbig = float(rbl) * 2.0 * np.pi / K0
                    prof = _vb3._lossy(_vb3.uni(_vb3.EPS_HALF), im, im != 0.0)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        lay = [bs.build_layer(m, Rbig, N, prof, K0,
                                              basis="nodal"),
                               bs.build_layer(m, Rbig, N, prof, K0,
                                              basis="nodal", thickness=0.42),
                               bs.build_layer(m, Rbig, N, prof, K0,
                                              basis="nodal")]
                    v, det, nw = _vb3.armed(lay)
                    raw = _vb3.disarmed(lay)
                    en = np.asarray(raw["energy"], float)
                    ei, ex = _vb3.ceiling_excess(lay, raw)
                    triv.append(dict(m=m, N=N, rbl=rbl, im=im, verdict=v,
                                     detector=det, warnings=nw,
                                     n_channels=int(len(raw["inc"])),
                                     closure=(float(np.max(np.abs(en - 1.0)))
                                              if en.size else None),
                                     exc=[ei, ex]))
                    print("  TRIV m=%d N=%d rbl=%-4g im=%-8.3g n=%d "
                          "closure=%-10s exc=%-11s -> %s %s"
                          % (m, N, rbl, im, triv[-1]["n_channels"],
                             "%.3e" % triv[-1]["closure"]
                             if triv[-1]["closure"] is not None else "-",
                             "%.3e" % ei if ei is not None else "-", v, det),
                          flush=True)
    bad = [r for r in rows if r["verdict"] != "returned" or r["warnings"]]
    triv_refused = [r for r in triv if r["verdict"] != "returned"]
    triv_refused_lossy = [r for r in triv_refused if r["im"] != 0.0]
    summary = dict(
        candidates=len(cands), healthy=len(healthy),
        healthy_loss_rows=len(rows),
        healthy_rows_refused_or_warned=len(bad), examples=bad[:10],
        trivial_rows=len(triv),
        trivial_refused=len(triv_refused),
        trivial_refused_lossy=len(triv_refused_lossy),
        trivial_worst_closure=max([r["closure"] for r in triv
                                   if r["closure"] is not None] or [None]),
        seconds=time.time() - t0)
    for k in sorted(summary):
        if k != "examples":
            print(" ", k, summary[k])
    _vb3.dump("v2_fp", dict(screened=screened, rows=rows, trivial=triv,
                            summary=summary))


PARTS = {"halfspace": part_halfspace, "census": part_census, "fp": part_fp}

if __name__ == "__main__":
    a = _vb3.arm()
    _vb3.require_tree(a)
    print("ARM", a["build"], a["loaded_kernel"], "t%s" % a["threads"],
          a["lumenairy_file"], flush=True)
    for name in (sys.argv[1:] or list(PARTS)):
        print("== PART", name, flush=True)
        PARTS[name]()
