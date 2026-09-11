"""GAP 2, INDEPENDENT -- the split of ``_check_nodal_passivity`` into a media
gate (both detectors) and an incidence-lossless gate (the energy detector
alone), measured on fixtures written for this verification.

Sub-commands (``python v1_gap2.py <part>``):

``ladder``   the decision table: 6 geometries x {loss on layers[0], on the
             EXIT half-space, on BOTH} x a 13-rung ``Im/Re`` ladder from 0 to
             1e-1, each solved in BOTH bases with the guard disarmed and then
             once armed.  PRE/POST by running the same probe against the
             1ac6de7e tree with ``LUM_PROBE_TAG=BASE``.
``gain``     gain (``Im eps < 0``) must disarm BOTH detectors, before and after.
``switch``   ``BOR_NODAL_PASSIVITY_GUARD = False`` must return the SAME BYTES
             the round-2 tree returns, on every row of the ladder.

All three write one JSON per arm.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _vb3  # noqa: E402

K0 = _vb3.K0

#: 13 rungs, ``Im(eps)/Re(eps)`` on the chosen layer(s).  0 is the lossless
#: control; 1e-12 straddles ``_BOR_LOSSLESS_REL_IM`` (the predicate's own
#: boundary); 1e-1 is past the channel gate's ``|Im qn|`` bar.
RUNGS = (0.0, 3e-13, 1e-12, 3e-12, 1e-11, 1e-9, 1e-8, 1e-6, 1e-5,
         1e-4, 1e-3, 1e-2, 1e-1)

#: (family, m, N, Rbig/lambda) -- six geometries, four families, m 0..3,
#: Rbig from 1 to 16 vacuum wavelengths.
GEOMS = (("grate", 1, 120, 2.0),
         ("para", 0, 120, 4.0),
         ("core", 2, 120, 1.0),
         ("bilayer", 3, 120, 2.0),
         ("grate", 1, 200, 8.0),
         ("para", 2, 120, 16.0))

WHERES = ("inc", "exit", "both")


def _row(family, m, N, rbl, where, im):
    lay = _vb3.stack("nodal", family, m, N, rbl, im, where)
    twin = _vb3.stack("staggered", family, m, N, rbl, im, where)
    raw = _vb3.disarmed(lay)
    stg = _vb3.disarmed(twin)
    en = np.asarray(raw["energy"], float)
    et = np.asarray(stg["energy"], float)
    v, det, nw = _vb3.armed(lay)
    ei, ex = _vb3.ceiling_excess(lay, raw)
    set_right = (len(raw["inc"]) == len(stg["inc"])
                 and len(raw["out"]) == len(stg["out"]))
    return dict(family=family, m=m, N=N, rbl=rbl, where=where, im=im,
                verdict=v, detector=det, warnings=nw,
                n_inc=int(len(raw["inc"])), n_out=int(len(raw["out"])),
                t_inc=int(len(stg["inc"])), t_out=int(len(stg["out"])),
                set_right=bool(set_right),
                nodal_max=(float(np.max(en)) if en.size else None),
                nodal_min=(float(np.min(en)) if en.size else None),
                twin_excess=(float(np.max(np.abs(et - 1.0)))
                             if et.size else None),
                exc_inc=ei, exc_exit=ex,
                hash=_vb3.answer_hash(raw))


def part_ladder():
    rows = []
    t0 = time.time()
    for (family, m, N, rbl) in GEOMS:
        for where in WHERES:
            for im in RUNGS:
                r = _row(family, m, N, rbl, where, im)
                rows.append(r)
                print("  %-8s m=%d N=%d rbl=%-4g %-5s im=%-8.3g nch=%d/%d "
                      "twin=%d/%d max=%-11s texc=%-10s ei=%-11s ex=%-11s "
                      "-> %s %s"
                      % (family, m, N, rbl, where, im, r["n_inc"], r["n_out"],
                         r["t_inc"], r["t_out"],
                         "%.6g" % r["nodal_max"] if r["nodal_max"] is not None
                         else "-",
                         "%.3e" % r["twin_excess"]
                         if r["twin_excess"] is not None else "-",
                         "%.3e" % r["exc_inc"] if r["exc_inc"] is not None
                         else "-",
                         "%.3e" % r["exc_exit"] if r["exc_exit"] is not None
                         else "-", r["verdict"], r["detector"]), flush=True)
    by_where = {}
    for w in WHERES:
        sub = [r for r in rows if r["where"] == w]
        det = {}
        for r in sub:
            if r["verdict"] == "REFUSED":
                det[r["detector"]] = det.get(r["detector"], 0) + 1
        by_where[w] = dict(n=len(sub),
                           refused=sum(r["verdict"] == "REFUSED" for r in sub),
                           by_detector=det,
                           warned=sum(r["warnings"] > 0 for r in sub))
    per_geom = {}
    for (family, m, N, rbl) in GEOMS:
        key = "%s_m%d_N%d_rbl%g" % (family, m, N, rbl)
        for w in WHERES:
            sub = [r for r in rows if r["where"] == w and r["family"] == family
                   and r["m"] == m and r["N"] == N and r["rbl"] == rbl]
            det = {}
            for r in sub:
                if r["verdict"] == "REFUSED":
                    det[r["detector"]] = det.get(r["detector"], 0) + 1
            per_geom["%s/%s" % (key, w)] = dict(
                n=len(sub),
                refused=sum(r["verdict"] == "REFUSED" for r in sub),
                by_detector=det)
    summary = dict(rows=len(rows), geoms=len(GEOMS), rungs=len(RUNGS),
                   nodal_solves=len(rows), staggered_twins=len(rows),
                   by_where=by_where, per_geom=per_geom,
                   seconds=time.time() - t0)
    print(" SUMMARY", {k: summary[k] for k in ("rows", "by_where")})
    _vb3.dump("v1_ladder", dict(rows=rows, summary=summary))


def part_gain():
    """GAIN (``Im eps < 0``) must leave BOTH detectors off, before and after."""
    rows = []
    for (family, m, N, rbl) in GEOMS[:4]:
        for where in ("inc", "exit", "both", "mid", "all"):
            for im in (-1e-14, -1e-12, -3e-12, -1e-9, -1e-6, -1e-3):
                lay = _vb3.stack("nodal", family, m, N, rbl, im, where)
                v, det, nw = _vb3.armed(lay)
                raw = _vb3.disarmed(lay)
                en = np.asarray(raw["energy"], float)
                ei, ex = _vb3.ceiling_excess(lay, raw)
                rows.append(dict(family=family, m=m, N=N, rbl=rbl,
                                 where=where, im=im, verdict=v, detector=det,
                                 warnings=nw,
                                 n_inc=int(len(raw["inc"])),
                                 nodal_max=(float(np.max(en)) if en.size
                                            else None),
                                 exc_inc=ei, exc_exit=ex))
                print("  GAIN %-8s m=%d %-5s im=%-9.3g max=%-11s ei=%-11s "
                      "ex=%-11s -> %s %s"
                      % (family, m, where, im,
                         "%.6g" % rows[-1]["nodal_max"]
                         if rows[-1]["nodal_max"] is not None else "-",
                         "%.3e" % ei if ei is not None else "-",
                         "%.3e" % ex if ex is not None else "-", v, det),
                      flush=True)
    would_fire = [r for r in rows
                  if (r["exc_inc"] is not None and r["exc_inc"] > 5e-10)
                  or (r["exc_exit"] is not None and r["exc_exit"] > 5e-10)
                  or (r["nodal_max"] is not None and r["nodal_max"] > 1.001)]
    summary = dict(rows=len(rows),
                   refused=sum(r["verdict"] == "REFUSED" for r in rows),
                   warned=sum(r["warnings"] > 0 for r in rows),
                   rows_a_disarmed_detector_would_have_fired_on=len(would_fire))
    print(" SUMMARY", summary)
    _vb3.dump("v1_gain", dict(rows=rows, summary=summary))


def part_switch():
    """The escape hatch must restore the PRE-round behaviour exactly: with
    ``BOR_NODAL_PASSIVITY_GUARD = False`` the answer is the same BYTES, and no
    refusal and no warning is raised anywhere on the ladder."""
    import warnings as _w

    import lumenairy.elements.bor.bor_solve as bs
    rows = []
    for (family, m, N, rbl) in GEOMS[:4]:
        for where in WHERES:
            for im in (0.0, 3e-12, 1e-9, 1e-6, 1e-3, 1e-1):
                lay = _vb3.stack("nodal", family, m, N, rbl, im, where)
                prev = bs.BOR_NODAL_PASSIVITY_GUARD
                bs.BOR_NODAL_PASSIVITY_GUARD = False
                try:
                    with _w.catch_warnings(record=True) as caught:
                        _w.simplefilter("always")
                        res = bs.solve(lay, K0)
                    nw = len([x for x in caught if "R + T" in str(x.message)])
                    raised = ""
                except bs.BORNodalPassivityError as exc:   # must never happen
                    res, nw, raised = None, 0, str(exc)[:120]
                finally:
                    bs.BOR_NODAL_PASSIVITY_GUARD = prev
                rows.append(dict(family=family, m=m, N=N, rbl=rbl,
                                 where=where, im=im,
                                 hash=(_vb3.answer_hash(res) if res is not None
                                       else None),
                                 warnings=nw, raised=raised))
                print("  SWITCH %-8s m=%d %-5s im=%-9.3g hash=%s nw=%d %s"
                      % (family, m, where, im, rows[-1]["hash"][:16]
                         if rows[-1]["hash"] else "RAISED", nw, raised),
                      flush=True)
    summary = dict(rows=len(rows),
                   raised=sum(bool(r["raised"]) for r in rows),
                   warned=sum(r["warnings"] > 0 for r in rows))
    print(" SUMMARY", summary)
    _vb3.dump("v1_switch", dict(rows=rows, summary=summary))


PARTS = {"ladder": part_ladder, "gain": part_gain, "switch": part_switch}

if __name__ == "__main__":
    a = _vb3.arm()
    _vb3.require_tree(a)
    print("ARM", a["build"], a["loaded_kernel"], "t%s" % a["threads"],
          a["lumenairy_file"], flush=True)
    for name in (sys.argv[1:] or list(PARTS)):
        print("== PART", name, flush=True)
        PARTS[name]()
