"""V6 -- the two open items round 3 carries forward, proved out on the
verifier's OWN devices.

A verification is bidirectional: the claimed successes have to be refuted and
the claimed FAILURES have to be reproduced, on devices the fix did not use.

**R3-B (was D-1)** -- ``_cross_layer_sliver`` measures the own-scale as the
GLOBAL minimum wall spacing over every layer, so ONE sliver-thin feature that
a single layer legitimately owns lowers that scale for the whole stack and the
cross-layer screen goes silent.  Arm A is the same stack without the liner
(the screen fires, the solve is refused); arm B adds a liner ONE layer owns
and nothing else (the screen returns ``None`` and the same wrong answer is
RETURNED).

**R3-C (was D-2)** -- ``_segment_passive`` answers False for a ``str``
payload, so a stack carrying material KEYS is never provably passive and the
guard is never reached.  Arm A is a resolved stack (refused); arm B is the
same geometry through ``prepare()`` with keyed materials (returned, with the
probe never run).

Also recorded, because both are cheap and both are contract claims the round-3
message rewrite could have broken: the arbiter leaves no ``_sliver_probe``
attribute on the caller's stack, and ``PMM_SLIVER_GUARD = False`` restores the
pre-guard behaviour bit for bit.

    python v6_open_items.py out.json
"""
import hashlib
import json
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import v_fixtures as F  # noqa: E402

from lumenairy.elements.pmm import PMMStack  # noqa: E402
from lumenairy.elements.pmm import stack as ps  # noqa: E402


def _sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:32]


def r3b():
    """One OWNED liner anywhere disarms the cross-layer refusal.

    Two regimes are scored, because the severity depends on which one a real
    caller lands in.  ``own / w >= 100`` is what the screen demands, so a
    liner disarms the screen only when it is within 100x of the manufactured
    cell -- the arm below pairs a PHYSICAL 2 nm liner (1.48e-3 of the 1.35 um
    period) with wall mismatches from 1e-3 down, which is the pairing an
    ordinary coated device would actually produce."""
    out = []
    for delta in (3e-5, 1e-5, 3e-6):
        for liner in (None, 1e-6, 1e-7):
            st = (F.vliner(delta, liner=liner) if liner is not None
                  else F.vliner_free(delta))
            cur = F.unguarded(st)
            hit = F.screen_hit(st)
            refused, msg, _o, warns = F.guarded(
                F.vliner(delta, liner=liner) if liner is not None
                else F.vliner_free(delta))
            out.append(dict(delta=delta, liner=liner, worst=cur["worst"],
                            screen=hit is not None,
                            own=(None if hit is None else hit[4]),
                            w_narrow=(None if hit is None else hit[0]),
                            own_over_w=(None if hit is None
                                        else hit[4] / hit[0]),
                            refused=refused, n_warns=len(warns),
                            within_layer_warn=any("WITHIN-LAYER" in w
                                                  for w in warns),
                            arm="pico"))
    # the PHYSICAL arm: a 2 nm liner on a 1.35 um period
    phys = 2.0e-9 / 1.35e-6
    for delta in (1e-3, 3e-4, 1e-4, 3e-5, 1e-5):
        for liner in (None, phys):
            st = (F.vliner(delta, liner=liner) if liner is not None
                  else F.vliner_free(delta))
            cur = F.unguarded(st)
            hit = F.screen_hit(st)
            ref0 = F.unguarded(F.vliner(0.0, liner=liner) if liner is not None
                               else F.vliner_free(0.0))
            e = F.move_shared(cur, ref0, pol=1)
            refused, _m, _o, warns = F.guarded(
                F.vliner(delta, liner=liner) if liner is not None
                else F.vliner_free(delta))
            out.append(dict(delta=delta, liner=liner, worst=cur["worst"],
                            screen=hit is not None,
                            own=(None if hit is None else hit[4]),
                            w_narrow=(None if hit is None else hit[0]),
                            own_over_w=(None if hit is None
                                        else hit[4] / hit[0]),
                            refused=refused, n_warns=len(warns),
                            err_d=e / delta, kind=F.classify(e, delta),
                            within_layer_warn=any("WITHIN-LAYER" in w
                                                  for w in warns),
                            arm="physical"))
    # the DISARM WINDOW itself: own / w between 1 and 100 is where the screen
    # is silent BECAUSE of the liner.  A 2 nm liner on a 1.35 um period puts
    # that window at wall mismatches of 15 pm .. 2 nm, which is the range a
    # coated device's wall coordinates actually differ by.
    ref0 = F.unguarded(F.vliner(0.0, liner=phys))
    for d in np.logspace(-4.85, -3.0, 14):
        d = float(d)
        st = F.vliner(d, liner=phys)
        cur = F.unguarded(st)
        hit = F.screen_hit(st)
        e = F.move_shared(cur, ref0, pol=1)
        refused, _m, _o, warns = F.guarded(F.vliner(d, liner=phys))
        out.append(dict(delta=d, liner=phys, worst=cur["worst"],
                        screen=hit is not None,
                        own=(None if hit is None else hit[4]),
                        w_narrow=(None if hit is None else hit[0]),
                        own_over_w=(None if hit is None
                                    else hit[4] / hit[0]),
                        refused=refused, n_warns=len(warns),
                        err_d=e / d, kind=F.classify(e, d),
                        within_layer_warn=any("WITHIN-LAYER" in w
                                              for w in warns),
                        arm="disarm_window"))
    return out


def r3c():
    """A KEYED prepared stack is outside the guard entirely."""
    P, WL = 1.35e-6, 1.064e-6
    a0, b0, delta = 0.3120, 0.6790, 3e-6

    def build(keyed):
        st = PMMStack(P, n_superstrate=1.0, n_substrate=1.52, degree=12,
                      far_field_orders=21, min_feature=P * F.NO_SNAP)
        for k in (0, 1):
            dd = delta * k
            lo = "LO" if keyed else 2.56
            hi = "HI" if keyed else 12.25
            st.add_layer(0.12e-6, segments=[(a0 - dd, lo),
                                            (b0 + dd - (a0 - dd), hi),
                                            (1.0 - (b0 + dd), lo)])
        return st

    def resolved():
        st = build(False)
        st.set_source(WL, theta=0.28)
        return st

    passive_plain = ps._stack_provably_passive(resolved())
    refused_a, msg_a, _o, _warns_a = F.guarded(resolved())

    prep = build(True).prepare()
    n_probe = {"n": 0}
    orig = ps._sliver_probe_solve

    def counting(*a, **kw):
        n_probe["n"] += 1
        return orig(*a, **kw)

    ps._sliver_probe_solve = counting
    try:
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            try:
                o, R, T, _J = prep.solve(wavelength=WL, theta=0.28,
                                         materials={"LO": 2.56, "HI": 12.25})
                keyed_refused, keyed_worst = False, float(np.max(
                    np.real(np.asarray(R)).sum(axis=-1)
                    + np.real(np.asarray(T)).sum(axis=-1)))
            except ValueError as exc:
                keyed_refused, keyed_worst = True, float("nan")
                rec = list(rec) + [type("W", (), {"message": str(exc)})]
            keyed_warns = [str(w.message) for w in rec]
    finally:
        ps._sliver_probe_solve = orig
    # the guard's own view of the keyed stack, before materials are resolved
    keyed_stack = build(True)
    keyed_stack.set_source(WL, theta=0.28)
    return dict(
        resolved_refused=refused_a, resolved_passive=passive_plain,
        resolved_msg_head=(msg_a or "")[:120],
        keyed_provably_passive=ps._stack_provably_passive(keyed_stack),
        keyed_screen=ps._sliver_screen(keyed_stack) is not None,
        keyed_refused=keyed_refused, keyed_worst=keyed_worst,
        keyed_probe_calls=n_probe["n"],
        keyed_warns=keyed_warns,
        keyed_n_warns=len(keyed_warns))


def contracts():
    """The two contract claims the message rewrite could have broken."""
    st = F.vgmr(3e-6, degree=6)
    before = sorted(vars(st))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            st.solve()
        refused = False
    except ValueError:
        refused = True
    after = sorted(vars(st))
    # guard disarmed -> the pre-guard behaviour, bit for bit
    raw = F.unguarded(F.vgmr(3e-6, degree=6))
    was = ps.PMM_SLIVER_GUARD
    ps.PMM_SLIVER_GUARD = False
    try:
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            o, R, T, J = F.vgmr(3e-6, degree=6).solve()
        i = np.argsort(np.asarray(o).ravel())
        off_R = _sha(np.real(np.asarray(R))[:, i])
        off_T = _sha(np.real(np.asarray(T))[:, i])
        off_warns = [str(w.message) for w in rec]
    finally:
        ps.PMM_SLIVER_GUARD = was
    return dict(refused=refused,
                probe_attr_left=("_sliver_probe" in after),
                new_attrs=[a for a in after if a not in before],
                guard_off_returns=True,
                guard_off_bitid=bool(off_R == _sha(raw["R"])
                                     and off_T == _sha(raw["T"])),
                guard_off_warns=off_warns)


def main():
    dest = sys.argv[1] if len(sys.argv) > 1 else "v6_open_items.json"
    t0 = time.perf_counter()
    doc = dict(build=F.build_info(), r3b=r3b(), r3c=r3c(),
               contracts=contracts())
    doc["wall"] = time.perf_counter() - t0
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(F.jsonable(doc), fh, indent=1)
    print(json.dumps(F.jsonable(doc["r3b"]), indent=1))
    print(json.dumps(F.jsonable({k: v for k, v in doc["r3c"].items()
                                 if k != "keyed_warns"}), indent=1))
    print(json.dumps(F.jsonable({k: v for k, v in doc["contracts"].items()
                                 if k != "guard_off_warns"}), indent=1))
    print(f"-> {dest}")


if __name__ == "__main__":
    main()
