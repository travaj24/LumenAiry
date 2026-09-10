"""W4 -- the ARBITER's side effects, independently measured.

  A  the probe NEVER mutates the caller's stack: a deep state hash of every
     public and private attribute the stack carries, taken before and after a
     guarded solve that fires the arbiter, plus ``_src`` identity, the layer
     tuples' identity, and a prepared object's two caches.
  B  it never RECURSES: ``_sliver_probe`` is set on the clone, and a clone
     that itself carries a sliver is not arbitrated again -- counted by
     spying on ``_sliver_probe_solve``.
  C  the COST: guarded solve vs the extra solve, and the ratio (claimed 0.20x
     Windows / 0.27x WSL).
  D  the FIRING RATE on a CONVERGED correct box (claimed 0 of 600).
  E  the ``unknown`` branch: keyed / dispersive stacks fall back to the
     round-1 decision AND the message says the attribution could not be run.

    python w4_side_effects.py [out.json]
"""
import hashlib
import json
import os
import sys
import time
import warnings

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from w_fixtures import classify, shared_move, unguarded, wbuild  # noqa: E402

import lumenairy  # noqa: E402
from lumenairy.elements.pmm import PMMStack  # noqa: E402
from lumenairy.elements.pmm import stack as ps  # noqa: E402

O11 = dict(period=1.2e-6, wl=0.85e-6, th=0.15, a0=0.27865, b0=0.62505,
           eh=2.25, ep=9.0, dz=0.32e-6 / 4)


def attr_map(st):
    """A per-attribute fingerprint of everything the stack stores, including
    the layer tuples' ``id()``s -- so a probe that REPLACED a layer list
    (rather than mutating it) is caught too.  Returned per attribute, not as
    one hash, because the SOLVE itself legitimately writes ``_modal``: the
    claim under test is that the guard adds NOTHING to that set."""
    out = {}
    for k, v in vars(st).items():
        if k == "_layers":
            out[k] = hashlib.sha256(
                "".join(f"{float(t)}|{float(ang)}|{id(segs)}|"
                        + "".join(f"{float(w)}|"
                                  + (e if isinstance(e, str)
                                     else repr(np.asarray(e,
                                                          dtype=complex)))
                                  for w, e in segs)
                        for t, segs, ang in v).encode()).hexdigest()[:16]
        else:
            try:
                out[k] = hashlib.sha256(repr(v).encode()).hexdigest()[:16]
            except Exception:                                  # noqa: BLE001
                out[k] = f"<{type(v).__name__}>"
    return out


def changed(a, b):
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))


class Spy:
    """Counts (and forwards) every arbiter re-solve."""

    def __init__(self):
        self.n = 0
        self.srcs = []
        self.mfs = []
        self.probe_flags = []
        self._orig = ps._sliver_probe_solve

    def __enter__(self):
        spy = self

        def patched(stack, mf_fix, src):
            spy.n += 1
            spy.mfs.append(float(mf_fix))
            spy.srcs.append(dict(src))
            out = spy._orig(stack, mf_fix, src)
            return out

        ps._sliver_probe_solve = patched
        return self

    def __exit__(self, *a):
        ps._sliver_probe_solve = self._orig
        return False


def arm_A():
    """The probe must not touch the caller's stack."""
    out = {}
    for delta in (1e-4, 3e-5):
        st = wbuild(delta, 14, **O11)
        before = attr_map(st)
        src_id, layers_id = id(st._src), id(st._layers)
        src_copy = dict(st._src)
        with Spy() as spy, warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            try:
                st.solve()
                verdict = "returned"
            except ValueError as exc:
                verdict = "REFUSED" if "SLIVER" in str(exc) else "raised"
        after = attr_map(st)
        # the CONTROL: the same solve with the guard disarmed, so the
        # attributes the SOLVE writes are separated from any the guard would
        st2 = wbuild(delta, 14, **O11)
        b2 = attr_map(st2)
        was = ps.PMM_SLIVER_GUARD
        ps.PMM_SLIVER_GUARD = False
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                st2.solve()
        finally:
            ps.PMM_SLIVER_GUARD = was
        ctrl = changed(b2, attr_map(st2))
        out[f"delta={delta:g}"] = dict(
            verdict=verdict, probes=spy.n,
            changed_with_guard=changed(before, after),
            changed_without_guard=ctrl,
            state_identical_modulo_solve=changed(before, after) == ctrl,
            src_object_replaced=id(st._src) != src_id,
            layers_object_replaced=id(st._layers) != layers_id,
            src_value_identical=dict(st._src) == src_copy,
            has_sliver_probe_attr=hasattr(st, "_sliver_probe"),
            min_feature=float(st.min_feature),
            n_warnings=len(rec))
    # the PREPARED path: the caches must not gain the probe's entries
    stp = PMMStack(1.2e-6, n_substrate=1.0, degree=14, far_field_orders=31,
                   min_feature=1.2e-18)
    a0, b0, d = 0.27865, 0.62505, 1e-4
    for k in range(2):
        dd = d * k
        stp.add_layer(0.32e-6 / 4, segments=[(a0 - dd, "LC"),
                                             (b0 + dd - (a0 - dd), 9.0),
                                             (1.0 - (b0 + dd), 2.25)])
    prep = stp.prepare()
    with Spy() as spy:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                prep.solve(wavelength=0.85e-6, angle=0.15,
                           materials={"LC": 2.25})
                pv = "returned"
            except ValueError as exc:
                pv = "REFUSED" if "SLIVER" in str(exc) else f"raised: {exc}"
    out["prepared"] = dict(verdict=pv, probes=spy.n,
                           probe_src=spy.srcs[:1],
                           eig_cache=len(prep._eig_cache),
                           mats_cache=len(prep._mats_cache),
                           src_on_stack=getattr(stp, "_src", None) is not None)
    # and the PREPARED path's caches must be untouched by the probe: solve a
    # HEALTHY prepared point twice and compare cache sizes
    sth = PMMStack(1.2e-6, n_substrate=1.0, degree=14, far_field_orders=31)
    sth.add_layer(0.08e-6, segments=[(0.3, "LC"), (0.7, 2.25)])
    ph = sth.prepare()
    with Spy() as spy2:
        for _ in range(2):
            ph.solve(wavelength=0.85e-6, angle=0.15, materials={"LC": 2.25})
    out["prepared_healthy"] = dict(probes=spy2.n, eig_cache=len(ph._eig_cache),
                                   mats_cache=len(ph._mats_cache))
    return out


def arm_B():
    """No recursion: the clone carries ``_sliver_probe`` and is never
    arbitrated.  Measured by counting probes -- exactly one per refusal."""
    rows = {}
    for delta in (3e-3, 1e-4, 3e-5):
        st = wbuild(delta, 14, **O11)
        with Spy() as spy:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    st.solve()
                    v = "returned"
                except ValueError:
                    v = "REFUSED"
        rows[f"delta={delta:g}"] = dict(verdict=v, probes=spy.n)
    # and directly: a stack MARKED as a probe is inert
    st = wbuild(3e-5, 14, **O11)
    st._sliver_probe = True
    with Spy() as spy:
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            try:
                st.solve()
                v = "returned"
            except ValueError:
                v = "REFUSED"
    rows["marked_probe"] = dict(verdict=v, probes=spy.n,
                                screen=ps._sliver_screen(st) is not None,
                                hazard=ps._within_layer_hazard(
                                    st, None) is not None)
    return rows


def arm_C(reps=7):
    """Cost.  The guarded solve is timed with the guard DISARMED (so it is the
    solve alone) and the arbiter's extra solve is timed on its own clone."""
    st = wbuild(3e-5, 14, **O11)
    hit = ps._cross_layer_sliver([L[1] for L in st._layers],
                                 st.min_feature / st.period)
    mf = 2.0 * hit[3] * float(st.period)
    was = ps.PMM_SLIVER_GUARD
    ps.PMM_SLIVER_GUARD = False
    base = []
    try:
        for _ in range(reps):
            t = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                st.solve()
            base.append(time.perf_counter() - t)
    finally:
        ps.PMM_SLIVER_GUARD = was
    probe = []
    for _ in range(reps):
        t = time.perf_counter()
        ps._sliver_probe_solve(st, mf, dict(st._src))
        probe.append(time.perf_counter() - t)
    b, p = float(np.median(base)), float(np.median(probe))
    return dict(guarded_solve_ms=b * 1e3, arbiter_ms=p * 1e3, ratio=p / b,
                reps=reps)


def arm_D(n_delta=50):
    """The firing rate on a CONVERGED correct box: five fixtures x degrees
    x deltas, counting how many rows run the arbiter and how many of those
    the continuity rule calls RIGHT."""
    fixtures = {
        "C_nir": dict(period=1.05e-6, wl=0.98e-6, th=0.42, a0=0.2350,
                      b0=0.6650, eh=2.10, ep=4.00, dz=0.15e-6),
        "O11": O11,
        "B_vis": dict(period=0.74e-6, wl=0.53e-6, th=0.31, a0=0.19137,
                      b0=0.71429, eh=1.96, ep=6.25, dz=0.06e-6),
    }
    deltas = np.geomspace(3e-3, 1e-4, n_delta)     # the CONVERGED band only
    n_rows = n_fire = n_fire_right = n_right = 0
    worst_right = 0.0
    for name, kw in fixtures.items():
        for deg in (12, 14, 16, 18):
            ref = unguarded(wbuild(0.0, deg, **kw))
            for d in deltas:
                st = wbuild(float(d), deg, **kw)
                with Spy() as spy:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        try:
                            st.solve()
                        except ValueError:
                            pass
                cur = unguarded(st)
                e = shared_move(cur, ref, pol=1)
                k = classify(e, float(d))
                n_rows += 1
                n_fire += 1 if spy.n else 0
                if k == "right":
                    n_right += 1
                    worst_right = max(worst_right, abs(cur["worst"] - 1.0))
                    n_fire_right += 1 if spy.n else 0
    return dict(n_rows=n_rows, n_arbiter_fired=n_fire, n_right=n_right,
                n_arbiter_fired_on_right=n_fire_right,
                worst_superunity_among_right=worst_right, fixtures=len(
                    fixtures))


def arm_E():
    """The ``unknown`` branch."""
    out = {}
    # (1) DISPERSIVE: a callable eps.  _stack_provably_passive answers False,
    #     so the geometric screen is never reached at all.
    st = PMMStack(1.2e-6, n_substrate=1.0, degree=14, far_field_orders=31,
                  min_feature=1.2e-18)
    a0, b0, d = 0.27865, 0.62505, 3e-5
    for k in range(2):
        dd = d * k
        st.add_layer(0.32e-6 / 4,
                     segments=[(a0 - dd, 2.25),
                               (b0 + dd - (a0 - dd), lambda w: 9.0 + 0.0 * w),
                               (1.0 - (b0 + dd), 2.25)])
    st.set_source(0.85e-6, theta=0.15)
    with Spy() as spy, warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        try:
            o, R, T, J = st.solve_vs_wavelength(np.array([0.85e-6]),
                                                theta=0.15)
            v = "returned"
            tot = float(np.max(np.real(R).sum(-1) + np.real(T).sum(-1)))
        except ValueError as exc:
            v, tot = ("REFUSED" if "SLIVER" in str(exc) else "raised"), None
    out["dispersive_sweep"] = dict(
        verdict=v, probes=spy.n, worst=tot,
        passive=ps._stack_provably_passive(st),
        screen=ps._sliver_screen(st) is not None,
        warnings=[str(w.message)[:160] for w in rec])
    # (2) NO RESOLVED SOURCE -> the arbiter cannot run: force it by calling
    #     the guard with a stack whose _src was never set.
    st2 = wbuild(3e-5, 14, **O11)
    cur = unguarded(st2)
    st2._src = None
    with Spy() as spy, warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        try:
            ps._warn_stack_energy(cur["R"], cur["T"], stack=st2)
            v = "returned"
            msg = ""
        except ValueError as exc:
            v, msg = "REFUSED", str(exc)
    out["no_source"] = dict(
        verdict=v, probes=spy.n,
        says_could_not_be_run=("could NOT be run" in msg),
        arbiter_verdict=str(ps._sliver_arbiter(st2, cur["worst"], cur["R"],
                                               cur["T"], None)[0]),
        warnings=[str(w.message)[:120] for w in rec])
    # (3) a stack whose super-unity is BETWEEN the trigger and the round-1
    #     bar, with the arbiter forced to 'unknown': round 1 WARNED there, so
    #     round 2 must warn too (not raise).
    st3 = wbuild(1e-4, 14, **O11)
    cur3 = unguarded(st3)
    st3._src = None
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        try:
            ps._warn_stack_energy(cur3["R"], cur3["T"], stack=st3)
            v3 = "returned"
        except ValueError:
            v3 = "REFUSED"
    out["no_source_below_round1_bar"] = dict(
        verdict=v3, worst=cur3["worst"],
        above_round1_bar=cur3["worst"] > 1.0 + ps._STACK_SUPERUNITY_BAR,
        n_warnings=len(rec))
    return out


def main():
    out_path = (sys.argv[1] if len(sys.argv) > 1
                else os.path.join(HERE, "w4_side_effects.json"))
    lib = os.path.abspath(lumenairy.__file__)
    print("lumenairy:", lib)
    out = dict(lumenairy=lib, python=sys.version.split()[0],
               numpy=np.__version__)
    for name, fn in (("A_no_mutation", arm_A), ("B_no_recursion", arm_B),
                     ("C_cost", arm_C), ("D_firing_rate", arm_D),
                     ("E_unknown", arm_E)):
        t = time.time()
        out[name] = fn()
        print(f"== {name} ({time.time() - t:.1f} s)")
        print(json.dumps(out[name], indent=1, default=str))
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print("->", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
