"""E1-9 -- the mutation matrix, pillar rows.  Each engineered defect of
``_e1common.mutate`` against the two PATTERNED gates at fixed M (the slab
rows -- the Berreman gates -- are ``e3_arms_*.json``):

* pillar  -- the OOP30 disk under the c3 map (E1-5), normal and conical
  (25, 40 deg), M = 6: the largest change of the 36-entry R / T vector and
  of both Jones matrices against the correct arm;
* slant   -- the eps-4 disk slanted (0.2, 0) under the c3 map (E1-6),
  normal, M = 6, likewise;
* slantoop-- the OOP30 disk slanted (E1-7), conical, M = 6.

usage: python e9_mutations.py <gate> <kind|correct>     -> e9_<gate>_<kind>.json
       python e9_mutations.py summary
"""
import json
import os
import sys

import _common as C
import _e1common as E
import e5_pillar as E5
import numpy as np

GATES = {"pillar": ("oop30", None, 0.0, 0.0),
         "pillar_con": ("oop30", None, 25.0, 40.0),
         "slant": ("eps4", (0.2, 0.0), 0.0, 0.0),
         "slantoop_con": ("oop30", (0.2, 0.0), 25.0, 40.0)}
M = 6


def run(gate, kind):
    mat, slant, th, ph = GATES[gate]
    t33 = (4.0 * np.eye(3, dtype=complex) if mat == "eps4"
           else E5.MATS[mat])
    cm, eps = E5.disk_cells("c3", t33)

    def solve():
        st = E.PMM2DStackPure(E5.P, E5.P, n_superstrate=1.0,
                              n_substrate=E5.NSUB, n_modes=M, n_orders=3,
                              cmap=cm)
        st.add_layer(E5.DEP, eps_cell=eps, slant=slant)
        st.set_source(E5.WL, theta=np.deg2rad(th), phi=np.deg2rad(ph))
        o, R, T, J = st.solve(jones=True)
        o, R, T = np.asarray(o), np.asarray(R), np.asarray(T)
        return {"vec": C.vec(o, R, T).tolist(),
                "Jr": np.asarray(J, dtype=complex),
                "Jt": np.asarray(E.jones_t(st), dtype=complex),
                "closure": float(np.abs(R.sum(1) + T.sum(1) - 1).max())}
    if kind == "correct":
        res = solve()
    else:
        with E.mutate(kind):
            res = solve()
    E.dump(f"e9_{gate}_{kind}.json", res)
    print(gate, kind, res["closure"], flush=True)


def summary():
    here = os.path.dirname(os.path.abspath(__file__))
    out = {}
    for gate in GATES:
        fn = os.path.join(here, f"e9_{gate}_correct.json")
        if not os.path.exists(fn):
            continue
        ref = json.load(open(fn))
        rv = np.array(ref["vec"])
        def cx(d):
            return np.array(d["re"]) + 1j * np.array(d["im"])
        rJr, rJt = cx(ref["Jr"]), cx(ref["Jt"])
        out[gate] = {"correct_closure": ref["closure"]}
        for kind in E.MUTATIONS:
            fk = os.path.join(here, f"e9_{gate}_{kind}.json")
            if not os.path.exists(fk):
                continue
            r = json.load(open(fk))
            out[gate][kind] = {
                "dRT": float(np.abs(np.array(r["vec"]) - rv).max()),
                "dJr": float(np.abs(cx(r["Jr"]) - rJr).max()),
                "dJt": float(np.abs(cx(r["Jt"]) - rJt).max()),
                "closure": r["closure"]}
    E.dump("e9_summary.json", out)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "summary":
        summary()
    else:
        run(sys.argv[1], sys.argv[2])
