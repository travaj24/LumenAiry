"""TASK 2 follow-ups -- two questions ``v2_hybrid.py`` opened.

(A) THE LOSSY CLAIM, SHARPENED.  ``v2_hybrid.py`` finds that a hybrid stack
    with a LOSSY SPACER but a LOSSLESS patterned cell is NOT bit-identical
    across the round-2 arms at oblique incidence (5.5e-13 in per-order
    efficiency).  Which layer's loss is the discriminator?  Four fixtures at
    the same oblique/conical mount: loss in neither / spacer only / cell only /
    both.

(B) THE RESIDUAL CONDITIONING.  ``cond(a + b)`` at the PMM interface stays at
    1e7 .. 1e8 POST-fix on three of my five spacer fixtures, against 2e2 on the
    no-spacer control.  Is that residual a coincidence effect the fix does not
    reach, or an intrinsic near-degeneracy of the geometry?  Discriminator: a
    SPACER DETUNE ladder read on ``cond``, POST-fix.  If the residual falls
    away with the detune it is the coincidence; if it does not, it is the
    geometry.

Run:  PYTHONPATH=. python validation/probe_verify_branch_cut_round2/v7_followups.py out.json
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _vcommon as VC  # noqa: E402

VC.pin_tree()
import v2_hybrid as V2  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else "v7.json"
WL = V2.WL


def cellA(rel, loss=0.0):
    c = np.full((8, 8), 2.25, dtype=complex)
    c[1:4, 2:4] = 2.25 * (1.0 + rel)
    return c + 1j * loss


def hybrid(spacer_loss, cell_loss, theta, phi, M=4, detune=0.0):
    from lumenairy.elements.pmm import PMM2DStackHybrid
    st = PMM2DStackHybrid(0.62e-6, 0.58e-6, n_superstrate=1.0,
                          n_substrate=1.63, degree=7, n_orders=M,
                          symmetry=False)
    sp = (2.25 + 1j * spacer_loss) * (1.0 + detune)
    st.add_layer(0.12e-6, eps=sp)
    st.add_layer(0.23e-6, eps_cell=cellA(1e-6, cell_loss))
    st.add_layer(0.09e-6, eps=sp)
    st.set_source(WL, theta=theta, phi=phi)
    return st


def measure(spacer_loss, cell_loss, theta, phi, M=4, detune=0.0):
    tap = V2.CondTap()
    row = {}
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        try:
            with tap:
                res = hybrid(spacer_loss, cell_loss, theta, phi, M,
                             detune).solve()
            R = np.asarray(res[1])
            T = np.asarray(res[2])
            row["R"] = R.tolist()
            row["T"] = T.tolist()
            row["sumRT"] = float(R.sum() + T.sum())
            row["closure"] = float(R.sum() + T.sum() - 2.0)
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        row["warnings"] = sorted({type(w.message).__name__ for w in ws})
    row["cond"] = tap.summary()
    return row


def run():
    payload = {"loss_partition": {}, "cond_vs_detune": {}}

    # (A) which layer's loss decides whether the arms agree?
    for name, (sl, cl) in {
            "neither": (0.0, 0.0),
            "spacer_only": (1e-2, 0.0),
            "cell_only": (0.0, 1e-2),
            "both": (1e-2, 1e-2),
            "spacer_only_weak": (1e-6, 0.0),
            "cell_only_weak": (0.0, 1e-6),
    }.items():
        for mount, (th, ph) in {"normal": (0.0, 0.0),
                                "conical": (np.deg2rad(12.0),
                                            np.deg2rad(20.0))}.items():
            payload["loss_partition"][f"{name}_{mount}"] = measure(
                sl, cl, th, ph)

    # (B) POST-fix cond(a+b) against the spacer detune
    for d in (0.0, 1e-9, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 3e-1):
        for M in (3, 4, 5):
            r = measure(0.0, 0.0, 0.0, 0.0, M=M, detune=d)
            r.pop("R", None)
            r.pop("T", None)
            payload["cond_vs_detune"][f"d{d:.0e}_M{M}"] = r
    VC.dump(OUT, payload)

    st = VC.stamp()
    print(f"ARM = {st['arm']}")
    print("\n(A) loss partition -- closure and cond, per fixture:")
    for k, r in payload["loss_partition"].items():
        print(f"  {k:26s} closure {r.get('closure', float('nan')):13.5e}  "
              f"cond_pmm_ifc "
              f"{r['cond'].get('per_site_max', {}).get('pmm interface mode-match (a+b)', float('nan')):11.3e}"
              f"  {r.get('warnings') or ''} {r.get('error') or ''}")
    print("\n(B) POST-fix cond(a+b) against the spacer detune:")
    for k, r in payload["cond_vs_detune"].items():
        pm = r["cond"].get("per_site_max", {})
        print(f"  {k:12s} closure {r.get('closure', float('nan')):12.4e}  "
              f"pmm ifc {pm.get('pmm interface mode-match (a+b)', float('nan')):11.3e}  "
              f"star {max([v for kk, v in pm.items() if 'star' in kk] or [float('nan')]):11.3e}")


if __name__ == "__main__":
    run()
