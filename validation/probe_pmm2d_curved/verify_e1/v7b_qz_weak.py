"""V7b -- the QZ fallback forced off on WEAKLY lossy permeabilities: does
the Cholesky (which reads only the lower triangle) fail loudly, or return a
silently wrong answer?  Loss scaled from 1e-1 down to 1e-8 on the
symmetric lossy mu; unmapped and sheared-map slab, conical, M = 5."""
import _ve1common as V
import _ve1mut as VM
import numpy as np

out = {}
base = V.MU_SYM_LOSSY.real.astype(complex)
for s in (1e-1, 1e-2, 1e-3, 1e-5, 1e-8, 1e-11, 1e-13):
    mu = base + 1j * s * np.diag([1.0, 0.7, 0.9])
    mu[0, 1] = mu[1, 0] = 0.15 + 0.2j * s
    for mp in ("none", "sh4"):
        cm = V.make_map(mp, V.SLAB["P"])
        rec = {}
        for arm in ("none", "qz_off"):
            with VM.vmutate(arm):
                try:
                    r = V.slab_run(V.DIRGEN, cm, 5, "c", mu=mu)
                    rec[arm] = dict(worst=V.worst(r), clo=r["clo"])
                except Exception as exc:
                    rec[arm] = f"RAISES {type(exc).__name__}: {exc}"[:120]
        out[f"{mp}_{s:g}"] = rec
        print(mp, s, rec, flush=True)
V.dump("v7b_qz_weak.json", out)
