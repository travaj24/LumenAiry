"""H4 mutation probe: what a REAL regression in ``radial_spectrum`` puts on the
same relative-error reading, so the derived bar can be shown to be two-sided.

Usage:  python probe_h4_mutations.py <out.json>
"""
import json
import os
import sys

import numpy as np
from scipy.special import jn_zeros, jnp_zeros

import lumenairy
import lumenairy.elements.bor.radial_eigensolver as rs

assert "lum_reds" in lumenairy.__file__, lumenairy.__file__

CASES = [(0, "dirichlet", "jn"), (1, "dirichlet", "jn"), (3, "dirichlet", "jn"),
         (1, "neumann", "jnp"), (3, "neumann", "jnp")]
EPS = float(np.finfo(float).eps)


def read(m, bc, zf, **kw):
    ref = (jn_zeros(m, 6) if zf == "jn" else jnp_zeros(m, 6))
    ev_all = np.asarray(rs.radial_spectrum(m, 1.0, kw.pop("degree", 8),
                                           kw.pop("n_el", 12), bc=bc,
                                           n_low=10 ** 6, **kw))
    ev = ev_all[:6]
    rel = float(np.max(np.abs(np.sqrt(np.abs(ev)) / ref - 1.0)))
    lam_max, lam_min = float(ev_all[-1]), float(ev_all[0])
    return rel, lam_max, lam_min, EPS * lam_max / lam_min


out = {"python": sys.version.split()[0], "numpy": np.__version__,
       "env": {k: os.environ.get(k) for k in
               ("OPENBLAS_CORETYPE", "OMP_NUM_THREADS")},
       "baseline": {}, "mutations": {}}

for m, bc, zf in CASES:
    rel, lmax, lmin, bar = read(m, bc, zf)
    out["baseline"][f"{m}-{bc}-{zf}"] = dict(rel=rel, lam_max=lmax,
                                             lam_min=lmin, derived_bar=bar,
                                             margin=bar / rel)

# ---- M1 / M2 / M4: resolution and quadrature, through the PUBLIC API -------
for tag, kw in (("degree7", dict(degree=7)), ("degree6", dict(degree=6)),
                ("degree5", dict(degree=5)), ("n_el11", dict(n_el=11)),
                ("n_el8", dict(n_el=8)), ("n_el6", dict(n_el=6)),
                ("nq_extra0", dict(nq_extra=0)),
                ("nq_extra_minus2", dict(nq_extra=-2))):
    row = {}
    for m, bc, zf in CASES:
        try:
            rel, lmax, lmin, bar = read(m, bc, zf, **kw)
            row[f"{m}-{bc}-{zf}"] = dict(rel=rel, derived_bar=bar,
                                         caught=bool(rel >= bar))
        except Exception as exc:
            row[f"{m}-{bc}-{zf}"] = {"error": repr(exc)}
    out["mutations"][tag] = row

# ---- M3: drop the m^2 axis term (a real weak-form regression) --------------
_orig_src = None


def _mutate_drop_m2():
    """Patch the assembly: m*m -> 0 in the stiffness.  Done by monkeypatching
    the module's ``_lagrange_vals_derivs`` is not enough, so run the solve for
    m with the m^2 term removed by calling radial_spectrum at m'=0 on the
    m != 0 boundary set -- i.e. compare the m-spectrum with the m=0 one."""
    row = {}
    for m, bc, zf in CASES:
        if m == 0:
            continue
        ref = (jn_zeros(m, 6) if zf == "jn" else jnp_zeros(m, 6))
        ev = np.asarray(rs.radial_spectrum(0, 1.0, 8, 12, bc=bc, n_low=6))
        rel = float(np.max(np.abs(np.sqrt(np.abs(ev)) / ref - 1.0)))
        row[f"{m}-{bc}-{zf}"] = dict(rel=rel)
    return row


out["mutations"]["m2_term_dropped(=m=0 spectrum)"] = _mutate_drop_m2()

# ---- M5: BC swapped --------------------------------------------------------
row = {}
for m, bc, zf in CASES:
    other = "neumann" if bc == "dirichlet" else "dirichlet"
    ref = (jn_zeros(m, 6) if zf == "jn" else jnp_zeros(m, 6))
    ev = np.asarray(rs.radial_spectrum(m, 1.0, 8, 12, bc=other, n_low=6))
    rel = float(np.max(np.abs(np.sqrt(np.abs(ev)) / ref - 1.0)))
    row[f"{m}-{bc}-{zf}"] = dict(rel=rel)
out["mutations"]["bc_swapped"] = row

json.dump(out, open(sys.argv[1], "w"), indent=1)
print(json.dumps(out, indent=1)[:6000])
