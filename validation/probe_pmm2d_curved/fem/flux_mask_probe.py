"""Probe: does a z-MASKED (IfPos) volume flux integral -- DynaMeta _poynting_flux_rt's construction --
reproduce the slab-aligned flux on the same field?  Explains the DynaMeta R_flux/T_flux mismatch on
this diffracting cell.  ASCII-only source."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fem_circle as F
import ngsolve as ng

mesh, lay = F.build(1.0)
gfu, info = F.solve(mesh, lay, p=3)
K0 = F.K0
Erf = gfu + ng.CoefficientFunction((0.0, info["r_f"] * ng.exp(1j * K0 * ng.z), 0.0))
cE = ng.curl(gfu) + ng.CoefficientFunction((-1j * K0 * info["r_f"] * ng.exp(1j * K0 * ng.z), 0.0, 0.0))
sz = 1j * (Erf[0] * ng.Conj(cE[1]) - Erf[1] * ng.Conj(cE[0]))
out = {}
for (zlo, zhi) in ((500.0, 950.0), (550.0, 900.0), (575.0, 875.0)):
    mask = ng.IfPos(ng.z - zlo, 1.0, 0.0) * ng.IfPos(zhi - ng.z, 1.0, 0.0)
    for order in (8, 20):
        num = complex(ng.Integrate(sz * mask, mesh, order=order)).real
        den = complex(ng.Integrate(mask, mesh, order=order)).real
        out["%g-%g o%d" % (zlo, zhi, order)] = num / (den * K0)
print(json.dumps(out, indent=1))
