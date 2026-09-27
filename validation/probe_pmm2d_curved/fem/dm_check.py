"""Cross-check: DynaMeta solve_fem (0-order R/T + Poynting-flux all-order R_flux/T_flux) on the SAME
quarter-cell geometry as fem_circle.py (single-slab buffers named superstrate/substrate, DynaMeta's
own PML alpha = 1j).  ASCII-only source."""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fem_circle as F


def main(hscale, p, hedge):
    import ngsolve as ng
    from dynameta.geometry.specs import OpticalSpec
    from dynameta.optics.ngsolve_layered import OpticalGeometry
    from dynameta.optics.solver import solve_fem
    ng.SetNumThreads(6)
    mesh, lay = F.build(hscale, nslab=1, h_edge=hedge, dm_names=True)
    mesh.Curve(p)
    zs, zt, H = lay["zs"], lay["zt"], F.H
    zint = {"pml_bot": (zs - lay["dpml"], zs), "substrate": (zs, 0.0), "pillar": (0.0, H),
            "host": (0.0, H), "superstrate": (H, zt), "pml_top": (zt, zt + lay["dpml"])}
    mat = {"pml_bot": "sub", "substrate": "sub", "pillar": "pil", "host": "air", "superstrate": "air",
           "pml_top": "air"}
    role = {"pml_bot": "pml", "substrate": "sub", "pillar": "inclusion", "host": "layer",
            "superstrate": "sup", "pml_top": "pml"}
    geo = OpticalGeometry(mesh=mesh, z_intervals_nm=zint, period_x_nm=F.Q, period_y_nm=F.Q,
                          z_super_interface_nm=zt, z_sub_interface_nm=zs, material_by_region=mat,
                          role_by_region=role, n_px=0, n_py=0, sym_x=True, sym_y=True)
    eps = mesh.MaterialCF({"pillar": F.EPS_P, "substrate": F.N_SUB ** 2, "pml_bot": F.N_SUB ** 2},
                          default=1.0)
    opt = OpticalSpec(polarization="y", linear_solver="sparsecholesky")
    t = time.time()
    res = solve_fem(geo, F.WL * 1e-9, eps, opt, order=p, n_super=1.0, n_sub=F.N_SUB)
    out = dict(hscale=hscale, p=p, hedge=hedge, ntet=mesh.ne, R00=res.R, T00=res.T, r=[res.r.real, res.r.imag],
               R_flux=res.R_flux, T_flux=res.T_flux, A_indep=res.A_independent, fit_relres=res.fit_relres,
               dt=time.time() - t)
    print(json.dumps(out))
    with open(os.path.join(HERE, "dm_check.jsonl"), "a") as fh:
        fh.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main(float(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]) if len(sys.argv) > 3 else None)
