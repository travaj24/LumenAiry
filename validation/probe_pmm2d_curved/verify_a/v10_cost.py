"""V10 -- cost of the mapped assembly: wall time of the operator assembly
(Granet2DTransverseE construction, best of 2) and the PEAK resident set of a
fresh process that builds one solver and runs its region eig, at M = 6, 8, 10
on the verifier's 3 x 3 stripe cell.

usage: v10_cost.py run            (spawns one child per (config, M))
       v10_cost.py child <cfg> <M>
"""
import json
import os
import subprocess
import sys
import time

CFGS = ("none", "identity", "sine0.08", "harm_asym")


def child(cfg, M):
    import _vcommon as C
    import psutil

    from lumenairy.elements.pmm import twod_staggered as TS
    from lumenairy.elements.pmm._curvemap import IdentityMap
    eps = C.cell("stripe")
    if cfg == "none":
        args, kw = (3, 3), {}
    elif cfg == "identity":
        args, kw = (3, 3), {"cmap": IdentityMap(3, 3, C.P, C.P)}
    else:
        f = C.sine(0.08) if cfg == "sine0.08" else C.HarmonicStretch(
            0.10, 0.04, 0.9)
        cm = C.stretch_map(f)
        args, kw = (cm.u_walls, cm.v_walls), {"cmap": cm}
    ts = []
    for _ in range(2):
        t = time.perf_counter()
        s = TS.Granet2DTransverseE(C.P, C.P, *args, M, eps, k0=C.K0, **kw)
        ts.append(time.perf_counter() - t)
    nq = int(s._qrule[0].size) if s.cmap is not None else None
    t = time.perf_counter()
    TS._region_modes(s)
    teig = time.perf_counter() - t
    p = psutil.Process()
    if os.name == "nt":
        peak = p.memory_info().peak_wset
    else:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    print(json.dumps(dict(cfg=cfg, M=M, assembly_s=min(ts), eig_s=teig,
                          nq=nq, peak_rss_MB=peak / 2 ** 20)))


def run():
    import _vcommon as C
    rows = []
    for M in (6, 8, 10):
        for cfg in CFGS:
            out = subprocess.run([sys.executable, __file__, "child", cfg,
                                  str(M)], capture_output=True, text=True,
                                 env=os.environ.copy(), check=True)
            r = json.loads(out.stdout.strip().splitlines()[-1])
            rows.append(r)
            print(r, flush=True)
    C.dump("v10_cost", {"rows": rows})


if __name__ == "__main__":
    if sys.argv[1] == "child":
        child(sys.argv[2], int(sys.argv[3]))
    else:
        run()
