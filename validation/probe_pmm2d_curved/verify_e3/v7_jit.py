"""V7 (E3-6): jit traces ONCE over a 10-radius sweep; compiled timings; the
memory peak of the compile.

    python v7_jit.py M

A Python counter inside the traced function counts traces; jit's own cache
size is read too.  Memory: the process peak working set (Windows,
psutil peak_wset) / peak RSS (Linux, ru_maxrss) before the template, after
the template, after the forward compile, after the gradient compile.  The
box is shared with sibling builds: every time is an upper bound.
"""
import sys

from _ve3 import WL, P, PMM2DStackPure, dump, jax, jnp, np, tic

from lumenairy.elements.pmm import Circle

M = int(sys.argv[1])


def peak_mb():
    try:
        import psutil
        mi = psutil.Process().memory_info()
        return float(getattr(mi, "peak_wset", mi.rss)) / 2**20
    except ImportError:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


out = {"M": M, "peak_MB": {"start": peak_mb()}}
t = tic()
st = PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=1.45, n_modes=M,
                    n_orders=2, backend="jax")
st.add_layer(0.45, shapes=[Circle(0.6, 0.6, 0.33, 3.5)], background_eps=1.0)
st.set_source(WL)
tw = st.jax_twin()
out["template_s"] = tic() - t
out["peak_MB"]["template"] = peak_mb()
count = {"n": 0}


def f(r):
    count["n"] += 1
    p = tw.params()
    p["layers"][0]["shapes"] = [Circle(0.6, 0.6, r, 3.5)]
    _o, R, T, J = st.solve(params=p)
    return T[0, tw.p0]


radii = np.linspace(0.30, 0.36, 10)
fj = jax.jit(f)
t = tic()
fj(radii[0]).block_until_ready()
out["fwd_compile_plus_run_s"] = tic() - t
out["peak_MB"]["fwd_compiled"] = peak_mb()
ts = []
for r in radii[1:]:
    t = tic()
    fj(r).block_until_ready()
    ts.append(tic() - t)
out["fwd_run_s"] = [min(ts), float(np.median(ts)), max(ts)]
out["traces_after_fwd_sweep"] = count["n"]
out["fwd_cache_size"] = int(fj._cache_size())
count["n"] = 0
gj = jax.jit(jax.grad(f))
t = tic()
gj(radii[0]).block_until_ready()
out["grad_compile_plus_run_s"] = tic() - t
out["peak_MB"]["grad_compiled"] = peak_mb()
ts = []
for r in radii[1:]:
    t = tic()
    gj(r).block_until_ready()
    ts.append(tic() - t)
out["grad_run_s"] = [min(ts), float(np.median(ts)), max(ts)]
out["traces_after_grad_sweep"] = count["n"]
out["grad_cache_size"] = int(gj._cache_size())
# a vmapped sweep: one trace for all ten radii at once
count["n"] = 0
try:
    vg = jax.jit(jax.vmap(jax.grad(f)))
    t = tic()
    vg(jnp.asarray(radii)).block_until_ready()
    out["vmap_grad_compile_plus_run_s"] = tic() - t
    out["traces_vmap"] = count["n"]
except Exception as exc:  # noqa: BLE001 -- the probe records it
    out["vmap_error"] = type(exc).__name__ + ": " + str(exc)[:200]
out["peak_MB"]["end"] = peak_mb()
# the NumPy reference solve time
t = tic()
sn = PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=1.45, n_modes=M,
                    n_orders=2)
sn.add_layer(0.45, shapes=[Circle(0.6, 0.6, 0.33, 3.5)], background_eps=1.0)
sn.set_source(WL)
sn.solve()
out["numpy_solve_s"] = tic() - t
print(out)
print(dump(f"v7_jit_M{M}.json", out))
