"""Smoke: do the round-2 coincidence fixtures solve at all, and does the
NumPy PMM copy see LAYER eigenvalues on each of them?

Not a decision probe -- it exists so the measurement probes are written against
surfaces that run.  Prints one line per surface.
"""
import sys
import warnings

sys.path.insert(0, "validation/probe_fix_branch_cut_round2")
import b_fixtures as F  # noqa: E402

F.require_local_tree()

from lumenairy.elements.pmm import (  # noqa: E402
    PMM2DStackHybrid,
    PMM2DStackPure,
    pmm_efficiency_2d,
    pmm_efficiency_2d_cell,
    pmm_efficiency_2d_staggered,
    pmm_jones_2d,
    pmm_jones_2d_staggered,
)

WL, PX, D = F.WL, F.PX, F.DEPTH
NS = 1.5                       # substrate index; NS**2 = 2.25 = the host


def run(name, fn, jones=False):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with F.PMMEigSpy() as spy:
            try:
                res = fn()
            except Exception as exc:
                print("%-34s RAISED %s" % (name, repr(exc)[:110]))
                return
        s = spy.summary()
    cl = F.closure_jones(res) if jones else F.closure_eff(res)
    warn = ";".join(sorted({type(x.message).__name__ for x in w}))
    print("%-34s closure %+.4e  calls=%s modes=%s oncut=%s incoming=%s  %s"
          % (name, cl, s.get("n_calls"), s.get("modes"), s.get("on_cut"),
             s.get("incoming_after_exact_pin"), warn))


run("pmm_efficiency_2d_cell coinc",
    lambda: pmm_efficiency_2d_cell(PX, PX, F.pillar_cell(), NS, 1.0, D, WL,
                                   degree=7, n_orders=5))
run("pmm_efficiency_2d coinc",
    lambda: pmm_efficiency_2d(PX, PX, 6.0, 2.25, (0.125e-6, 0.375e-6),
                              (0.125e-6, 0.375e-6), NS, 1.0, D, WL,
                              degree=7, n_orders=5))
run("pmm_jones_2d coinc",
    lambda: pmm_jones_2d(PX, PX, F.tensor_cell(), NS, 1.0, D, WL,
                         degree=7, n_orders=3), jones=True)


def hybrid(spacer, nsub):
    st = PMM2DStackHybrid(PX, PX, n_substrate=nsub, n_superstrate=1.0,
                          degree=7, n_orders=4)
    if spacer:
        st.add_layer(0.1e-6, eps=2.25)
    st.add_layer(D, eps_cell=F.pillar_cell())
    if spacer:
        st.add_layer(0.1e-6, eps=2.25)
    return st.set_source(WL, theta=0.0).solve()


run("hybrid stack region-coinc", lambda: hybrid(False, NS), jones=True)
run("hybrid stack spacer+region", lambda: hybrid(True, NS), jones=True)
run("hybrid stack spacer only", lambda: hybrid(True, 1.63), jones=True)
run("hybrid stack neither", lambda: hybrid(False, 1.63), jones=True)

run("stag efficiency coinc",
    lambda: pmm_efficiency_2d_staggered(PX, PX, F.stag_cell(), NS, 1.0, D, WL,
                                        degree=6, n_orders=4))
run("stag jones tensor coinc",
    lambda: pmm_jones_2d_staggered(PX, PX, F.stag_tensor_cell(), NS, 1.0, D,
                                   WL, degree=6, n_orders=3), jones=True)
run("stag jones OOP coinc",
    lambda: pmm_jones_2d_staggered(PX, PX, F.stag_tensor_cell(oop=0.3), NS,
                                   1.0, D, WL, degree=6, n_orders=3),
    jones=True)


def pure(spacer, nsub):
    st = PMM2DStackPure(PX, PX, n_substrate=nsub, n_superstrate=1.0,
                        degree=6, n_orders=4)
    if spacer:
        st.add_layer(0.1e-6, eps=2.25)
    st.add_layer(D, eps_cell=F.stag_cell())
    if spacer:
        st.add_layer(0.1e-6, eps=2.25)
    return st.set_source(WL, theta=0.0).solve()


run("pure stack region-coinc", lambda: pure(False, NS), jones=True)
run("pure stack spacer+region", lambda: pure(True, NS), jones=True)
run("pure stack neither", lambda: pure(False, 1.63), jones=True)

print("stamp:", {k: v for k, v in F.arm_stamp().items()
                 if k in ("arm", "pmm_twod_has_exact_pin", "jax_copies")})
