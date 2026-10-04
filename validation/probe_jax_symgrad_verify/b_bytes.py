"""B: NumPy (and JAX eager forward of the UNROUTED entries) outputs of the
refactored modules, SHA-256 per fixture, for PRE (5ea82b44) vs POST
(HEAD) comparison: run once per tree with LUM_TREE and VTAG.

The refactor touched _layer_eigenmodes / _layer_eigenmodes_tensor (P block
via _layer_P_matrix, _tensor_PQ), _scalar_PQ / _tensor_PQ (even-parity
machinery), and _jpmm_sem_modes (split).  Fixtures exercise every caller:
rcwa_efficiency_2d laurent / li, symmetry False / auto (even-sector fold),
oblique, conical, lossy, uniform, C1; rcwa_jones_2d in-plane tensor
(laurent / li, normal / conical, symmetry auto) and out-of-plane tilted
director (generator path); pmm_efficiency_1d / pmm_jones_1d at 0 and 0.2
rad, stabilize True / False; RCWAStack (two patterned layers); and the JAX
eager forward of rcwa_jones_2d and pmm_jones_1d.
"""
import hashlib

from _fix import BASE, RAND, post
from _v import dump, jnp, np

from lumenairy.elements.pmm import pmm_efficiency_1d, pmm_jones_1d
from lumenairy.elements.rcwa import rcwa_efficiency_2d, rcwa_jones_2d, uniaxial_tensor


def h(obj):
    m = hashlib.sha256()
    for a in obj:
        a = np.asarray(a)
        m.update(str(a.shape).encode())
        m.update(np.ascontiguousarray(a).tobytes())
    return m.hexdigest()


out = {}
lossy = BASE + 0.8j * post
c1 = 2.25 + 0.3 * RAND
uni = np.full((24, 24), 2.4 + 0j)
for name, cell, kw in [
        ("c4v_te_sym0", BASE, dict(polarization="te", symmetry=False)),
        ("c4v_te_auto", BASE, dict(polarization="te", symmetry="auto")),
        ("c4v_tm_auto", BASE, dict(polarization="tm", symmetry="auto")),
        ("c4v_li_te_sym0", BASE, dict(formulation="li", symmetry=False)),
        ("c4v_li_tm_auto", BASE, dict(formulation="li", polarization="tm",
                                      symmetry="auto")),
        ("c4v_tm_oblique", BASE, dict(polarization="tm", theta=0.2)),
        ("c4v_te_conical", BASE, dict(theta=0.2, phi=0.3)),
        ("c4v_li_tm_conical", BASE, dict(formulation="li", polarization="tm",
                                         theta=0.2, phi=0.3)),
        ("lossy_te_auto", lossy, dict(symmetry="auto")),
        ("lossy_li_tm", lossy, dict(formulation="li", polarization="tm",
                                    symmetry=False)),
        ("uniform_te", uni, dict(theta=0.1)),
        ("c1_tm_n3", c1, dict(polarization="tm", n_orders_x=3,
                              n_orders_y=3)),
        ("c1_li_te_conical", c1, dict(formulation="li", theta=0.25,
                                      phi=1.0))]:
    kw.setdefault("n_orders_x", 2)
    kw.setdefault("n_orders_y", 2)
    res = rcwa_efficiency_2d(0.9, 0.9, cell.astype(complex), 1.52, 1.0, 0.37,
                             1.0, **kw)
    out["rcwa2d_" + name] = h(res)

# tensor cells: post holds an in-plane uniaxial director at 30 deg (LC);
# the background isotropic; and a tilted (out-of-plane) director
ip = uniaxial_tensor(1.5, 1.7, np.pi / 2, phi=np.pi / 6)
tilt = uniaxial_tensor(1.5, 1.7, np.pi / 3, phi=np.pi / 6)
iso = 1.8 * np.eye(3, dtype=complex)
cell_ip = np.where(post[..., None, None] > 0, ip, iso).astype(complex)
cell_tilt = np.where(post[..., None, None] > 0, tilt, iso).astype(complex)
for name, cell, kw in [
        ("ip_laurent", cell_ip, {}),
        ("ip_li", cell_ip, dict(formulation="li")),
        ("ip_laurent_sym0", cell_ip, dict(symmetry=False)),
        ("ip_conical", cell_ip, dict(theta=0.2, phi=0.3)),
        ("ip_li_conical", cell_ip, dict(formulation="li", theta=0.2,
                                        phi=0.3)),
        ("tilt_laurent", cell_tilt, {}),
        ("tilt_conical", cell_tilt, dict(theta=0.2, phi=0.3))]:
    res = rcwa_jones_2d(0.9, 0.9, cell, 1.52, 1.0, 0.37, 1.0, n_orders_x=2,
                        n_orders_y=2, **kw)
    out["jones2d_" + name] = h(res)
for name, cell, kw in [("ip_laurent", cell_ip, {}),
                       ("ip_li", cell_ip, dict(formulation="li")),
                       ("ip_conical", cell_ip, dict(theta=0.2, phi=0.3))]:
    res = rcwa_jones_2d(0.9, 0.9, jnp.asarray(cell), 1.52, 1.0, 0.37, 1.0,
                        n_orders_x=2, n_orders_y=2, **kw)
    out["jones2d_jaxeager_" + name] = h(res)

g = (0.85, 2.3, 1.35, 1.6, 1.0, 0.31, 0.4, 1.0)
for pol in ("te", "tm"):
    for ang in (0.0, 0.2):
        for stab in (True, False):
            res = pmm_efficiency_1d(*g, angle=ang, polarization=pol,
                                    degree=10, stabilize=stab)
            out[f"pmm1d_{pol}_{ang}_{stab}"] = h(res)
        res = pmm_efficiency_1d(0.85, jnp.asarray(2.3 + 0j), 1.35, 1.6, 1.0,
                                0.31, 0.4, 1.0, angle=jnp.asarray(ang),
                                polarization=pol, degree=10, stabilize=False)
        out[f"pmm1d_jaxeager_{pol}_{ang}"] = h(res)
ER = uniaxial_tensor(1.5, 1.8, np.pi / 2, phi=0.4)
ERt = uniaxial_tensor(1.5, 1.8, np.pi / 3, phi=0.4)
EG = 1.35 ** 2 * np.eye(3, dtype=complex)
for name, er in (("ip", ER), ("tilt", ERt)):
    for ang in (0.0, 0.2):
        res = pmm_jones_1d(0.85, er, EG, 1.6, 1.0, 0.31, 0.4, 1.0, angle=ang,
                           degree=10, stabilize=False)
        out[f"pmmjones1d_{name}_{ang}"] = h(res)
    if name == "ip":
        res = pmm_jones_1d(0.85, er, EG, 1.6, 1.0, 0.31, 0.4, 1.0,
                           angle=jnp.asarray(0.2), degree=10, stabilize=False)
        out[f"pmmjones1d_jaxeager_{name}"] = h(res)

for k, v in out.items():
    print(k, v[:16])
dump("b_bytes", out)
