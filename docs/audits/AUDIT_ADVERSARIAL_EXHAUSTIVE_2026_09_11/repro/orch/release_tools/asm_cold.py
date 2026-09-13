import sys, subprocess, os
sys.path.insert(0, ".")
os.environ["OPENBLAS_NUM_THREADS"] = "1"
# 1. mechanism: which scipy modules does the first ASM call pull in a fresh interpreter?
probe = r"""
import sys, numpy as np
import lumenairy
before = sorted(m for m in sys.modules if m.startswith('scipy'))
from lumenairy.propagators import angular_spectrum_propagate
E = np.ones((256, 256), dtype=np.complex128)
angular_spectrum_propagate(E, 1e-6, 1e-6, 1e-3)
after = sorted(m for m in sys.modules if m.startswith('scipy'))
print('scipy modules after import lumenairy:', len(before), [m for m in before if m.count('.') <= 1][:8])
print('scipy top-level packages pulled by the first ASM call:', sorted({m.split('.')[1] for m in after if m not in before and m.count('.') >= 1}))
"""
r = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=300)
print(r.stdout.strip() or r.stderr[-1500:])
# 2. the cold peak model, the test's own method
from tests.unit.test_niche_audit_w3_infra import _measure_asm_peak
MiB = 1024 * 1024
rows = {}
for dtype in ("complex128", "complex64"):
    for n in (256, 512, 1024, 2048):
        cold, steady, est = _measure_asm_peak(n, dtype)
        rows[(dtype, n)] = cold
        print(f"{dtype:10s} N={n:5d}  cold={cold/MiB:8.2f} MiB  steady={steady/MiB:8.2f}  est={est/MiB:8.2f}  est/cold={est/cold:.3f}")
    ns = (256, 512, 1024, 2048)
    for a, b in zip(ns, ns[1:]):
        s = (rows[(dtype, b)] - rows[(dtype, a)]) / (b*b - a*a)
        F = rows[(dtype, a)] - s * a * a
        print(f"   pair {a:5d}->{b:5d}: slope {s:6.2f} B/px  fixed {F/MiB:6.2f} MiB")
