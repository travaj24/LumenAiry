import sys, warnings, threading, time, numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
import lumenairy as la

print("=== A. deprecated_alias stacklevel: does the warning point at the USER? ===")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    try:
        la.load_zmx_prescription("does-not-exist.zmx")
    except Exception:
        pass
    for rec in w:
        if issubclass(rec.category, DeprecationWarning):
            print("  warning filename:", rec.filename.split("\\")[-1], " line:", rec.lineno)
            print("  message:", str(rec.message)[:100])
print("  (this script is p9_infra.py -- a correct stacklevel names THIS file)")

print("")
print("=== B. _deprecation._emit stacklevel arithmetic (synthetic) ===")
from lumenairy import _deprecation as D
def public_fn():
    D.warn_deprecated_kwarg('old', 'new', function='public_fn', version_removed='5.48')
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    public_fn()
    for rec in w:
        print("  direct-helper call -> filename %s line %d  (public_fn is defined at line 21 of this file)"
              % (rec.filename.split("\\")[-1], rec.lineno))
alias = D.deprecated_alias(lambda: None, old_name='old_thing')
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    alias()
    for rec in w:
        print("  deprecated_alias shim -> filename %s line %d" % (rec.filename.split("\\")[-1], rec.lineno))

print("")
print("=== C. deprecated_alias per-call cost ===")
f = lambda a, b: a + b
g = D.deprecated_alias(f, old_name='g')
warnings.simplefilter("ignore")
t0 = time.perf_counter()
for _ in range(20000):
    f(1, 2)
t1 = time.perf_counter()
for _ in range(20000):
    g(1, 2)
t2 = time.perf_counter()
print("  raw %.2f us/call ; aliased %.2f us/call ; overhead %.2f us"
      % ((t1 - t0) / 20000 * 1e6, (t2 - t1) / 20000 * 1e6, (t2 - t1 - (t1 - t0)) / 20000 * 1e6))
warnings.resetwarnings()

print("")
print("=== D. _check_2d_scalar_field: what it accepts/rejects ===")
from lumenairy._validation import _check_2d_scalar_field as chk
cases = [
    ("2-D complex128", np.zeros((4, 4), complex)),
    ("2-D float64 (REAL)", np.zeros((4, 4), float)),
    ("2-D int64", np.zeros((4, 4), np.int64)),
    ("2-D object", np.empty((4, 4), object)),
    ("2-D non-contiguous", np.zeros((8, 8), complex)[::2, ::2]),
    ("2-D bool", np.zeros((4, 4), bool)),
    ("1-D", np.zeros(4, complex)),
    ("3-D ensemble", np.zeros((3, 4, 4), complex)),
    ("nested list", [[1, 2], [3, 4]]),
    ("np.matrix", np.matrix(np.zeros((4, 4)))),
    ("0-d array", np.array(1.0)),
]
for name, obj in cases:
    try:
        chk(obj, 'probe', input_kind='field')
        print("  %-22s ACCEPTED" % name)
    except Exception as ex:
        print("  %-22s rejected: %s" % (name, type(ex).__name__))

print("")
print("=== E. lumenairy_context: restore on exception + nesting ===")
import numpy as _np
base = la.get_default_complex_dtype()
try:
    with la.lumenairy_context(complex_dtype=_np.complex64):
        assert _np.dtype(la.get_default_complex_dtype()) == _np.complex64
        with la.lumenairy_context(complex_dtype=_np.complex128):
            assert _np.dtype(la.get_default_complex_dtype()) == _np.complex128
        assert _np.dtype(la.get_default_complex_dtype()) == _np.complex64
        raise RuntimeError("boom")
except RuntimeError:
    pass
print("  restored after exception + nesting:", _np.dtype(la.get_default_complex_dtype()) == _np.dtype(base))

print("")
print("=== F. lumenairy_context thread-safety (two threads, opposing dtypes) ===")
observed = []
def worker(dt, n):
    for _ in range(n):
        with la.lumenairy_context(complex_dtype=dt):
            observed.append((dt, _np.dtype(la.get_default_complex_dtype())))
ts = [threading.Thread(target=worker, args=(_np.complex64, 200)),
      threading.Thread(target=worker, args=(_np.complex128, 200))]
for t in ts: t.start()
for t in ts: t.join()
bad = sum(1 for want, got in observed if _np.dtype(want) != got)
print("  observations: %d, mismatched inside the with-block: %d (%.1f%%)"
      % (len(observed), bad, 100.0 * bad / max(1, len(observed))))
print("  dtype after both threads:", _np.dtype(la.get_default_complex_dtype()),
      " (was", _np.dtype(base), ")")
la.set_default_complex_dtype(base)

print("")
print("=== G. cache registry: duplicate-name shadowing ===")
from lumenairy import _cache_registry as CR
calls = []
CR.register_cache_clearer('probe_dup', lambda: calls.append('first'))
CR.register_cache_clearer('probe_dup', lambda: calls.append('second'))
CR.clear_all_registered_caches()
print("  after clear_all, calls =", calls, "(second registration silently dropped)" if calls == ['first'] else "")
def boom(): raise ImportError("clearer broken")
CR.register_cache_clearer('probe_boom', boom)
CR.clear_all_registered_caches()
print("  a raising clearer is swallowed silently: no exception propagated")
CR._unregister_for_test('probe_dup'); CR._unregister_for_test('probe_boom')

print("")
print("=== H. ByteBudgetedLRU: nbytes of views, thread-safe get+put ===")
from lumenairy.cache import ByteBudgetedLRU, deep_nbytes
big = np.zeros((1024, 1024), np.float64)      # 8 MB
view = big[0:1, 0:1]
print("  deep_nbytes(view of 8MB base) = %d  (view.nbytes=%d)" % (deep_nbytes(view), view.nbytes))
print("  deep_nbytes((a, a)) double-count check:", deep_nbytes((big, big)), "vs single", deep_nbytes(big))
c = ByteBudgetedLRU('probe', max_bytes=4 * 1024 * 1024, register=False)
stored = c.put('k', big)
print("  put of an 8MB value under a 4MB ceiling stored?", stored, "(expect False)")

print("")
print("=== I. coatings polarization alias: reachable through propagate_through_system ===")
from lumenairy.propagators.system import propagate_through_system
E = np.ones((8, 8), complex)
outs = {}
for pol in ('s', 'p', 'te', 'tm'):
    el = {'type': 'coating', 'layers': [(1.38, 100e-9)], 'wavelength': 550e-9,
          'angle': np.radians(60), 'n_substrate': 1.52, 'polarization': pol,
          'port': 'transmission'}
    out = propagate_through_system(E, [el], dx=1e-6, wavelength=550e-9)
    val = np.abs(np.atleast_2d(out if not isinstance(out, tuple) else out[0])[0, 0])
    outs[pol] = float(val)
print("  |t| by polarization kwarg:", {k: round(v, 8) for k, v in outs.items()})
print("  te == s ?", abs(outs['te'] - outs['s']) < 1e-12,
      "  te == p ?", abs(outs['te'] - outs['p']) < 1e-12)
