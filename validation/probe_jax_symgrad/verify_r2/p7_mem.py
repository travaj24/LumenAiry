"""Cost at a larger order count (the 26 GB observation of the first p2 run):
rcwa_efficiency_2d, 24 x 24 four-fold cross cell, 5 x 5 orders, jit(grad)
and jit(jacrev), rule ON vs OFF, each in a FRESH subprocess; peak RSS,
first call (trace + compile + run) and steady run time.
python p7_mem.py            -> driver
python p7_mem.py ON grad    -> one worker"""
import json
import os
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))

if len(sys.argv) == 1:
    from _vc import BUILD  # noqa: F401
    rows = []
    for mode in ("grad", "jacrev"):
        for on in ("OFF", "ON"):
            r = subprocess.run([sys.executable, os.path.join(HERE, "p7_mem.py"),
                                on, mode], capture_output=True, text=True,
                               stdin=subprocess.DEVNULL, timeout=3600,
                               env=dict(os.environ))
            line = [ln for ln in r.stdout.splitlines() if ln.startswith("{")]
            row = json.loads(line[-1]) if line else dict(
                on=on, mode=mode, err=r.stderr[-400:])
            print(row, flush=True)
            rows.append(row)
    json.dump({"build": BUILD, "rows": rows},
              open(os.path.join(HERE, f"p7_mem_{BUILD}.json"), "w"), indent=1)
    sys.exit(0)

on, mode = sys.argv[1], sys.argv[2]
import psutil  # noqa: E402

proc = psutil.Process()
peak = [0]
stop = [False]


def watch():
    while not stop[0]:
        peak[0] = max(peak[0], proc.memory_info().rss)
        time.sleep(0.05)


threading.Thread(target=watch, daemon=True).start()
from _vc import jax, jnp, np  # noqa: E402

from lumenairy.backend import set_jax_cluster_rule  # noqa: E402
from lumenairy.elements.rcwa import rcwa_efficiency_2d  # noqa: E402

set_jax_cluster_rule(on == "ON")
S = 24
CROSS = np.ones((S, S), complex)
CROSS[8:16, :] = 0.0
CROSS[:, 8:16] = 0.0
ARMX = np.zeros((S, S))
ARMX[8:16, 16:24] = 1.0
BASE = np.where(CROSS == 0.0, 4.0 + 0.3j, 1.44 + 0j)


def f(t):
    e = jnp.asarray(BASE) + t * jnp.asarray(ARMX)
    _o, R, T = rcwa_efficiency_2d(1.3, 1.3, e, 1.5, 1.0, 0.3, 1.0,
                                  n_orders_x=5, n_orders_y=5)
    return jnp.concatenate([R, T])


fn = (jax.jit(jax.grad(lambda t: jnp.sum(f(t) * jnp.arange(242.0))))
      if mode == "grad" else jax.jit(jax.jacrev(f)))
base_rss = proc.memory_info().rss
t0 = time.perf_counter()
jax.block_until_ready(fn(0.0))
first = time.perf_counter() - t0
ts = []
for _ in range(3):
    t0 = time.perf_counter()
    jax.block_until_ready(fn(0.0))
    ts.append(time.perf_counter() - t0)
stop[0] = True
print(json.dumps(dict(on=on, mode=mode, first_s=first, run_s=min(ts),
                      peak_rss_gb=peak[0] / 1e9,
                      rss_before_gb=base_rss / 1e9)))
