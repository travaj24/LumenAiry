"""``LUMENAIRY_DISABLE_JAX=1`` forces the JAX backend OFF even when jax is
installed (2026-09-01: a Windows Application Control policy blocked jaxlib's
native DLL on a worker box, so ``find_spec('jax')`` said "available" while the
lazy import died mid-initialization and every later touch of the half-imported
module raised).  Two-sided: with the variable set the flag is False regardless
of the install; without it the flag equals the ``find_spec`` answer.  Each arm
runs in a FRESH interpreter (the flag is evaluated once at import)."""
import os
import subprocess
import sys

_PROBE = ("import importlib.util as u; from lumenairy.backend import array as a; "
          "print(int(a.JAX_AVAILABLE), int(u.find_spec('jax') is not None))")


def _run(env_extra):
    env = {k: v for k, v in os.environ.items() if k != "LUMENAIRY_DISABLE_JAX"}
    env.update(env_extra)
    out = subprocess.run([sys.executable, "-c", _PROBE], env=env,
                         capture_output=True, text=True, check=True).stdout
    flag, installed = (int(x) for x in out.split())
    return bool(flag), bool(installed)


def test_disable_jax_env_forces_flag_off():
    flag, _installed = _run({"LUMENAIRY_DISABLE_JAX": "1"})
    assert flag is False


def test_without_env_flag_tracks_install():
    flag, installed = _run({})
    assert flag == installed
