#!/usr/bin/env bash
# VERIFY-WP-C2 ROUND 2 -- the mutation matrix, re-run and EXTENDED.
#
# Each mutant is a fresh ``git archive`` of the round-2 tip with ONE edit,
# run in its own directory with its own PYTHONPATH.  The library is never
# edited in the verification worktree.
#
# M1  the round-2 census mutant: ``sphere_normal`` dropped from
#     ``ray_fan_data``'s SIGNATURE while its body still mentions it
#     (round 2 claims "3 failed" on both builds)
# M2  the keyword is threaded but DROPPED before the inner trace:
#     ``ray_fan_data`` forwards ``_way_back_kwargs()`` with no arguments
# M3  ``_way_back_kwargs`` maps ``None`` to ``'generic'`` instead of to
#     "name nothing"
# M4  ``_library_trace_default`` returns the WRONG route for
#     ``renormalize``
# M5  the JAX tracer GAINS the NumPy domain clamp
# M6  the NumPy tracer LOSES its domain clamp
# M7  a new exported entry point that traces with NO way back is added
#     (the census must name it)
#
# Usage:  bash vr2_mutants.sh <win|wsl> <outdir>
set -u
BUILD="${1:-win}"
OUT="${2:-/c/tmp/lum_vc2b/validation/probe_verify_c2_round2}"
SRC=/c/tmp/lum_vc2b
MUT=/c/tmp/lum_vc2b_mut
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

if [ "$BUILD" = "wsl" ]; then
  RUN() { wsl -e bash -lc "cd /mnt/c/tmp/lum_vc2b_mut && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=/mnt/c/tmp/lum_vc2b_mut ~/lumvenv/bin/python -m pytest $* -q -p no:randomly --capture=sys -rf 2>&1 | tail -25"; }
else
  RUN() { (cd "$MUT" && PYTHONPATH="$MUT" python -m pytest $* -q -p no:randomly --capture=sys -rf 2>&1 | tail -25); }
fi

fresh() {
  rm -rf "$MUT"; mkdir -p "$MUT"
  git -C "$SRC" archive feat/c2-analytic-normal-round2 | tar -x -C "$MUT"
}

banner() { echo; echo "=================== $1 ==================="; }

TESTS_CENSUS="tests/unit/test_c2_analytic_normal_default.py tests/unit/test_verify_c2_analytic_normal.py"
TESTS_ALL="$TESTS_CENSUS"

# ---------------------------------------------------------------- M0
banner "M0 control (no mutation)"
fresh
RUN $TESTS_ALL

# ---------------------------------------------------------------- M1
banner "M1 ray_fan_data loses sphere_normal from its SIGNATURE"
fresh
python - "$MUT" <<'PY'
import pathlib, sys
p = pathlib.Path(sys.argv[1]) / 'lumenairy' / 'raytrace' / 'ray_fan.py'
t = p.read_text(encoding='cp1252')
old = """def ray_fan_data(
    surfaces: List['Surface'],
    wavelength: float,
    semi_aperture: float,
    field_angle: float = 0.0,
    n_rays: int = 101,
    *,
    renormalize: Optional[str] = None,
    sphere_normal: Optional[str] = None,
)"""
new = """def ray_fan_data(
    surfaces: List['Surface'],
    wavelength: float,
    semi_aperture: float,
    field_angle: float = 0.0,
    n_rays: int = 101,
    *,
    renormalize: Optional[str] = None,
)"""
assert old in t, 'M1 anchor missing'
t = t.replace(old, new, 1)
t = t.replace("""        **_way_back_kwargs(renormalize, sphere_normal))""",
              """        **_way_back_kwargs(renormalize, None))""", 1)
p.write_text(t, encoding='cp1252')
print('M1 applied')
PY
RUN $TESTS_ALL

# ---------------------------------------------------------------- M2
banner "M2 keyword threaded but DROPPED before the inner trace"
fresh
python - "$MUT" <<'PY'
import pathlib, sys
p = pathlib.Path(sys.argv[1]) / 'lumenairy' / 'raytrace' / 'ray_fan.py'
t = p.read_text(encoding='cp1252')
old = "        **_way_back_kwargs(renormalize, sphere_normal))"
assert t.count(old) >= 4, t.count(old)
t = t.replace(old, "        **_way_back_kwargs())", 1)
p.write_text(t, encoding='cp1252')
print('M2 applied')
PY
RUN $TESTS_ALL

# ---------------------------------------------------------------- M3
banner "M3 _way_back_kwargs maps None to 'generic'"
fresh
python - "$MUT" <<'PY'
import pathlib, sys
p = pathlib.Path(sys.argv[1]) / 'lumenairy' / 'raytrace' / 'trace.py'
t = p.read_text(encoding='cp1252')
old = """    kwargs = {}
    if renormalize is not None:
        kwargs['renormalize'] = renormalize
    if sphere_normal is not None:
        kwargs['sphere_normal'] = sphere_normal
    return kwargs"""
new = """    kwargs = {}
    kwargs['renormalize'] = ('surface' if renormalize is None
                             else renormalize)
    kwargs['sphere_normal'] = ('generic' if sphere_normal is None
                               else sphere_normal)
    return kwargs"""
assert old in t, 'M3 anchor missing'
p.write_text(t.replace(old, new, 1), encoding='cp1252')
print('M3 applied')
PY
RUN $TESTS_ALL

# ---------------------------------------------------------------- M4
banner "M4 _library_trace_default returns the wrong route for renormalize"
fresh
python - "$MUT" <<'PY'
import pathlib, sys
p = pathlib.Path(sys.argv[1]) / 'lumenairy' / 'raytrace' / 'trace.py'
t = p.read_text(encoding='cp1252')
old = """    import inspect
    value = inspect.signature(trace).parameters[keyword].default
    _LIBRARY_TRACE_DEFAULTS[keyword] = value
    return value"""
new = """    import inspect
    value = inspect.signature(trace).parameters[keyword].default
    if keyword == 'renormalize':
        value = 'surface'
    _LIBRARY_TRACE_DEFAULTS[keyword] = value
    return value"""
assert old in t, 'M4 anchor missing'
p.write_text(t.replace(old, new, 1), encoding='cp1252')
print('M4 applied')
PY
RUN $TESTS_ALL

# ---------------------------------------------------------------- M5
banner "M5 the JAX tracer GAINS the NumPy domain clamp"
fresh
python - "$MUT" <<'PY'
import pathlib, sys
p = pathlib.Path(sys.argv[1]) / 'lumenairy' / 'raytrace' / 'jax_trace.py'
t = p.read_text(encoding='cp1252')
anchor = ("        # Pure spherical: outward unit normal is "
          "(x, y, z - R) / R.\n"
          "        nx = state.x / R_safe\n")
assert anchor in t, 'M5 anchor missing'
repl = ("        # M5 MUTANT: the NumPy domain clamp, added to JAX.\n"
        "        _h2 = (state.x * state.x + state.y * state.y) / "
        "(R_safe * R_safe)\n"
        "        nx = jnp.where(_h2 >= 0.9999, jnp.nan, state.x / R_safe)\n")
t = t.replace(anchor, repl, 1)
p.write_text(t, encoding='cp1252')
print('M5 applied')
PY
RUN $TESTS_ALL

# ---------------------------------------------------------------- M6
banner "M6 the NumPy tracer LOSES its domain clamp"
fresh
python - "$MUT" <<'PY'
import pathlib, sys
p = pathlib.Path(sys.argv[1]) / 'lumenairy' / 'raytrace' / 'surface.py'
t = p.read_text(encoding='cp1252')
old = 'valid = norm < 0.9999'
n = t.count(old)
assert n == 3, n
t = t.replace(old, 'valid = norm < 1.0e9')
p.write_text(t, encoding='cp1252')
print('M6 applied, replacements:', n)
PY
RUN $TESTS_ALL

# ---------------------------------------------------------------- M7
banner "M7 a NEW exported entry point that traces with no way back"
fresh
python - "$MUT" <<'PY'
import pathlib, sys
root = pathlib.Path(sys.argv[1])
p = root / 'lumenairy' / 'raytrace' / 'ray_fan.py'
t = p.read_text(encoding='cp1252')
t += '''


def spot_centroid_quick(surfaces, wavelength, semi_aperture,
                        field_angle=0.0, n_rays=21):
    """A new exported entry point that traces and offers NO way back."""
    import numpy as _np
    from .trace import _make_bundle, trace as _tr
    h = _np.linspace(-semi_aperture, semi_aperture, n_rays)
    z = _np.zeros(n_rays)
    rays = _make_bundle(h, z.copy(), z + _np.sin(field_angle),
                        z.copy(), wavelength)
    res = trace(rays, surfaces, wavelength, output_filter='last')
    return float(_np.mean(res.image_rays.x))
'''
p.write_text(t, encoding='cp1252')
init = root / 'lumenairy' / 'raytrace' / '__init__.py'
it = init.read_text(encoding='cp1252')
it += '\nfrom .ray_fan import spot_centroid_quick as spot_centroid_quick\n'
init.write_text(it, encoding='cp1252')
print('M7 applied')
PY
RUN $TESTS_ALL

echo
echo "=================== done ==================="
