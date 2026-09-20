#!/usr/bin/env bash
# WP-C2 ROUND 3 -- the mutants this round's new ids have to catch.
#
# VR2-D4: before round 3, dropping the forward (``**_way_back_kwargs()``
# with no arguments) in any of these four left the WHOLE C2 suite green --
# 58 passed, 0 failed on each (VERIFY-WP-C2 round 2, M8-M11).  The keyword
# stayed in the signature, so the census stayed green; it simply went
# nowhere.  Each must now be caught by a NAMED id.
#
#   M8   elements/_lens_traced.py                  -> apply_real_lens_traced
#   M9   propagators/asymptotic_canonical_fit.py   -> fit_canonical_polynomials
#                                                     + fit_hf_polynomials
#   M10  analysis/image_plane_wfe.py               -> eval_image_plane_wfe
#   M11  analysis/aberration.py                    -> caustic_diagnostic
#   M12  elements/_lens_real.py                    -> apply_real_lens (the
#                                                     seventeenth, VR2-D1)
#
# VR2-D3: ``_library_trace_default`` was pinned for ``sphere_normal`` only,
# and a mutant that answers the WRONG route for ``renormalize`` passed the
# whole suite (58 passed).
#
#   M4   raytrace/trace.py::_library_trace_default -> wrong renormalize
#
# Usage:  bash r3_mutants.sh <win|wsl>
set -u
BUILD="${1:-win}"
SRC=/c/tmp/lum_c2c
MUT=/c/tmp/lum_c2c_mut
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

if [ "$BUILD" = "wsl" ]; then
  RUN() { wsl -e bash -lc "cd /mnt/c/tmp/lum_c2c_mut && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=/mnt/c/tmp/lum_c2c_mut ~/lumvenv/bin/python -m pytest $* -q -p no:randomly --capture=sys -rf 2>&1 | tail -14"; }
else
  RUN() { (cd "$MUT" && PYTHONPATH="$MUT" python -m pytest $* -q -p no:randomly --capture=sys -rf 2>&1 | tail -14); }
fi

fresh() {
  rm -rf "$MUT"; mkdir -p "$MUT"
  git -C "$SRC" archive feat/c2-analytic-normal-round3 | tar -x -C "$MUT"
}

drop() {   # drop() <relative file>
  python - "$MUT" "$1" <<'PY'
import pathlib, re, sys
p = pathlib.Path(sys.argv[1]) / sys.argv[2]
t = p.read_text(encoding='cp1252')
new, n = re.subn(r'\*\*_way_back_kwargs\(renormalize,\s*\n?\s*sphere_normal\)',
                 '**_way_back_kwargs()', t, count=1)
assert n == 1, ('anchor missing in %s' % sys.argv[2])
p.write_text(new, encoding='cp1252')
print('dropped the forward in', sys.argv[2])
PY
}

wrong_helper() {
  python - "$MUT" <<'PY'
import pathlib, sys
p = pathlib.Path(sys.argv[1]) / 'lumenairy/raytrace/trace.py'
t = p.read_text(encoding='cp1252')
old = "    value = inspect.signature(trace).parameters[keyword].default\n"
new = ("    value = inspect.signature(trace).parameters[keyword].default\n"
       "    if keyword == 'renormalize':\n"
       "        value = 'surface'\n")
assert t.count(old) == 1
p.write_text(t.replace(old, new, 1), encoding='cp1252')
print('_library_trace_default now answers the wrong renormalize route')
PY
}

T="tests/unit/test_c2_analytic_normal_default.py tests/unit/test_verify_c2_analytic_normal.py tests/unit/test_verify_c2_round2.py"

echo "=================== M0  control ==================="
fresh
RUN $T

for spec in \
  "M8:lumenairy/elements/_lens_traced.py" \
  "M9:lumenairy/propagators/asymptotic_canonical_fit.py" \
  "M10:lumenairy/analysis/image_plane_wfe.py" \
  "M11:lumenairy/analysis/aberration.py" \
  "M12:lumenairy/elements/_lens_real.py" ; do
  name="${spec%%:*}"; file="${spec#*:}"
  echo; echo "=================== $name  $file ==================="
  fresh
  drop "$file"
  RUN $T
done

echo; echo "=================== M4  _library_trace_default ==================="
fresh
wrong_helper
RUN $T

echo
echo "=================== done ==================="

# VR2-D5's own mutant (M13, the prior term dropped from the Gauss-Newton
# Hessian) lives in r3_w6a2_mutant.sh, because it has to be run twice --
# once under each bar -- against the same mutant tree.
