#!/usr/bin/env bash
# VERIFY-WP-C2 ROUND 2 -- mutants for the SEVEN entry points that
# ``test_c2_none_stamps_nothing_on_the_entry_points`` does not parametrize.
#
# That arm is the only IN-PROCESS pin on "the forwarded keyword actually
# reaches the internal trace", and it covers nine of the sixteen.  These
# mutants ask what happens when the forward is dropped in one of the other
# seven: the keyword is still in the signature (so the census passes) and
# still accepted (so nothing raises), it simply goes nowhere.
#
# M8   apply_real_lens_traced            -- forward dropped
# M9   fit_canonical_polynomials         -- forward dropped
# M10  eval_image_plane_wfe              -- forward dropped
# M11  caustic_diagnostic                -- forward dropped
#
# Usage:  bash vr2_mutants2.sh <win|wsl>
set -u
BUILD="${1:-win}"
SRC=/c/tmp/lum_vc2b
MUT=/c/tmp/lum_vc2b_mut2
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

if [ "$BUILD" = "wsl" ]; then
  RUN() { wsl -e bash -lc "cd /mnt/c/tmp/lum_vc2b_mut2 && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=/mnt/c/tmp/lum_vc2b_mut2 ~/lumvenv/bin/python -m pytest $* -q -p no:randomly --capture=sys -rf 2>&1 | tail -12"; }
else
  RUN() { (cd "$MUT" && PYTHONPATH="$MUT" python -m pytest $* -q -p no:randomly --capture=sys -rf 2>&1 | tail -12); }
fi

fresh() {
  rm -rf "$MUT"; mkdir -p "$MUT"
  git -C "$SRC" archive feat/c2-analytic-normal-round2 | tar -x -C "$MUT"
}

drop() {   # drop() <relative file>
  python - "$MUT" "$1" <<'PY'
import pathlib, sys
p = pathlib.Path(sys.argv[1]) / sys.argv[2]
t = p.read_text(encoding='cp1252')
import re
new, n = re.subn(r'\*\*_way_back_kwargs\(renormalize,\s*\n?\s*sphere_normal\)',
                 '**_way_back_kwargs()', t, count=1)
assert n == 1, ('anchor missing in %s' % sys.argv[2])
p.write_text(new, encoding='cp1252')
print('dropped the forward in', sys.argv[2])
PY
}

T="tests/unit/test_c2_analytic_normal_default.py tests/unit/test_verify_c2_analytic_normal.py"

for spec in \
  "M8:lumenairy/elements/_lens_traced.py" \
  "M9:lumenairy/propagators/asymptotic_canonical_fit.py" \
  "M10:lumenairy/analysis/image_plane_wfe.py" \
  "M11:lumenairy/analysis/aberration.py" ; do
  name="${spec%%:*}"; file="${spec#*:}"
  echo; echo "=================== $name  $file ==================="
  fresh
  drop "$file"
  RUN $T
done
echo
echo "=================== done ==================="
