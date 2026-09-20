"""VERIFY-WAVE5-HYGIENE2 round 2 -- the verifier's own mutation matrix.

Each arm edits ONE shipped expression in a scratch export of the merged tip
(never the worktree) and runs a selection of ids against it.  An arm that
leaves its selection green is a gate the round does not have.

    python vh3_mutate.py <base-tree> <out.json> [arm ...]

``<base-tree>`` is a ``git archive`` export of the tip holding ``lumenairy/``
and ``tests/``; the pristine copies of the three touched files sit one level
above it as ``carrier.pristine.py`` etc.  Nothing here imports lumenairy, so
the harness cannot be confused about which tree it edited.
"""
import json
import os
import subprocess
import sys

# --------------------------------------------------------------------------
# The arms.  (file, [(old, new), ...], selection, what it breaks)
# --------------------------------------------------------------------------
KERNEL_TAIL = ("        phase -= root0\n        return phase\n")
KERNEL_END = ("        phase += (L * qx[None, :] + M * qy[:, None]) / Nz"
              "         # (s.q)/N\n    return phase\n")

ARMS = {
    # ---- the consolidated kernel itself ---------------------------------
    'M1_kernel_negated': (
        'carrier.py',
        [(KERNEL_TAIL, "        phase -= root0\n        return -phase\n"),
         (KERNEL_END, "        phase += (L * qx[None, :] + M * qy[:, None])"
                      " / Nz         # (s.q)/N\n    return -phase\n")],
        'the ONE kernel returns the conjugate phase at all three sites'),
    'M2_linear_term_dropped': (
        'carrier.py',
        [("    if L or M:\n        phase += (L * qx[None, :] + "
          "M * qy[:, None]) / Nz         # (s.q)/N\n",
          "    if False:\n        phase += (L * qx[None, :] + "
          "M * qy[:, None]) / Nz         # (s.q)/N\n")],
        'the (s.q)/N chief-ray subtraction is dropped (tilted legs only)'),
    'M3_one_site_conjugated': (
        'carrier.py',
        [("    phase = _exact_dispersion_phase(qx, qy, k, tilt, bld,\n"
          "                                    "
          "'_collins_exact_kernel_correction')\n",
          "    phase = -_exact_dispersion_phase(qx, qy, k, tilt, bld,\n"
          "                                     "
          "'_collins_exact_kernel_correction')\n")],
        "ONE call site conjugates the kernel -- the shape the consolidation "
        "is supposed to make impossible"),
    'M4_root0_not_subtracted': (
        'carrier.py',
        [(KERNEL_TAIL, "        return phase\n"),
         ("    phase = bld.sqrt(rad)\n    phase -= root0\n",
          "    phase = bld.sqrt(rad)\n")],
        'the q = 0 value k N is left in (double-counts the piston)'),
    'M5_evanescent_clamp_removed': (
        'carrier.py',
        [("        np.maximum(phase, 0.0, out=phase)\n"
          "        np.sqrt(phase, out=phase)\n",
          "        np.sqrt(phase, out=phase)\n"),
         ("    bld.maximum(rad, 0.0, out=rad)\n    phase = bld.sqrt(rad)\n",
          "    phase = bld.sqrt(rad)\n")],
        'the evanescent band is no longer clamped to zero'),
    'M6_tilt_guard_weakened': (
        'carrier.py',
        [("    if not (s2 < 1.0):\n", "    if not (s2 <= 1.0):\n")],
        'the |s|^2 < 1 refusal admits the grazing |s| = 1 direction'),
    'M13_complex64_fold_removed': (
        'carrier.py',
        [("        ph = bld.mod(arg, 2.0 * np.pi)\n", "        ph = arg\n")],
        'the complex64 mod-2pi fold before the float32 cast is dropped'),
    # ---- the V-D3 public Collins leg ------------------------------------
    'M10_public_leg_demotes_again': (
        'carrier.py',
        [("f\"outside the trace.\")\n    env_a = xp.asarray(env)\n",
          "f\"outside the trace.\")\n    env_a = np.asarray(env)\n")],
        'the public leg demotes an eager JAX array to host NumPy again'),
    'M11_input_box_host_fft': (
        'carrier.py',
        [("        xp, is_jax, _bld = _backend_of(env)\n"
          "        fft2, _ifft2 = _fft2_pair(xp, is_jax)\n"
          "        spectrum = fft2(_as_c_order(env, np.complex128, xp))\n",
          "        from .fft_infra import _fft2\n"
          "        spectrum = _fft2(np.ascontiguousarray("
          "env, dtype=np.complex128))\n")],
        'the input-box transform goes back to a host FFT'),
    'M12_traced_refusal_removed': (
        'carrier.py',
        [("    xp, _is_jax, _bld = _backend_of(env)\n    if _is_traced(env):\n",
          "    xp, _is_jax, _bld = _backend_of(env)\n"
          "    if False and _is_traced(env):\n")],
        'the designed traced ValueError on the public leg is removed'),
    # ---- the V-D5 warning threshold -------------------------------------
    'M7_budget_back_to_1e15': (
        '_bluestein.py',
        [("_PHASE_BUDGET_MAX = 1e-6 / _EPS64",
          "_PHASE_BUDGET_MAX = 1e15  #")],
        'the chirp phase-budget threshold reverts to the historical 1e15'),
    'M8_budget_a_decade_low': (
        '_bluestein.py',
        [("_PHASE_BUDGET_MAX = 1e-6 / _EPS64",
          "_PHASE_BUDGET_MAX = 4.5e8  #")],
        'the threshold drops a decade below its derivation (the other side)'),
    'M14_budget_guard_deleted': (
        '_bluestein.py',
        [("    if phase_budget > _PHASE_BUDGET_MAX:\n",
          "    if False and phase_budget > _PHASE_BUDGET_MAX:\n")],
        'the phase-budget warning never fires at all'),
    # ---- the near-focus tau switch --------------------------------------
    'M9_tau_armed_at_1e_4': (
        'carrier.py',
        [("_GAP_KERNEL_ACCURACY_TAU = None",
          "_GAP_KERNEL_ACCURACY_TAU = 1e-4")],
        'the near-focus accuracy rule ships ARMED at tau = 1e-4'),
    'M15_tau_honours_explicit_exact': (
        'carrier.py',
        [("            if dep > float(_GAP_KERNEL_ACCURACY_TAU) \\\n"
          "                    and _kernel_asked != 'exact':\n",
          "            if dep > float(_GAP_KERNEL_ACCURACY_TAU):\n")],
        "an explicit gap_kernel='exact' would be silently downgraded when "
        "the rule is armed"),
}


def apply_arm(base, arm):
    fname, edits, _what = ARMS[arm]
    src = os.path.join(base, 'lumenairy', 'propagators', fname)
    pristine = os.path.join(os.path.dirname(base.rstrip('/\\')),
                            fname.replace('.py', '.pristine.py'))
    with open(pristine, encoding='cp1252') as fh:
        text = fh.read()
    for old, new in edits:
        n = text.count(old)
        if n != 1:
            raise SystemExit(f"{arm}: anchor occurs {n} times, not once:\n"
                             f"{old[:120]!r}")
        text = text.replace(old, new)
    with open(src, 'w', encoding='cp1252') as fh:
        fh.write(text)
    return src


def restore(base):
    for fname in ('carrier.py', '_bluestein.py', 'mft.py'):
        pristine = os.path.join(os.path.dirname(base.rstrip('/\\')),
                                fname.replace('.py', '.pristine.py'))
        with open(pristine, encoding='cp1252') as fh:
            text = fh.read()
        with open(os.path.join(base, 'lumenairy', 'propagators', fname), 'w',
                  encoding='cp1252') as fh:
            fh.write(text)


SELECTION = [
    'tests/unit/test_wave5_h2_collins_jax.py',
    'tests/unit/test_wave5_h2_near_focus_table.py',
    'tests/unit/test_wave5_h2_mft_direct.py',
    'tests/unit/test_verify_wave5_hyg2.py',
    'tests/unit/test_audit2609_b4_collins_transport.py',
]

#: The WSL lane's selection.  This box was running another agent's 4-worker
#: pytest sweep throughout, and under that contention the full selection above
#: advanced at ~6 % of one core: the b4 file's 2048x2048 gates (TestGateA/B/C/D)
#: alone are 235 s of the recorded 308 s.  They are dropped there, leaving the
#: three b4 classes that read the exact KERNEL -- which is what every arm in
#: this matrix touches.  V-D19's own numbers are taken on WSL by
#: ``vh3_v19_s4.py`` instead, which builds the same fixture without pytest.
SELECTION_LIGHT = [
    'tests/unit/test_wave5_h2_collins_jax.py',
    'tests/unit/test_wave5_h2_near_focus_table.py',
    'tests/unit/test_wave5_h2_mft_direct.py',
    'tests/unit/test_verify_wave5_hyg2.py',
    'tests/unit/test_audit2609_b4_collins_transport.py::TestSameTheorem',
    'tests/unit/test_audit2609_b4_collins_transport.py::TestKernelRefinement',
    'tests/unit/test_audit2609_b4_collins_transport.py'
    '::TestQuadratureComplementarity',
]


def run(base, selection, extra=()):
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', LUMENAIRY_MEM_BUDGET_MB='4096')
    cmd = [sys.executable, '-m', 'pytest', *selection, '-q', '--capture=sys',
           '-p', 'no:cacheprovider', '--no-header', *extra]
    p = subprocess.run(cmd, cwd=base, env=env, capture_output=True, text=True,
                       timeout=7200)
    tail = (p.stdout or '')[-4000:] + (p.stderr or '')[-1500:]
    return p.returncode, tail


def summarise(tail):
    for line in reversed(tail.splitlines()):
        s = line.strip()
        if ('passed' in s or 'failed' in s or 'error' in s
                or 'no tests ran' in s):
            return s
    return '(no summary line)'


def failing_ids(tail):
    out = []
    for line in tail.splitlines():
        if line.startswith('FAILED ') or line.startswith('ERROR '):
            out.append(line.split(' - ')[0].strip())
    return out


def main(base, outpath, *arms):
    arms = list(arms)
    sel = SELECTION
    if arms and arms[0] == '--light':
        sel = SELECTION_LIGHT
        arms = arms[1:]
    arms = arms or list(ARMS)
    results = {}
    restore(base)
    rc, tail = run(base, sel)
    results['_selection'] = list(sel)
    results['M0_pristine'] = {
        'what': 'the unmutated tip -- the control',
        'rc': rc, 'summary': summarise(tail), 'failed': failing_ids(tail)}
    print(f"M0_pristine: {results['M0_pristine']['summary']}")
    for arm in arms:
        restore(base)
        apply_arm(base, arm)
        rc, tail = run(base, sel)
        results[arm] = {'what': ARMS[arm][2], 'file': ARMS[arm][0],
                        'rc': rc, 'summary': summarise(tail),
                        'failed': failing_ids(tail)}
        print(f"{arm}: {results[arm]['summary']}  "
              f"({len(results[arm]['failed'])} ids)")
        sys.stdout.flush()
    restore(base)
    with open(outpath, 'w', encoding='cp1252') as fh:
        json.dump(results, fh, indent=1, sort_keys=True)
    print(f"-> {outpath}")


if __name__ == '__main__':
    main(*sys.argv[1:])
