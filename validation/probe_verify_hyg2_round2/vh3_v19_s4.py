"""VERIFY-WAVE5-HYGIENE2 round 2 -- V-D19's premise/claim and the S4 bound.

 V1  The restated ``b4::TestGateCTwoGroupChain::
     test_the_two_transports_agree_on_this_chain``: its PREMISE (every Collins
     leg on ``'tf'``, and the quadrature margin ``max(K1, K3)``) and its CLAIM
     (``max|a-b|`` against ``1e3 eps peak``), re-measured on the fixture the
     test itself builds -- the fixture function is called directly, so the
     numbers are the id's own and not a re-implementation of it.
     Both sides of the bar are then probed: a disagreement of 8.5e-06 of peak
     (the quadrature-switch signature the id exists to catch) must FAIL it,
     and a last-bit re-association must PASS it.

 S4  The ``|P'''| <= 100 eps |P| / h^3`` bound, two-sided: the shipped
     quadratic merit must satisfy it at every rung, and a genuinely cubic
     control merit must FAIL it -- the second half is what stops the bound
     being vacuous.

    PYTHONPATH=<tree> python vh3_v19_s4.py OUT.json [--skip-v19]
"""
import json
import sys

import numpy as np

EPS = float(np.finfo(np.float64).eps)


def main(out_path, *flags):
    import lumenairy
    res = {'lumenairy_file': lumenairy.__file__,
           'python': sys.version.split()[0], 'platform': sys.platform}

    # ---------------------------------------------------------------- V1
    if '--skip-v19' not in flags:
        sys.path.insert(0, 'tests/unit')
        import test_audit2609_b4_collins_transport as B4
        arms = B4.p5_arms.__wrapped__()
        a = np.asarray(arms['sziklas_res'].field)
        b = np.asarray(arms['collins_res'].field)
        stages = [st for st in arms['collins_res'].stages
                  if st.get('collins_form')]
        forms = [st['collins_form'] for st in stages]
        margins = [max(max(st.get('collins_k1') or (0.0, 0.0)),
                       max(st.get('collins_k3') or (0.0, 0.0)))
                   for st in stages]
        k1s = [list(map(float, st.get('collins_k1') or ()))
               for st in stages]
        k3s = [list(map(float, st.get('collins_k3') or ()))
               for st in stages]
        peak = float(np.max(np.abs(a)))
        bar = 1e3 * EPS * peak
        got = float(np.max(np.abs(a - b)))
        # --- the two sides of the bar ---------------------------------
        b_switch = b + (8.5e-06 * peak) * np.exp(
            1j * np.linspace(0.0, 3.0, b.size).reshape(b.shape))
        got_switch = float(np.max(np.abs(a - b_switch)))
        # a LAST-BIT re-association: nudge every entry by one ULP
        b_ulp = (np.nextafter(b.real, np.inf)
                 + 1j * np.nextafter(b.imag, np.inf))
        got_ulp = float(np.max(np.abs(a - b_ulp)))
        res['V1'] = {
            'n_collins_stages': len(stages),
            'forms': forms, 'all_tf': bool(forms and set(forms) == {'tf'}),
            'k1': k1s, 'k3': k3s,
            'margins': [float(m) for m in margins],
            'min_margin': float(min(margins)) if margins else None,
            'premise_bar': 1.5,
            'premise_holds': bool(margins and min(margins) > 1.5),
            'peak': peak, 'bar': bar, 'max_abs_diff': got,
            'diff_over_peak': got / peak if peak else None,
            'claim_holds': got <= bar,
            'switch_signature_diff': got_switch,
            'switch_signature_fails_the_bar': got_switch > bar,
            'ulp_reassociation_diff': got_ulp,
            'ulp_reassociation_passes_the_bar': got_ulp <= bar,
            'decades_bar_to_switch': float(np.log10(got_switch / bar)),
        }
        print(f"V1 forms={set(forms)}  margins={[f'{m:.4f}' for m in margins]}"
              f"  peak={peak:.6e}")
        print(f"V1 max|a-b| = {got:.6e}  bar {bar:.6e}  claim "
              f"{res['V1']['claim_holds']}")
        print(f"V1 switch signature {got_switch:.4e} fails "
              f"{res['V1']['switch_signature_fails_the_bar']} "
              f"({res['V1']['decades_bar_to_switch']:.2f} decades over); "
              f"1-ULP {got_ulp:.4e} passes "
              f"{res['V1']['ulp_reassociation_passes_the_bar']}")

    # ---------------------------------------------------------------- S4
    sys.path.insert(0, 'tests/unit')
    import jax
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    import test_wave5_h2_collins_jax as CJ

    amp0 = jnp.asarray(np.real(CJ._gauss()))
    a0 = np.asarray(amp0)
    ij = np.unravel_index(int(np.argmax(a0)), a0.shape)

    quad = CJ._merit_factory(gap_kernel='fresnel',
                             on_collins_sampling='ignore')

    def cubic(a):
        return quad(a) ** 3

    out = {}
    for tag, merit in (('quadratic_shipped', quad), ('cubic_control', cubic)):
        P0 = float(merit(amp0))
        rows = []
        for h in (1e-1, 1e-2, 1e-3, 1e-4):
            def at(s, m=1, h=h, merit=merit):
                ap = np.array(a0)
                ap[ij] += s * m * h
                return float(merit(jnp.asarray(ap)))
            p3 = ((at(+1, 2) - 2 * at(+1) + 2 * at(-1) - at(-1, 2))
                  / (2.0 * h ** 3))
            floor = EPS * abs(P0) / h ** 3
            rows.append({'h': h, 'p3': p3, 'floor': floor,
                         'over_floor': abs(p3) / floor if floor else None,
                         'p3_is_exactly_zero': p3 == 0.0})
        out[tag] = {
            'P0': P0, 'rows': rows,
            'max_over_floor': max(r['over_floor'] for r in rows),
            'min_over_floor': min(r['over_floor'] for r in rows),
            'satisfies_100x_bound': all(r['over_floor'] <= 100.0
                                        for r in rows),
            'any_rung_exactly_zero': any(r['p3_is_exactly_zero']
                                         for r in rows),
        }
        print(f"S4 {tag}: over-floor "
              + "  ".join(f"h={r['h']:.0e}:{r['over_floor']:.3g}"
                          for r in rows)
              + f"   <=100x {out[tag]['satisfies_100x_bound']}")
    res['S4'] = out
    res['S4_two_sided'] = {
        'quadratic_within_bound': out['quadratic_shipped'][
            'satisfies_100x_bound'],
        'cubic_control_breaks_bound': not out['cubic_control'][
            'satisfies_100x_bound'],
        'separation_decades': float(np.log10(
            out['cubic_control']['min_over_floor']
            / max(out['quadratic_shipped']['max_over_floor'], 1e-300))),
    }
    print(f"S4 two-sided: {res['S4_two_sided']}")

    with open(out_path, 'w', encoding='cp1252') as fh:
        json.dump(res, fh, indent=1, sort_keys=True, default=str)
    print(f"lumenairy.__file__ = {lumenairy.__file__}")
    print(f"-> {out_path}")


if __name__ == '__main__':
    main(*sys.argv[1:])
