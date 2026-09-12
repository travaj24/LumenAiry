"""Assemble the consolidated audit report: curated Markdown sections + every partition report,
rendered client-side with marked.js inside a designed HTML shell.  Output: lumenairy_audit.html"""
import json, os, re, glob, html
ROOT = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(ROOT, 'reports')

def read(name):
    p = os.path.join(R, name)
    return open(p, encoding='utf-8').read() if os.path.exists(p) else ''

curated_order = [
    'CONSOLIDATED_HEADER.md',
    'CONSOLIDATED_SUMMARY.md',
    'CONSOLIDATED_PART_LENS.md',
    'CONSOLIDATED_PART_LENS_CORE.md',
    'CONSOLIDATED_PART_LENS_ADD.md',
    'CONSOLIDATED_PART_LENS_ADD2.md',
    'CONSOLIDATED_PART_CARRIER_PROP.md',
    'CONSOLIDATED_PART_PROP_HF.md',
    'CONSOLIDATED_PART_PROP_HF2.md',
    'CONSOLIDATED_PART_RT_ELEM_UI.md',
    'CONSOLIDATED_PART_REST.md',
    'CONSOLIDATED_PART_REST2.md',
    'CONSOLIDATED_PART_REST3.md',
    'CONSOLIDATED_PART_PMM2D.md',
    'CONSOLIDATED_PART_RCWA.md',
    'CONSOLIDATED_PART_TESTS.md',
    'CONSOLIDATED_CROSSCUT.md',
]
curated = '\n\n'.join(read(n) for n in curated_order if read(n))

# partition reports (full text) in a sensible order
part_order = ['ORCHESTRATOR', 'RL-CORE', 'RL-MODELS', 'TR-INFRA', 'TR-MAIN-1', 'TR-MAIN-2', 'TR-SIBLINGS',
              'MASLOV-GBD-FGA', 'CARRIER', 'PROP-CORE', 'PROP-CORE-SUB1', 'PROP-HF', 'ASYMPTOTIC', 'RAYTRACE',
              'THIN-ELEMENTS-GLASS', 'ANALYSIS', 'PMM-1D', 'PMM-2D', 'RCWA-EME-BOR',
              'POLAR-SOURCES-INFRA', 'IO-OPTIMIZE', 'UI', 'TESTS-ARCH']
parts = []
for tag in part_order:
    txt = read(tag + '.md')
    if txt:
        parts.append((tag, txt))

def md_script(id_, text):
    # embed markdown safely inside a <script type="text/markdown"> block
    safe = text.replace('</script', '<\\/script')
    return f'<script type="text/markdown" id="{id_}">{safe}</script>'

toc_curated = [
    ('summary', 'Executive summary'),
    ('lens', 'apply_real_lens family'),
    ('carrier', 'Carrier chain & propagators'),
    ('raytrace', 'Ray tracing, elements, UI'),
    ('rest', 'Analysis, asymptotics, sources, I/O'),
    ('solvers', 'Rigorous solvers: PMM, RCWA, EME, BOR'),
    ('tests', 'Tests, CI & architecture'),
    ('crosscut', 'Cross-cutting themes & recommendations'),
]

appendix_items = ''.join(
    f'<details class="rep" id="rep-{tag}"><summary><span class="tag">{tag}</span> full auditor report</summary>'
    f'<div class="md" data-src="md-{tag}"></div></details>' for tag, _ in parts)
appendix_scripts = ''.join(md_script(f'md-{tag}', txt) for tag, txt in parts)
appendix_toc = ''.join(f'<li><a href="#rep-{tag}">{tag}</a></li>' for tag, _ in parts)

page = f'''<title>Lumenairy Adversarial Audit</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&family=IBM+Plex+Serif:wght@500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root {{
  --bg:#f4f6f7; --panel:#ffffff; --ink:#18212b; --ink-2:#4a5866; --ink-3:#7b8794; --rule:#d7dde3;
  --accent:#1f6f8b; --accent-soft:#e3eff3; --code-bg:#eef2f4;
  --p0:#b3261e; --p1:#c25e12; --p2:#9a7b00; --p3:#5f6b78; --ok:#2e7d4f;
  --p0-bg:#fbe9e7; --p1-bg:#fdf0e4; --p2-bg:#faf3d6; --p3-bg:#eceff2; --ok-bg:#e6f3ea;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg:#0f151b; --panel:#151d25; --ink:#e6ebef; --ink-2:#aab6c2; --ink-3:#7f8b98; --rule:#2a3542;
  --accent:#6fb7d0; --accent-soft:#16303a; --code-bg:#1c2630;
  --p0:#ff7b6f; --p1:#f0a35a; --p2:#e0c85a; --p3:#a3afbb; --ok:#6fcf97;
  --p0-bg:#3a1a17; --p1-bg:#3a2712; --p2-bg:#332e10; --p3-bg:#222b34; --ok-bg:#16301f;
}} }}
:root[data-theme="dark"] {{
  --bg:#0f151b; --panel:#151d25; --ink:#e6ebef; --ink-2:#aab6c2; --ink-3:#7f8b98; --rule:#2a3542;
  --accent:#6fb7d0; --accent-soft:#16303a; --code-bg:#1c2630;
  --p0:#ff7b6f; --p1:#f0a35a; --p2:#e0c85a; --p3:#a3afbb; --ok:#6fcf97;
  --p0-bg:#3a1a17; --p1-bg:#3a2712; --p2-bg:#332e10; --p3-bg:#222b34; --ok-bg:#16301f;
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font:15px/1.55 "IBM Plex Sans", system-ui, -apple-system, Segoe UI, sans-serif; }}
a {{ color:var(--accent); }}
.shell {{ display:grid; grid-template-columns: 240px minmax(0,1fr); gap:0; min-height:100vh; }}
nav {{ position:sticky; top:0; height:100vh; overflow:auto; padding:28px 20px; border-right:1px solid var(--rule); background:var(--panel); }}
nav h2 {{ font:600 11px/1.2 "IBM Plex Mono", monospace; letter-spacing:.12em; text-transform:uppercase; color:var(--ink-3); margin:22px 0 8px; }}
nav ul {{ list-style:none; padding:0; margin:0; }}
nav li {{ margin:0; }}
nav a {{ display:block; padding:5px 8px; border-radius:6px; color:var(--ink-2); text-decoration:none; font-size:13.5px; }}
nav a:hover, nav a:focus-visible {{ background:var(--accent-soft); color:var(--ink); outline:none; }}
main {{ padding:40px 56px 80px; max-width: 1180px; }}
.hero {{ border-bottom:1px solid var(--rule); padding-bottom:22px; margin-bottom:28px; }}
.hero .eyebrow {{ font:500 12px/1.2 "IBM Plex Mono", monospace; letter-spacing:.14em; text-transform:uppercase; color:var(--accent); }}
.hero h1 {{ font:600 34px/1.15 "IBM Plex Serif", Georgia, serif; margin:10px 0 8px; text-wrap:balance; }}
.hero p {{ color:var(--ink-2); max-width:70ch; margin:6px 0; }}
.kpis {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap:12px; margin:18px 0 4px; }}
.kpi {{ background:var(--panel); border:1px solid var(--rule); border-radius:8px; padding:12px 14px; }}
.kpi .n {{ font:600 26px/1 "IBM Plex Mono", monospace; font-variant-numeric: tabular-nums; }}
.kpi .l {{ color:var(--ink-3); font-size:12.5px; margin-top:6px; }}
.kpi.p0 .n {{ color:var(--p0); }} .kpi.p1 .n {{ color:var(--p1); }} .kpi.p2 .n {{ color:var(--p2); }} .kpi.p3 .n {{ color:var(--p3); }} .kpi.ok .n {{ color:var(--ok); }}
.md h1 {{ font:600 28px/1.2 "IBM Plex Serif", Georgia, serif; margin:36px 0 12px; text-wrap:balance; }}
.md h2 {{ font:600 22px/1.25 "IBM Plex Serif", Georgia, serif; margin:44px 0 12px; padding-top:14px; border-top:1px solid var(--rule); text-wrap:balance; }}
.md h3 {{ font:600 17px/1.3 "IBM Plex Sans", sans-serif; margin:28px 0 8px; }}
.md h4 {{ font:600 15px/1.3 "IBM Plex Sans", sans-serif; margin:20px 0 6px; color:var(--ink-2); }}
.md p, .md li {{ max-width: 78ch; }}
.md code {{ font:13px/1.4 "IBM Plex Mono", Menlo, Consolas, monospace; background:var(--code-bg); padding:1px 5px; border-radius:4px; }}
.md pre {{ background:var(--code-bg); border:1px solid var(--rule); border-radius:8px; padding:12px 14px; overflow-x:auto; font:12.5px/1.45 "IBM Plex Mono", Menlo, Consolas, monospace; }}
.md pre code {{ background:none; padding:0; }}
.md .tbl {{ overflow-x:auto; margin:12px 0 18px; border:1px solid var(--rule); border-radius:8px; }}
.md table {{ border-collapse:collapse; width:100%; font-size:13.5px; }}
.md th, .md td {{ text-align:left; vertical-align:top; padding:8px 10px; border-bottom:1px solid var(--rule); }}
.md th {{ background:var(--panel); font-weight:600; color:var(--ink-2); font-size:12.5px; letter-spacing:.02em; position:sticky; top:0; }}
.md tr:last-child td {{ border-bottom:none; }}
.md td:first-child {{ font-family:"IBM Plex Mono", monospace; font-size:12.5px; white-space:nowrap; color:var(--ink-3); }}
.md blockquote {{ margin:12px 0; padding:8px 14px; border-left:3px solid var(--accent); color:var(--ink-2); background:var(--accent-soft); border-radius:0 6px 6px 0; }}
.sev {{ display:inline-block; font:600 11px/1 "IBM Plex Mono", monospace; letter-spacing:.06em; padding:4px 7px; border-radius:5px; white-space:nowrap; }}
.sev.p0 {{ color:var(--p0); background:var(--p0-bg); }} .sev.p1 {{ color:var(--p1); background:var(--p1-bg); }}
.sev.p2 {{ color:var(--p2); background:var(--p2-bg); }} .sev.p3 {{ color:var(--p3); background:var(--p3-bg); }}
.vfy {{ display:inline-block; font:600 11px/1 "IBM Plex Mono", monospace; color:var(--ok); background:var(--ok-bg); padding:4px 6px; border-radius:5px; margin-left:4px; }}
details.rep {{ border:1px solid var(--rule); border-radius:8px; margin:12px 0; background:var(--panel); }}
details.rep > summary {{ cursor:pointer; padding:12px 16px; font-weight:500; list-style:none; }}
details.rep > summary::-webkit-details-marker {{ display:none; }}
details.rep > summary::before {{ content:"▸"; display:inline-block; width:18px; color:var(--ink-3); transition: transform .15s; }}
details.rep[open] > summary::before {{ transform:rotate(90deg); }}
details.rep .md {{ padding:0 22px 20px; }}
.tag {{ font:600 12px/1 "IBM Plex Mono", monospace; letter-spacing:.06em; color:var(--accent); background:var(--accent-soft); padding:4px 7px; border-radius:5px; margin-right:8px; }}
.note {{ color:var(--ink-3); font-size:13px; }}
@media (max-width: 900px) {{ .shell {{ grid-template-columns: 1fr; }} nav {{ position:static; height:auto; border-right:none; border-bottom:1px solid var(--rule); }} main {{ padding:24px 18px 60px; }} }}
@media (prefers-reduced-motion: reduce) {{ * {{ transition:none !important; }} }}
</style>
<div class="shell">
<nav>
  <h2>Report</h2>
  <ul>{''.join(f'<li><a href="#{a}">{t}</a></li>' for a, t in toc_curated)}</ul>
  <h2>Auditor reports</h2>
  <ul>{appendix_toc}</ul>
</nav>
<main>
  <header class="hero">
    <div class="eyebrow">Adversarial audit · 2026-09-11 · v5.45.x @ a1ff1e6e</div>
    <h1>Lumenairy Adversarial Audit</h1>
    <p>Twenty-two Opus auditor runs (twenty partitions plus two follow-up sub-audits) covered the 219 K-line library; each defect below was substantiated against an independent oracle, and the orchestrator re-measured the highest-impact claims on its own fixtures before accepting them. Severity chips: P0 wrong physics/data on a default path · P1 opt-in wrong physics, crash or stale cache · P2 performance/fragility · P3 quality. <span class="vfy">✔ verified</span> marks orchestrator re-measurement.</p>
    <div class="kpis" id="kpis"></div>
  </header>
  <div class="md" id="curated" data-src="md-curated"></div>
  <h2 id="appendix" style="font:600 22px/1.25 'IBM Plex Serif', Georgia, serif; margin:44px 0 12px; padding-top:14px; border-top:1px solid var(--rule);">Appendix — full auditor reports</h2>
  <p class="note">Each report is reproduced verbatim as returned by its auditor (scope read, findings with evidence and repro-script paths, performance opportunities, alternatives, code-organization notes, unverified suspicions, and what was checked and found correct).</p>
  {appendix_items}
</main>
</div>
{md_script('md-curated', curated)}
{appendix_scripts}
<script src="https://cdnjs.cloudflare.com/ajax/libs/marked/12.0.2/marked.min.js"></script>
<script>
(function() {{
  var sevRe = /\\*\\*\\[?(P[0-3])\\]?\\*\\*|\\[(P[0-3])\\]|(^|[\\s|])(P[0-3])(?=[\\s|\\/,;.)])/g;
  function render(el) {{
    var src = document.getElementById(el.getAttribute('data-src'));
    if (!src) return;
    var md = src.textContent.replace(/<\\\\\\/script/g, '</script');
    var out = marked.parse(md, {{ gfm: true, breaks: false, mangle: false, headerIds: true }});
    // severity chips and verified marks (post-render, text only)
    out = out.replace(/<strong>\\[?(P[0-3])\\]?<\\/strong>/g, function(m, p) {{ return '<span class="sev ' + p.toLowerCase() + '">' + p + '</span>'; }});
    out = out.replace(/\\[(P[0-3])\\]/g, function(m, p) {{ return '<span class="sev ' + p.toLowerCase() + '">' + p + '</span>'; }});
    out = out.replace(/(<td>|<h3[^>]*>|<li>)\\s*(P[0-3])(\\s*(?:\\(|\\/|,|<))/g, function(m, a, p, b) {{ return a + '<span class="sev ' + p.toLowerCase() + '">' + p + '</span>' + b; }});
    out = out.replace(/✔( orchestrator-verified| verified)?/g, '<span class="vfy">✔ verified</span>');
    out = out.replace(/<table>/g, '<div class="tbl"><table>').replace(/<\\/table>/g, '</table></div>');
    el.innerHTML = out;
  }}
  document.querySelectorAll('.md[data-src]').forEach(render);
  // anchors for the curated sections
  var map = {{ 'summary': /^1\\./, 'lens': /^2\\./, 'carrier': /^3\\./, 'raytrace': /^4\\./, 'rest': /^8\\./, 'solvers': /^11\\./, 'tests': /^14\\./, 'crosscut': /^15\\./ }};
  var hs = document.querySelectorAll('#curated h2');
  hs.forEach(function(h) {{ for (var k in map) {{ if (map[k].test(h.textContent.trim()) && !document.getElementById(k)) {{ h.id = k; }} }} }});
  // KPI tiles from the summary counts if present
  var kp = document.getElementById('kpis');
  var counts = (document.getElementById('md-curated').textContent.match(/KPI:\\s*P0=(\\d+)\\s*P1=(\\d+)\\s*P2=(\\d+)\\s*P3=(\\d+)\\s*verified=(\\d+)/) || []);
  if (counts.length) {{
    var labels = [['p0','P0 findings'],['p1','P1 findings'],['p2','P2 findings'],['p3','P3 findings'],['ok','orchestrator-verified']];
    kp.innerHTML = labels.map(function(l, i) {{ return '<div class="kpi ' + l[0] + '"><div class="n">' + counts[i+1] + '</div><div class="l">' + l[1] + '</div></div>'; }}).join('');
  }}
}})();
</script>
'''
out = os.path.join(ROOT, 'lumenairy_audit.html')
open(out, 'w', encoding='utf-8').write(page)
print('wrote', out, len(page), 'bytes;', len(parts), 'partition reports embedded:', ', '.join(t for t, _ in parts))
