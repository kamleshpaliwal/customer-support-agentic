"""Build docs/progress.html from docs/progress.md:  python scripts/build_progress_html.py"""
import sys
import html, re, sys
src, out = (sys.argv[1:3] + ["docs/progress.md", "docs/progress.html"][len(sys.argv[1:3]):])
lines = open(src).read().splitlines()
def inline(t):
    t = html.escape(t)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", t)
    return t
def box(t):
    if t.startswith("[x]"): return '<span class="b done">Done</span> ', t[3:].strip()
    if t.startswith("[~]"): return '<span class="b part">Partial</span> ', t[3:].strip()
    if t.startswith("[ ]"): return '<span class="b todo">To do</span> ', t[3:].strip()
    return "", t
body, i, counts = [], 0, {"done": 0, "todo": 0}
while i < len(lines):
    l = lines[i]
    if l.startswith("# "): title = l[2:]; i += 1; continue
    if l.startswith("## "): body.append(f"<h2>{inline(l[3:])}</h2>")
    elif l.startswith("### "): body.append(f"<h3>{inline(l[4:])}</h3>")
    elif l.startswith("|"):
        rows = []
        while i < len(lines) and lines[i].startswith("|"):
            cells = [c.strip() for c in lines[i].strip("|").split("|")]
            if not all(set(c) <= set("-: ") for c in cells): rows.append(cells)
            i += 1
        h = "".join(f"<th>{inline(c)}</th>" for c in rows[0])
        trs = ""
        for r in rows[1:]:
            b, r[1] = box(r[1]); counts["todo" if "To do" in b else "done"] += 1
            r[1] = b + inline(r[1]); cells = [inline(r[0]), r[1]] + [inline(c) for c in r[2:]]
            trs += "<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"
        body.append(f"<div class='tw'><table><tr>{h}</tr>{trs}</table></div>"); continue
    elif l.startswith("- "):
        items = []
        while i < len(lines) and (lines[i].startswith("- ") or lines[i].startswith("      ")):
            if lines[i].startswith("- "): items.append(lines[i][2:])
            else: items[-1] += " " + lines[i].strip()
            i += 1
        lis = ""
        for t in items:
            b, t = box(t)
            if "Done" in b: counts["done"] += 1
            lis += f"<li>{b}{inline(t)}</li>"
        body.append(f"<ul>{lis}</ul>"); continue
    elif re.match(r"\d+\. ", l): body.append(f"<p class='step'>{inline(l)}</p>")
    elif l.strip(): body.append(f"<p>{inline(l)}</p>")
    i += 1
d, t = counts["done"], counts["todo"]
pct = round(100 * d / (d + t))
page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>
:root{{--bg:#fafaf8;--fg:#1d1d1b;--mut:#6b6b66;--card:#fff;--line:#e4e4de;--ok:#1a7f4b;--okbg:#e3f4ea;--td:#b45309;--tdbg:#fdf0dc;--pt:#1d4ed8;--ptbg:#e4ecfc;--code:#efefe9}}
@media(prefers-color-scheme:dark){{:root{{--bg:#16161a;--fg:#ececea;--mut:#9a9a94;--card:#1e1e23;--line:#33333a;--ok:#5fd394;--okbg:#17352a;--td:#f2b25c;--tdbg:#3a2c14;--pt:#8fb1ff;--ptbg:#1c2a4a;--code:#2a2a30}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,sans-serif}}
main{{max-width:980px;margin:0 auto;padding:32px 16px 64px}}
h1{{font-size:26px;margin:0 0 6px}}h2{{font-size:19px;margin:34px 0 8px;padding-top:10px;border-top:1px solid var(--line)}}h3{{font-size:15px;margin:22px 0 6px;color:var(--mut);text-transform:uppercase;letter-spacing:.04em}}
.bar{{height:10px;background:var(--line);border-radius:6px;overflow:hidden;margin:14px 0 4px}}.bar i{{display:block;height:100%;width:{pct}%;background:var(--ok)}}
.sum{{color:var(--mut);margin:0 0 8px}}
ul{{list-style:none;padding:0;margin:8px 0}}li{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px 12px;margin:6px 0}}
.tw{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;background:var(--card);border:1px solid var(--line);border-radius:8px}}
th,td{{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line);vertical-align:top}}th{{color:var(--mut);font-size:12px;text-transform:uppercase}}td:first-child{{white-space:nowrap;color:var(--mut)}}
.b{{display:inline-block;font-size:11px;font-weight:600;padding:1px 8px;border-radius:10px;margin-right:6px}}.done{{background:var(--okbg);color:var(--ok)}}.todo{{background:var(--tdbg);color:var(--td)}}.part{{background:var(--ptbg);color:var(--pt)}}
code{{background:var(--code);padding:1px 5px;border-radius:4px;font-size:13px}}.step{{margin:4px 0}}
</style></head><body><main><h1>{html.escape(title)}</h1>
<div class="bar"><i></i></div><p class="sum"><strong>{d}</strong> done · <strong>{t}</strong> to do · {pct}% complete</p>
{''.join(body)}</main></body></html>"""
open(out, "w").write(page)
print(d, t, pct)
