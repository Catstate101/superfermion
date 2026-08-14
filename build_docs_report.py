"""Build + execute the docs_verification notebook: runs every code snippet
from website/content/docs (== superfermion.com/docs) and reports pass/fail."""
import nbformat
from nbclient import NotebookClient
import sys

# Windows consoles default to cp1252 — UTF-8 diagram output would crash the echo below
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = r"c:\Users\ASUS\OneDrive\Desktop\sfdocs\superfermion"
NB_PATH = REPO + r"\notebooks\docs_verification.ipynb"

md_intro = """# Superfermion Docs — Snippet Verification Report

**Scope**: every code block in the documentation frontend (`website/content/docs/*.mdx`),
which is byte-identical to the live site **superfermion.com/docs** (parity spot-checked for
`/docs` and `/docs/guides/qec` while the dev server runs at http://localhost:3000).

**Environment**: Python 3.13 + `superfermion 0.1.5` editable install with the compiled Rust
core (`_sf_core`). Optional deps checked separately below.

**Method**: each Python block runs in a **fresh subprocess** (clean interpreter, only
`import superfermion as sf`, `import numpy as np`, `from math import pi` as prelude).
Blocks that fail standalone are re-run in a **sequential page-context pass** (all Python
blocks of the page execute top-to-bottom in one shared namespace, like a reader pasting
them in order). That separates *broken snippets* from *snippets that depend on earlier
cells of the same page*.

**Verdicts**
- `PASS` — ran standalone, exited 0, produced output (or cleanly did nothing).
- `CTX_PASS` — runs only with variables from earlier snippets on the same page.
- `SKIP` — shell/pseudo-code block, needs an optional dependency, or needs cloud credentials.
- `FAIL` — errors in both standalone and context runs: the doc snippet is broken."""

cell_inventory = """import re, pathlib, sys, subprocess, textwrap, time, json, os
from collections import Counter, defaultdict

WEBSITE_DOCS = pathlib.Path(r"c:\\Users\\ASUS\\OneDrive\\Desktop\\sfdocs\\superfermion\\website\\content\\docs")
REPO_ROOT = r"c:\\Users\\ASUS\\OneDrive\\Desktop\\sfdocs\\superfermion"
MDX_FILES = sorted([p for p in WEBSITE_DOCS.rglob("*.mdx") if p.suffix == ".mdx"])
print("Doc pages:", len(MDX_FILES))
for p in MDX_FILES:
    print("  -", p.relative_to(WEBSITE_DOCS))

BLOCK_RE = re.compile(r"```(\\w+)?\\n(.*?)```", re.DOTALL)
snippets = []
for p in MDX_FILES:
    text = p.read_text(encoding="utf-8")
    rel = str(p.relative_to(WEBSITE_DOCS)).replace("\\\\", "/")
    for i, m in enumerate(BLOCK_RE.finditer(text)):
        snippets.append({"page": rel, "idx": i, "lang": (m.group(1) or "text").lower(), "code": m.group(2).rstrip()})

langs = Counter(s["lang"] for s in snippets)
print()
print("Total code blocks:", len(snippets), dict(langs))
print()
for s in snippets:
    first = (s["code"].splitlines()[0] if s["code"] else "(empty)")[:70]
    print(f"{s['page']:<34} #{s['idx']:<2} {s['lang']:<8} {first}")"""

cell_deps = """optional_deps = ["qiskit", "qiskit_aer", "cirq", "pennylane", "torch", "jax", "flax",
                 "tensorflow", "pyscf", "scipy", "networkx", "braket", "boto3", "matplotlib"]
dep_status = {}
for name in optional_deps:
    try:
        __import__(name)
        dep_status[name] = "installed"
    except Exception as e:
        dep_status[name] = "MISSING"
print("Optional dependency availability:")
for k, v in dep_status.items():
    print(f"  {k:<14} {v}")
print()
print("superfermion version:", __import__("superfermion").__version__)"""

cell_harness = """PRELUDE = (
    "import warnings; warnings.filterwarnings('ignore')\\n"
    "import os\\nos.environ['MPLBACKEND'] = 'Agg'\\n"
    "import numpy as np\\n"
    "from math import pi\\n"
    "import superfermion as sf\\n"
)

def run_standalone(code):
    script = PRELUDE + "\\n" + code
    t0 = time.time()
    try:
        r = subprocess.run([sys.executable, "-c", script], capture_output=True,
                           encoding="utf-8", errors="replace", timeout=180, cwd=REPO_ROOT)
        out = (r.stdout or "").strip()
        err = (r.stderr or "").strip()
        status = "PASS" if r.returncode == 0 else "FAIL"
    except subprocess.TimeoutExpired:
        return {"status": "TIMEOUT", "out": "", "err": "killed after 180s"}
    except Exception as e:
        return {"status": "ERROR", "out": "", "err": repr(e)}
    return {"status": status, "out": out, "err": err}

py_snips = [s for s in snippets if s["lang"] == "python"]
print("Python blocks to run:", len(py_snips))

CRED_RUN = ["ibm_brisbane", "aria-1", "rigetti", "sv1", "tn1", "dm1", "ionq_harmony",
            "oqc_lucy", "api.example.com", "your-ibm-token", "your-ionq-key",
            "my-braket-results", "list_devices()"]

results = []
for sn in py_snips:
    code = sn["code"]
    if any(c in code for c in CRED_RUN):
        r = {"status": "SKIP", "out": "", "err": "", "reason": "requires cloud credentials / network call"}
    else:
        r = run_standalone(code)
    r.update(sn)
    results.append(r)
    print(f"[{r['status']:<7}] {r['page']} #{r['idx']}")"""

cell_context = """OPT_DEPS = ["qiskit", "cirq", "pennylane", "jax", "flax", "torch", "tensorflow",
             "pyscf", "networkx", "braket", "boto3"]

def classify(r):
    if r["status"] != "FAIL":
        return r
    err = (r["err"] or "").lower()
    if "modulenotfounderror" in err:
        for d in OPT_DEPS:
            if d.lower() in err:
                r["status"] = "SKIP"
                r["reason"] = "optional dependency missing: " + r["err"].splitlines()[-1][:100]
                return r
    return r

results = [classify(r) for r in results]
fail_ids = [i for i, r in enumerate(results) if r["status"] == "FAIL"]
print("Standalone failures to retry in page context:", len(fail_ids))

by_page = defaultdict(list)
for i, sn in enumerate(py_snips):
    by_page[sn["page"]].append(i)

for page in by_page:
    idxs = by_page[page]
    if not any(results[i]["status"] == "FAIL" for i in idxs):
        continue
    lines = ["import warnings; warnings.filterwarnings('ignore')",
             "import os\\nos.environ['MPLBACKEND'] = 'Agg'",
             "import numpy as np", "from math import pi", "import superfermion as sf",
             "_snip = {}"]
    for k, i in enumerate(idxs):
        body = py_snips[i]["code"]
        lines.append(
            "try:\\n"
            f"    exec({body!r}, globals())\\n"
            f"    _snip[{k}] = ('PASS', '')\\n"
            "except Exception as e:\\n"
            f"    _snip[{k}] = ('FAIL', type(e).__name__ + ': ' + str(e)[:300])"
        )
    lines.append("import json; print('CTXJSON:' + json.dumps(_snip))")
    script = "\\n".join(lines)
    r = subprocess.run([sys.executable, "-c", script], capture_output=True,
                       encoding="utf-8", errors="replace", timeout=600, cwd=REPO_ROOT)
    ctx = {}
    for ln in (r.stdout or "").splitlines():
        if ln.startswith("CTXJSON:"):
            try:
                ctx = json.loads(ln[len("CTXJSON:"):])
            except Exception:
                ctx = {}
    for k, i in enumerate(idxs):
        if results[i]["status"] == "FAIL" and str(k) in ctx:
            st, msg = ctx[str(k)]
            if st == "PASS":
                results[i]["status"] = "CTX_PASS"
                results[i]["note"] = "needs variables from earlier snippets on this page"
            else:
                results[i]["ctx_err"] = msg
    print(f"context pass done: {page}")

# non-python blocks -> skip with reason
for s in snippets:
    if s["lang"] in ("bash", "sh"):
        results.append({**s, "status": "SKIP", "out": "", "err": "",
                        "reason": "shell command (pip/git) — not executed; superfermion 0.1.5 already installed"})
    elif s["lang"] in ("text", "rust"):
        results.append({**s, "status": "SKIP", "out": "", "err": "",
                        "reason": "pseudo-code / non-Python block"})

print("Total evaluated blocks:", len(results))"""

cell_report = """print("=" * 110)
print("DOC SNIPPET VERIFICATION REPORT  (superfermion 0.1.5 + Rust core, Python 3.13)")
print("=" * 110)
per_page = defaultdict(Counter)
for r in results:
    per_page[r["page"]][r["status"]] += 1
total = Counter()
hdr = f"{'PAGE':<34} {'PASS':>5} {'CTX':>4} {'FAIL':>5} {'SKIP':>5} {'TOTAL':>6}"
print(hdr)
print("-" * 110)
for page in sorted(per_page):
    c = per_page[page]; total.update(c)
    print(f"{page:<34} {c['PASS']:>5} {c['CTX_PASS']:>4} {c['FAIL']:>5} {c['SKIP']:>5} {sum(c.values()):>6}")
print("-" * 110)
print(f"{'TOTAL':<34} {total['PASS']:>5} {total['CTX_PASS']:>4} {total['FAIL']:>5} {total['SKIP']:>5} {sum(total.values()):>6}")
print()
print("# FAILURES (broken doc snippets)")
for r in results:
    if r["status"] == "FAIL":
        print(f"\\n--- {r['page']} block #{r['idx']} ---")
        print(r["code"][:700])
        err = r.get("ctx_err") or r.get("err") or "(no error captured)"
        print("ERROR:", err[:500])
print()
print("# SKIPPED")
for r in results:
    if r["status"] == "SKIP":
        first = (r["code"].splitlines()[0] if r["code"] else "")[:70]
        print(f"- {r['page']} #{r['idx']}: {r.get('reason','?')} | {first}")
print()
print("# CONTEXT-DEPENDENT (passes only with earlier snippets on the same page)")
for r in results:
    if r["status"] == "CTX_PASS":
        first = (r["code"].splitlines()[0] if r["code"] else "")[:70]
        print(f"- {r['page']} #{r['idx']}: {first}")
print()
print("# PASSING snippets with no printed output (ran cleanly but silent)")
silent = 0
for r in results:
    if r["status"] == "PASS" and not (r.get("out") or "").strip():
        silent += 1
        print(f"- {r['page']} #{r['idx']}: {(r['code'].splitlines()[0] if r['code'] else '')[:70]}")
print(f"  (silent count: {silent})")"""

md_footer = """## How to read this report

- `FAIL` blocks are snippets that **do not work as documented** — they raise an error even when
  pasted in page order. Each failure lists the error message.
- `CTX_PASS` blocks work only when the whole page is run top-to-bottom (they reuse variables like
  `qc`, `ansatz`, `obs` defined in earlier snippets). This is normal for sequential docs but means
  the snippet alone is not self-contained.
- `SKIP` blocks are: shell commands (install steps), non-Python pseudo-code, snippets needing
  optional packages not installed in this environment, or snippets that require real cloud
  credentials (IBM/IonQ/AWS).

**Environment note**: the frontend dev server (`pnpm dev`) serves these same pages at
http://localhost:3000, and the deployed site superfermion.com/docs is content-identical
(spot-checked). All runs use the locally built Rust core (`maturin develop --release`)."""

cells = [
    nbformat.v4.new_markdown_cell(md_intro),
    nbformat.v4.new_code_cell(cell_inventory),
    nbformat.v4.new_code_cell(cell_deps),
    nbformat.v4.new_code_cell(cell_harness),
    nbformat.v4.new_code_cell(cell_context),
    nbformat.v4.new_code_cell(cell_report),
    nbformat.v4.new_markdown_cell(md_footer),
]

nb = nbformat.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13"},
    },
)

print("Executing notebook (this runs all snippets; may take a few minutes)...")
client = NotebookClient(nb, kernel_name="python3", timeout=1800)
client.execute()
nbformat.write(nb, NB_PATH)
print("Notebook saved:", NB_PATH)

# Echo outputs for review
for i, c in enumerate(nb.cells):
    if c.cell_type != "code":
        continue
    print(f"\n===== CELL {i} =====")
    for o in c.get("outputs", []):
        t = o.get("output_type")
        if t == "stream":
            print(o.get("text", ""), end="")
        elif t in ("execute_result", "display_data"):
            print(o.get("data", {}).get("text/plain", ""))
        elif t == "error":
            print("KERNEL ERROR:", o.get("ename"), "-", o.get("evalue"))
