"""Print outputs of the saved docs_verification notebook (cp1252-safe)."""
import nbformat, sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
path = r"c:\Users\ASUS\OneDrive\Desktop\sfdocs\superfermion\notebooks\docs_verification.ipynb"
nb = nbformat.read(path, as_version=4)
for i, c in enumerate(nb.cells):
    if c.cell_type != "code":
        continue
    print(f"===== CELL {i} =====")
    for o in c.get("outputs", []):
        t = o.get("output_type")
        if t == "stream":
            print(o.get("text", ""), end="")
        elif t in ("execute_result", "display_data"):
            print(o.get("data", {}).get("text/plain", ""))
        elif t == "error":
            print("KERNEL ERROR:", o.get("ename"), "-", o.get("evalue"))
    print()
