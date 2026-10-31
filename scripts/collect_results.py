import argparse, csv, glob, json, os

ap = argparse.ArgumentParser(description="collect eval accuracy per difficulty for all runs")
ap.add_argument("--root", default="/kaggle/working")
ap.add_argument("--extra-roots", nargs="*", default=[], help="earlier sessions' outputs, e.g. under /kaggle/input")
a = ap.parse_args()

rows = {}
for root in [*a.extra_roots, a.root]:
    for path in glob.glob(os.path.join(root, "**", "eval.jsonl"), recursive=True):
        run = os.path.basename(os.path.dirname(path))
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                rows[run, r["step"], r["difficulty"]] = r

out = os.path.join(a.root, "results")
os.makedirs(out, exist_ok=True)
DIFFS = ["easy", "medium", "hard"]
METRICS = ["solved", "cell", "valid"]
FIELDS = ["run", "step", "boards"] + [f"{d}_{m}" for d in DIFFS for m in METRICS]
table = []
for run, step in sorted({k[:2] for k in rows}):
    row = {"run": run, "step": step, "boards": ""}
    for d in DIFFS:
        r = rows.get((run, step, d))
        if r:
            row["boards"] = r["boards"]
            row.update({f"{d}_{m}": round(r[m], 2) for m in METRICS})
    table.append(row)

with open(os.path.join(out, "results.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, FIELDS)
    w.writeheader()
    w.writerows(table)
with open(os.path.join(out, "results.md"), "w") as f:
    f.write("| " + " | ".join(FIELDS) + " |\n|" + "---|" * len(FIELDS) + "\n")
    for row in table:
        f.write("| " + " | ".join(str(row.get(k, "")) for k in FIELDS) + " |\n")
print(open(os.path.join(out, "results.md")).read())
print(f"{len(table)} rows -> {out}")
