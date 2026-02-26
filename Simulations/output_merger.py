import os
import sys
import glob
import pandas as pd


if len(sys.argv) != 2:
    raise SystemExit("Usage: python merge_pkls.py /path/to/folder")

folder = os.path.abspath(sys.argv[1])
files = sorted(glob.glob(os.path.join(folder, "*.pkl")))
if not files:
    raise SystemExit(f"No .pkl files found in: {folder}")

dfs = []
for fp in files:
    try:
        dfs.append(pd.read_pickle(fp))
    except Exception as e:
        print(f"Skipping unreadable: {fp} ({type(e).__name__}: {e})")

if not dfs:
    raise SystemExit("No readable .pkl files to merge.")

merged = pd.concat(dfs, axis=0)
merged = merged[~merged.index.duplicated(keep="last")]

parent = os.path.dirname(folder)
out_name = os.path.basename(folder.rstrip(os.sep)) + "_merged.pkl"
out_path = os.path.join(parent, out_name)

merged.to_pickle(out_path)
print(f"Wrote: {out_path}  (rows={len(merged)})")
