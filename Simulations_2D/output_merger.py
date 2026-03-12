import os
import sys
import glob
import pandas as pd

from aux_functions import atomic_write_pickle_df

def merge_one_delta(delta_dir: str, out_path: str):
    pattern = os.path.join(delta_dir, "L*_K*", "*.pkl")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"  - No files found under {delta_dir}")
        return

    dfs = []
    skipped = 0
    for fp in files:
        try:
            dfs.append(pd.read_pickle(fp))
        except Exception as e:
            skipped += 1
            print(f"    Skipping unreadable: {fp} ({type(e).__name__}: {e})")

    if not dfs:
        print(f"  - No readable files under {delta_dir}")
        return

    merged = pd.concat(dfs, axis=0)
    merged = merged[~merged.index.duplicated(keep="last")]

    atomic_write_pickle_df(merged, out_path)
    print(f"  - Wrote {out_path}  (rows={len(merged)}, files={len(files)}, skipped={skipped})")

def merge_root(root: str):
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        print(f"Skipping (not a directory): {root}")
        return

    delta_dirs = sorted(glob.glob(os.path.join(root, "delta*")))
    if not delta_dirs:
        print(f"No delta* folders in {root}")
        return

    print(f"Merging in {root}")
    for ddir in delta_dirs:
        if not os.path.isdir(ddir):
            continue
        delta_name = os.path.basename(ddir)       # e.g. "delta0"
        typedelta = delta_name[len("delta"):]     # e.g. "0"
        out_path = os.path.join(root, f"all_simulations_delta{typedelta}.pkl")
        merge_one_delta(ddir, out_path)

if len(sys.argv) < 2:
    raise SystemExit("Usage: python merge_pkl.py <root1> [<root2> ...]")

for root in sys.argv[1:]:
    merge_root(root)
