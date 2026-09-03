import os
import sys
import glob
import pandas as pd

sys.path.append('..')
sys.path.append('../..')
from aux_functions import atomic_write_pickle_df


def merge_pickles_in_dir(parent_dir: str, out_path: str):
    """
    Merge all .pkl files found under parent_dir/L*/ into one dataframe.
    """
    pattern = os.path.join(parent_dir, "L*", "*.pkl")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"  - No files found under {parent_dir}")
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
        print(f"  - No readable files under {parent_dir}")
        return

    merged = pd.concat(dfs, axis=0)
    merged = merged[~merged.index.duplicated(keep="last")]

    atomic_write_pickle_df(merged, out_path)
    print(f"  - Wrote {out_path} (rows={len(merged)}, files={len(files)}, skipped={skipped})")


def merge_root(root: str):
    """
    root can be either:
      - HORIZ: contains delta*/L*/*.pkl
      - VERT:  contains K*/L*/*.pkl
    """
    root = os.path.abspath(root)

    if not os.path.isdir(root):
        print(f"Skipping (not a directory): {root}")
        return

    delta_dirs = sorted(glob.glob(os.path.join(root, "delta*")))
    k_dirs = sorted(glob.glob(os.path.join(root, "K*")))

    did_any = False

    # HORIZ case: merge each delta* folder
    for ddir in delta_dirs:
        if not os.path.isdir(ddir):
            continue

        delta_name = os.path.basename(ddir)   # e.g. delta125
        out_path = os.path.join(root, f"all_simulations_{delta_name}.pkl")
        print(f"Merging {ddir}")
        merge_pickles_in_dir(ddir, out_path)
        did_any = True

    # VERT case: merge each K* folder
    for kdir in k_dirs:
        if not os.path.isdir(kdir):
            continue

        k_name = os.path.basename(kdir)       # e.g. K10.0
        out_path = os.path.join(root, f"all_simulations_{k_name}.pkl")
        print(f"Merging {kdir}")
        merge_pickles_in_dir(kdir, out_path)
        did_any = True

    if not did_any:
        print(f"No delta* or K* folders found in {root}")


if len(sys.argv) < 2:
    raise SystemExit("Usage: python merge_pkl.py <root1> [<root2> ...]")

for root in sys.argv[1:]:
    merge_root(root)