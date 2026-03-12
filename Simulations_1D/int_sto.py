import numpy as np
import pandas as pd
import os, sys
import time

import sys
sys.path.append('..')
from aux_functions import order_parameter, timer, atomic_write_pickle_df
from integration_functions import kuramoto_sakaguchi_euler

# Parameters
if sys.argv[1] == "0":
    delta = 0
    deltaname = "0"
elif sys.argv[1] == "atan5":
    delta = np.arctan(5)
    deltaname = "atan5"

print("TimeDep")

# 1D ring
L = int(sys.argv[2])
K = float(sys.argv[3])
T = 20000  # a bit above 100000 to allow for the next log-scale datapoint (startsampling*multsampling**n) to be included.
Nt = 100 * T
startsampling = 100
multsampling = 1.6
D = 0.01
M = int(sys.argv[4])

# Save ONE file per run, in a corresponding folder
out_dir = f"OutputTimeDep/delta{deltaname}/L{L}_K{K}"
os.makedirs(out_dir, exist_ok=True)

print("## Delta", deltaname)

with timer():

    print("dt=", T / Nt)

    for it in range(M):

        if it % 10 == 0:
            print(it)

        seed = np.random.randint(0, 1e8)
        rng = np.random.default_rng(seed)

        omega = rng.uniform(low=-1, high=1, size=L)

        theta0 = np.zeros((L, 1))

        # as well as the fields computed
        t, phases = kuramoto_sakaguchi_euler(
            "1d",
            L,
            omega,
            K,
            delta,
            theta0,
            T,
            Nt,
            startsampling=startsampling,
            multsampling=multsampling,
            stochastic=D,
            seed=seed,
        )

        R, Psi = order_parameter(phases)

        sim_id = f"L{L}_K{K}_seed{seed}"  # composite unique ID

        # one row per simulation; seed as identifier
        row_df = pd.DataFrame(
            [
                {
                    "sim_id": sim_id,
                    "seed": seed,
                    "L": L,
                    "K": K,
                    "delta": delta,
                    "T": T,
                    "Nt": Nt,
                    "startsampling": startsampling,
                    "multsampling": multsampling,
                    "D": D,
                    "omega": omega,
                    "t": t,
                    "phases": phases,
                    "R": R,
                    "Psi": Psi,
                }
            ]
        ).set_index("sim_id")

        # Save exactly ONE file per run (no read/concat/write of a big dataframe)
        output_path = os.path.join(out_dir, f"{sim_id}.pkl")
        atomic_write_pickle_df(row_df, output_path)