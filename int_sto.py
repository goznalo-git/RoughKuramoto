import numpy as np
import pandas as pd
import os, sys
import time

from aux_functions import order_parameter, timer
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
T = 2000  # a bit above 100000 to allow for the next log-scale datapoint (startsampling*multsampling**n) to be included.
Nt = 100*T
startsampling = 100
multsampling = 1.6
D = 0.01
M = int(sys.argv[4])

output_path = f"OutputTimeDep/Solutions_delta{deltaname}.pkl"

## Handle with care!!!!
# if os.path.exists(output_path):
#     os.remove(output_path)

print("## Delta", deltaname)

with timer():

    print("dt=", T / Nt)

    for it in range(M):

        if it % 10 == 0:
            print(it)

        seed = np.random.randint(0, 1e8)

        ####### TO DO #### DE-RANDOMIZE THIS ######
        omega = np.random.uniform(low=1, high=-1, size=L)

        # theta0 = np.random.uniform(0, 2*np.pi, size=L)
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

        # assuming R, Psi are computed somewhere here, as in your original code
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
                    "Psi": Psi
                }
            ]
        ).set_index("sim_id")

        # open DataFrame only when saving; append row and save back
        if os.path.exists(output_path):
            df = pd.read_pickle(output_path)
            df = pd.concat([df, row_df])
        else:
            df = row_df

        df = df[~df.index.duplicated(keep="last")]

        df.to_pickle(output_path)
