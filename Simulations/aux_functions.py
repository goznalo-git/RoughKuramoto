import numpy as np
import time
import os, re

def order_parameter(phases):
    """
    Compute the complex Kuramoto order parameter over time.

    Parameters
    ----------
    phases : array, shape (N, M)
        Unwrapped phases for N oscillators over M stored times.

    Returns
    -------
    R : (M,) array
        Magnitude of the order parameter over time.
    psi : (M,) array
        Mean phase (argument) over time in (-pi, pi].
    """
    z = np.mean(np.exp(1j * phases), axis=1)  # average over oscillators
    R = np.abs(z)
    psi = np.angle(z)
    return R, psi



## General stuff
class timer:
    def __init__(self):
        self.ts, self.te = 0, 0
        
    def __enter__(self):
        self.ts = time.perf_counter()    
    
    def __exit__(self, *args, **kwargs):
        self.te = time.perf_counter()
        dt = self.te - self.ts
        print(f"This chunk of code took {dt} seconds.")


def atomic_write_pickle_df(df, path: str):
    """
    Atomically write a DataFrame to a pickle file.
    Prevents half-written/corrupt .pkl if the job is interrupted mid-write.
    """
    tmp = path + ".tmp"
    df.to_pickle(tmp)
    os.replace(tmp, path)  # atomic replace