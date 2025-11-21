import numpy as np
import time
import os, re

    
def overwrite_dataset(group, name, data, **kwargs):
    """
    Create or overwrite a dataset in an HDF5 group.

    Parameters
    ----------
    group : h5py.Group
        The HDF5 group where the dataset lives.
    name : str
        Name of the dataset to create or overwrite.
    data : array-like
        Data to store (NumPy array or compatible).
    **kwargs :
        Extra options for create_dataset (e.g. compression='gzip').
    
    Returns
    -------
    dset : h5py.Dataset
        The newly created dataset.
    """
    if name in group:
        del group[name]  # remove the existing dataset or group
    dset = group.create_dataset(name, data=data, **kwargs)
    return dset



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