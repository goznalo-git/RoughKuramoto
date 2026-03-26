import numpy as np
from typing import Iterable, Tuple, Optional

def compute_surface_variance(phases: np.ndarray) -> np.ndarray:
    """
    Compute the surface variance of phases
    at each time from a Kuramoto–Sakaguchi simulation.

    Parameters
    ----------
    phases : ndarray
        Array of unwrapped phases with shape (Nt, N), as returned by
        `kuramoto_sakaguchi_euler`.

    Returns
    -------
    variance : ndarray of shape (Nt,)
        variance W(t) = <(θ_i - <θ>)^2> at each time step.
    """
    # Mean over oscillators at each time
    mean_phase = np.mean(phases, axis=1)
    # Subtract mean and compute the variance
    variance = np.mean((phases - mean_phase[:, None])**2, axis=1)
    return variance


def compute_surface_roughness(phases: np.ndarray) -> np.ndarray:
    """
    Compute the surface roughness (standard deviation of phases)
    at each time from a Kuramoto–Sakaguchi simulation.

    Parameters
    ----------
    phases : ndarray
        Array of unwrapped phases with shape (Nt, N), as returned by
        `kuramoto_sakaguchi_euler`.

    Returns
    -------
    roughness : ndarray of shape (Nt,)
        Roughness W(t) = sqrt( <(θ_i - <θ>)^2> ) at each time step.
    """
    # Mean over oscillators at each time
    mean_phase = np.mean(phases, axis=1)
    # Subtract mean and compute RMS fluctuation
    roughness = np.sqrt(np.mean((phases - mean_phase[:, None])**2, axis=1))
    return roughness
    

def compute_heightdifference_correlations(phases: np.ndarray) -> np.ndarray:
    """
    Compute the height-difference correlation function G(r, t) for a
    Kuramoto–Sakaguchi simulation.

    The height-difference correlation is defined as:
        G(r, t) = < [h(x + r, t) - h(x, t)]^2 >
    where the average is taken over spatial index x at fixed time t.

    Parameters
    ----------
    phases : ndarray
        Array of unwrapped phases with shape (Nt, N), interpreted as the
        interface height h(x, t).

    Returns
    -------
    heightheight : ndarray of shape (Nt, N)
        The correlation G(r, t) for separations r = 0, 1, ..., N-1 at each time t.
        The r-th column corresponds to separation r.
    """
    heightheight = np.empty((phases.shape[0], phases.shape[1]))
    for r in range(phases.shape[1]):
        heightheight[:,r] = np.mean((np.roll(phases, r, axis=1) - phases)**2, axis=1)

    return heightheight


def compute_structure_factor(phases: np.ndarray) -> np.ndarray:
    """
    Compute the structure factor function S(k, t) for a
    Kuramoto–Sakaguchi simulation.

    The structure factor is defined as:
        S(k, t) = < |hat(h)(k, t)|^2 >
    where the average is taken over frequency index k at fixed time t.

    Parameters
    ----------
    phases : ndarray
        Array of unwrapped phases with shape (Nt, N), interpreted as the
        interface height h(x, t).

    Returns
    -------
    structurefactor : ndarray of shape (Nt, N)
        The correlation S(k, t) for frequencies k = 0, 1, ..., N-1 at each time t.
        The k-th column corresponds to frequency k.
    """
    N = phases.shape[1]

    # Spatial Fourier transform along the lattice direction (axis=1)
    hhat = np.fft.fft(phases, axis=1)

    # Power spectrum (structure factor). Divide by N for the common FFT normalization.
    structurefactor = (np.abs(hhat) ** 2) / N
    
    return structurefactor


def compute_phase_covariance(phases: np.ndarray) -> np.ndarray:
    """
    Compute the phase covariance function C(r, t) for a Kuramoto–Sakaguchi simulation.

    The phase covariance is defined as:
        C(r, t) = < overline{φ(x, t) φ(x + r, t)} >  -  ( < overline{φ}(x, t) > )^2
    where overline{...} denotes an average over the spatial index x at fixed time t
    and <...> denotes an average over simulations.

    Note that, since the squaring is after the average over simulations, 
    here we just compute the two terms separately and afterwards the averaging 
    over simulations takes place, followed by the square and subtraction. 

    Parameters
    ----------
    phases : ndarray
        Array of (unwrapped) phases with shape (Nt, N), interpreted as φ(x, t).

    Returns
    -------
    hprodh_average : ndarray of shape (Nt, N)
        The spatial average of φ(x, t) φ(x + r, t) for separations r = 0, 1, ..., N-1 at each time t.
        The r-th column corresponds to separation r.
    h_average : ndarray of shape (Nt, )
        The spatial average of φ(x, t) at each time t.
    """
    Nt, N = phases.shape

    # Spatial mean at each time: overline{φ(x,t)}
    h_average = np.mean(phases, axis=1)  # shape (Nt,)

    hprodh_average = np.empty((Nt, N), dtype=float)
    for r in range(N):
        # <φ(x,t) φ(x+r,t)>
        hprodh_average[:, r] = np.mean(phases * np.roll(phases, r, axis=1), axis=1)

    return hprodh_average, h_average

