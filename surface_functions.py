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
        S(k, t) = < |\hat(h)(k, t)|^2 >
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
        C(r, t) = < φ(x, t) φ(x + r, t) >  -  ( < φ(x, t) > )^2
    where <...> denotes an average over the spatial index x at fixed time t.

    Parameters
    ----------
    phases : ndarray
        Array of (unwrapped) phases with shape (Nt, N), interpreted as φ(x, t).

    Returns
    -------
    cov : ndarray of shape (Nt, N)
        The covariance C(r, t) for separations r = 0, 1, ..., N-1 at each time t.
        The r-th column corresponds to separation r.
    """
    Nt, N = phases.shape

    # Spatial mean at each time: <φ(x,t)>
    mean_phi_t = np.mean(phases, axis=1)  # shape (Nt,)

    phase_covariance = np.empty((Nt, N), dtype=float)
    for r in range(N):
        # <φ(x,t) φ(x+r,t)>
        prod_mean = np.mean(phases * np.roll(phases, r, axis=1), axis=1)
        phase_covariance[:, r] = prod_mean - mean_phi_t**2

    return phase_covariance


def compute_rescaled_phases(
    phases: np.ndarray,
    t0: int,
    deltaTs: Iterable[int],
    beta: float,
) -> np.ndarray:
    """
    Compute rescaled phases (fluctuations) as in:
        ϕ_j(Δt) = [δφ_j(t0 + Δt) - δφ_j(t0)] / (Δt)^β
    where δφ_j(t) = φ_j(t) - \bar{φ}(t) and \bar{φ}(t) is the spatial mean at time t.

    Parameters
    ----------
    phases : ndarray, shape (Nt, N)
        Unwrapped phases φ_j(t).
    t0 : int
        Reference time index.
    deltas : iterable of int
        Time lags Δt (positive integers) with t0 + Δt < Nt.
    beta : float
        Rescaling exponent.

    Returns
    -------
    varphi : ndarray, shape (n_deltas, N)
        Rescaled phases ϕ_j for each Δt. Row i corresponds to deltas[i].
    """
    Nt, N = phases.shape
    deltaTs = np.asarray(list(deltaTs), dtype=int)

    if t0 < 0 or t0 >= Nt:
        raise ValueError("t0 must be a valid time index.")
    if np.any(deltaTs <= 0):
        raise ValueError("All Δt must be positive integers.")
    if np.any(t0 + deltaTs >= Nt):
        raise ValueError("Each Δt must satisfy t0 + Δt < Nt.")

    # δφ(t) = φ(t) - mean_x φ(t)
    mean_t = phases.mean(axis=1, keepdims=True)   # (Nt, 1)
    dphi = phases - mean_t                        # (Nt, N)

    dphi_t0 = dphi[t0]                            # (N,)

    varphi = np.empty((len(deltaTs), N), dtype=float)
    for i, Deltat in enumerate(deltaTs):
        varphi[i] = (dphi[t0 + Deltat] - dphi_t0) / (Deltat ** beta)
    return varphi


def compute_fluctuation_pdf(
    varphi: np.ndarray,
    bins: int | np.ndarray = 100,
    value_range: Optional[Tuple[float, float]] = None,
    axis: int = -1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the PDF of the rescaled phase fluctuations ϕ.

    Parameters
    ----------
    varphi : ndarray
        Array of rescaled phases ϕ. Typically shape (n_deltas, N),
        but any shape is allowed; the PDF is computed along `axis`.
    bins : int or ndarray, default 100
        Histogram bin specification passed to np.histogram.
    value_range : (min, max) or None
        Optional common range for the histogram.
    axis : int, default -1
        Axis along which to compute the PDF (e.g. spatial index).

    Returns
    -------
    bin_centers : ndarray, shape (nbins,)
        Centers of the histogram bins.
    pdf : ndarray
        Probability density function(s). If `varphi` has shape
        (n_deltas, N) and axis = -1, the result has shape (n_deltas, nbins).
    """
    varphi = np.asarray(varphi)

    # Move the target axis to the last position
    data = np.moveaxis(varphi, axis, -1)
    leading_shape = data.shape[:-1]

    # Flatten all samples to define common bins (important for collapse tests)
    flat = data.reshape(-1)

    if value_range is None:
        _, edges = np.histogram(flat, bins=bins, density=True)
    else:
        _, edges = np.histogram(flat, bins=bins, range=value_range, density=True)

    bin_centers = 0.5 * (edges[:-1] + edges[1:])

    # Compute PDFs slice by slice
    pdf = np.empty((*leading_shape, len(bin_centers)), dtype=float)
    for idx in np.ndindex(leading_shape):
        pdf[idx], _ = np.histogram(data[idx], bins=edges, density=True)

    return bin_centers, pdf
