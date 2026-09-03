import numpy as np
from typing import Iterable, Tuple, Optional

def compute_surface_variance(phases: np.ndarray) -> np.ndarray:
    """
    Compute the surface variance of phases
    at each time from a Kuramoto–Sakaguchi simulation.

    Note that it also works in 2D if the phases array is unraveled, e.g. shape = (Nt, L x L)/ 

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

    Note that it also works in 2D if the phases array is unraveled, e.g. shape = (Nt, L x L)/ 
    
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
    
import numpy as np


def compute_heightdifference_correlations(
    phases: np.ndarray,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the height-difference correlation function G(r, t) for a
    Kuramoto–Sakaguchi simulation.

    In one dimension, the height-difference correlation is defined as:
        G(r, t) = < [h(x + r, t) - h(x, t)]^2 >
    where the average is taken over spatial index x at fixed time t.

    In two dimensions, the correlations along each lattice direction are:
        G_1(r, t) = < [h(x + r, y, t) - h(x, y, t)]^2 >
        G_2(r, t) = < [h(x, y + r, t) - h(x, y, t)]^2 >
    where the averages are taken over both spatial coordinates.

    Periodic boundary conditions are assumed.

    Parameters
    ----------
    phases : ndarray
        Array of unwrapped phases.

        For a one-dimensional lattice, the expected shape is (Nt, N).

        For a two-dimensional square lattice, the expected shape is
        (Nt, L, L).

    Returns
    -------
    heightheight : ndarray of shape (Nt, N)
        For a one-dimensional lattice, G(r, t) for separations
        r = 0, 1, ..., N-1.

    heightheight_1, heightheight_2, heightheight_average : tuple of ndarrays
        For a two-dimensional lattice, the directional correlations and
        their average. Each array has shape (Nt, L).
    """
    if phases.ndim == 2:
        heightheight = np.empty((phases.shape[0], phases.shape[1]))

        for r in range(phases.shape[1]):
            heightheight[:, r] = np.mean(
                (np.roll(phases, r, axis=1) - phases) ** 2,
                axis=1,
            )

        return heightheight

    if phases.ndim == 3:
        if phases.shape[1] != phases.shape[2]:
            raise ValueError(
                "For a two-dimensional lattice, phases must have shape "
                "(Nt, L, L)."
            )

        L = phases.shape[1]

        heightheight_1 = np.empty((phases.shape[0], L))
        heightheight_2 = np.empty((phases.shape[0], L))

        for r in range(L):
            heightheight_1[:, r] = np.mean(
                (np.roll(phases, r, axis=1) - phases) ** 2,
                axis=(1, 2),
            )

            heightheight_2[:, r] = np.mean(
                (np.roll(phases, r, axis=2) - phases) ** 2,
                axis=(1, 2),
            )

        heightheight_average = (
            heightheight_1 + heightheight_2
        ) / 2

        return (
            heightheight_1,
            heightheight_2,
            heightheight_average,
        )

    raise ValueError(
        "phases must have shape (Nt, N) for a one-dimensional lattice "
        "or (Nt, L, L) for a two-dimensional lattice."
    )


def compute_structure_factor(
    phases: np.ndarray,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the structure factor S(k, t) for a Kuramoto–Sakaguchi
    simulation.

    In one dimension, the structure factor is:
        S(k, t) = |hat(h)(k, t)|^2 / N

    In two dimensions, a one-dimensional Fourier transform is performed
    separately along each lattice direction. The resulting power spectrum
    is averaged over the transverse spatial direction.

    Parameters
    ----------
    phases : ndarray
        Array of unwrapped phases.

        For a one-dimensional lattice, the expected shape is (Nt, N).

        For a two-dimensional square lattice, the expected shape is
        (Nt, L, L).

    Returns
    -------
    structurefactor : ndarray of shape (Nt, N)
        For a one-dimensional lattice, S(k, t) for frequencies
        k = 0, 1, ..., N-1.

    structurefactor_1, structurefactor_2,
    structurefactor_average : tuple of ndarrays
        For a two-dimensional lattice, the directional structure factors
        and their average. Each array has shape (Nt, L).
    """
    if phases.ndim == 2:
        N = phases.shape[1]

        # Spatial Fourier transform along the lattice direction
        hhat = np.fft.fft(phases, axis=1)

        # Power spectrum. Divide by N for the common FFT normalization.
        structurefactor = (np.abs(hhat) ** 2) / N

        return structurefactor

    if phases.ndim == 3:
        if phases.shape[1] != phases.shape[2]:
            raise ValueError(
                "For a two-dimensional lattice, phases must have shape "
                "(Nt, L, L)."
            )

        L = phases.shape[1]

        # Fourier transform along the first spatial direction. The power
        # spectrum is then averaged over the second spatial direction.
        hhat_1 = np.fft.fft(phases, axis=1)
        structurefactor_1 = np.mean(
            np.abs(hhat_1) ** 2,
            axis=2,
        ) / L

        # Fourier transform along the second spatial direction. The power
        # spectrum is then averaged over the first spatial direction.
        hhat_2 = np.fft.fft(phases, axis=2)
        structurefactor_2 = np.mean(
            np.abs(hhat_2) ** 2,
            axis=1,
        ) / L

        structurefactor_average = (
            structurefactor_1 + structurefactor_2
        ) / 2

        return (
            structurefactor_1,
            structurefactor_2,
            structurefactor_average,
        )

    raise ValueError(
        "phases must have shape (Nt, N) for a one-dimensional lattice "
        "or (Nt, L, L) for a two-dimensional lattice."
    )

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
    
    Note that it also works in 2D if the phases array is unraveled, e.g. shape = (Nt, L x L)/ 

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
