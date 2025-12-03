import numpy as np



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
    

def compute_height_height_correlations(phases: np.ndarray) -> np.ndarray:
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