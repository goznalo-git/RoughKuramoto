import numpy as np

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

