import numpy as np
from numba import njit

@njit(cache=True)
def neighbor_sum_1d(th, delta):
    # periodic nearest neighbors on a ring
    L = th.shape[0]
    out = np.empty_like(th)

    for i in range(L):
        im = L - 1 if i == 0 else i - 1
        ip = 0 if i == L - 1 else i + 1
        out[i] = (
            np.sin(th[im] - th[i] - delta) +
            np.sin(th[ip] - th[i] - delta)
        )

    return out


@njit(cache=True)
def neighbor_sum_2d(th, delta):
    # periodic nearest neighbors on a 2D torus
    L0, L1 = th.shape
    out = np.empty_like(th)

    for i in range(L0):
        im = L0 - 1 if i == 0 else i - 1
        ip = 0 if i == L0 - 1 else i + 1

        for j in range(L1):
            jm = L1 - 1 if j == 0 else j - 1
            jp = 0 if j == L1 - 1 else j + 1

            c = th[i, j]
            out[i, j] = (
                np.sin(th[im, j] - c - delta) +
                np.sin(th[ip, j] - c - delta) +
                np.sin(th[i, jm] - c - delta) +
                np.sin(th[i, jp] - c - delta)
            )

    return out


@njit(cache=True)
def step_1d(th, N, omega, K, delta, dt, stochastic):
    """
    Perform one Euler or Euler-Maruyama step in 1D.
    """
    if stochastic is not None:
        omega = np.random.normal(0.0, np.sqrt(stochastic / dt), N)

    dtheta = omega + K * neighbor_sum_1d(th, delta)
    return th + dt * dtheta


@njit(cache=True)
def step_2d(th, N, omega, K, delta, dt, stochastic):
    """
    Perform one Euler or Euler-Maruyama step in 2D.
    """
    if stochastic is not None:
        omega = np.random.normal(0.0, np.sqrt(stochastic / dt), N)
        omega = omega.reshape(th.shape[0], th.shape[1])

    dtheta = omega + K * neighbor_sum_2d(th, delta)
    return th + dt * dtheta


@njit(cache=True)
def kuramoto_sakaguchi_euler_1d(
    L,
    omega,
    K,
    delta,
    theta0,
    T,
    Nt,
    startsampling,
    multsampling,
    stochastic=None,
    seed=None
):
    # Set the random seed
    if stochastic is not None and seed is not None:
        np.random.seed(seed)

    N = L
    omega = omega.reshape(L)
    theta = theta0.copy().reshape(L)

    # Determine the integration step
    dt = T / Nt
    t = np.linspace(0, T, Nt)

    # First estimate the amount n of logarithmic timesteps from
    # startsampling * multsampling**(n) = Nt
    logsteps = int(np.ceil(np.log(Nt / startsampling) / np.log(multsampling)))

    # Preallocate output
    phases = np.empty((logsteps + 1, omega.size), dtype=np.float64)
    phases[0] = theta.ravel()

    # Integration step, and save logarithmically
    t_save = np.empty(logsteps + 2, dtype=np.int64)
    t_save[0] = 0
    for n in range(logsteps + 1):
        t_save[n + 1] = np.int64(np.floor(startsampling * multsampling**n))

    save_index = 1
    for ti in range(1, Nt):
        theta = step_1d(theta, N, omega, K, delta, dt, stochastic)

        if ti > t_save[save_index]:
            phases[save_index] = theta.ravel()
            save_index += 1

    return t[t_save[:-1]], phases


@njit(cache=True)
def kuramoto_sakaguchi_euler_2d(
    L,
    omega,
    K,
    delta,
    theta0,
    T,
    Nt,
    startsampling,
    multsampling,
    stochastic=None,
    seed=None
):
    # Set the random seed
    if stochastic is not None and seed is not None:
        np.random.seed(seed)

    N = L * L
    omega = omega.reshape(L, L)
    theta = theta0.copy().reshape(L, L)

    # Determine the integration step
    dt = T / Nt
    t = np.linspace(0, T, Nt)

    # First estimate the amount n of logarithmic timesteps from
    # startsampling * multsampling**(n) = Nt
    logsteps = int(np.ceil(np.log(Nt / startsampling) / np.log(multsampling)))

    # Preallocate output
    phases = np.empty((logsteps + 1, omega.size), dtype=np.float64)
    phases[0] = theta.ravel()

    # Integration step, and save logarithmically
    t_save = np.empty(logsteps + 2, dtype=np.int64)
    t_save[0] = 0
    for n in range(logsteps + 1):
        t_save[n + 1] = np.int64(np.floor(startsampling * multsampling**n))

    save_index = 1
    for ti in range(1, Nt):
        theta = step_2d(theta, N, omega, K, delta, dt, stochastic)

        if ti > t_save[save_index]:
            phases[save_index] = theta.ravel()
            save_index += 1

    return t[t_save[:-1]], phases


def kuramoto_sakaguchi_euler(
    topology: str,
    L: int,
    omega: np.ndarray,
    K: float,
    delta: float,
    theta0: np.ndarray,
    T: float,
    Nt: int,
    startsampling: int,
    multsampling: float,
    stochastic=None,
    seed=None
):
    """
    Integrate the Kuramoto–Sakaguchi model with Euler's method on a periodic lattice.

    dθ_i/dt = ω_i + K * Σ_{j->i} sin(θ_j - θ_i - δ)

    - topology:         "1d" (ring) or "2d" (square torus), nearest-neighbor coupling.
    - L:                system size (N=L for 1d, N=L*L for 2d).
    - omega:            array of natural frequencies (shape (L,) for 1d or (L,L) for 2d; flattened is also accepted).
    - K:                coupling strength.
    - delta:            Sakaguchi phase-lag δ.
    - theta0:           initial phases (same shape as omega).
    - T:                final integration time (>=0).
    - Nt:               timesteps (>0),
    - startsampling:    starting point of the logarithmic scale used for saving data,
    - multsampling:     multiplicative factor of the logarithmic scale used for saving data.
                        The data is therefore saved at timesteps [startsampling, startsampling*multsampling, startsampling*multsampling**2, ...]
    - stochastic:       whether we consider columnar noise (=None) or time dependent (=D), 
                        Gaussian distributed with zero mean and variance D. 
                        If time dependent, then the omega value is not used (but one should still be provided).
    - seed:             this random seed must be provided if the simulation is of the time-dependent model.

    Returns:
    - t:                np.ndarray with shape (Nt,) list of times.
    - phases:           ndarray with shape (Nt, N) containing *unwrapped* phases at each saved time,
                        flattened in row-major order.

    Note: this function calls specific 1d or 2d functions, the reason for it being that each of them is precompiled by numba, and that requires their inputs/outputs to have a fixed type (and the different nature of the 1d/2d arrays makes that impossible). 
    """
    if topology == "1d":
        return kuramoto_sakaguchi_euler_1d(
            L, omega, K, delta, theta0, T, Nt,
            startsampling, multsampling, stochastic, seed
        )
    else:
        return kuramoto_sakaguchi_euler_2d(
            L, omega, K, delta, theta0, T, Nt,
            startsampling, multsampling, stochastic, seed
        )