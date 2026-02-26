import numpy as np


def neighbor_sum_1d(th, delta):
    # periodic nearest neighbors on a ring
    return (np.sin(np.roll(th, +1) - th - delta) +
            np.sin(np.roll(th, -1) - th - delta))

def neighbor_sum_2d(th, delta):
    # periodic nearest neighbors on a 2D torus
    return (np.sin(np.roll(th, +1, axis=0) - th - delta) +
            np.sin(np.roll(th, -1, axis=0) - th - delta) +
            np.sin(np.roll(th, +1, axis=1) - th - delta) +
            np.sin(np.roll(th, -1, axis=1) - th - delta))


def step(topology, th, N, omega, K, delta, dt, stochastic, rng):
    """
    Perform one Euler or Euler-Maruyama (with noise instead of frequencies) step.
    """
    
    # Euler-Maruyama case, with sqrt(D*dt) * normalized noise.
    if stochastic is not None:
        omega = rng.normal(0, np.sqrt(stochastic/dt), size=N)
        
    if topology == "1d":
        dtheta = omega + K * neighbor_sum_1d(th, delta)
    else:
        dtheta = omega + K * neighbor_sum_2d(th, delta)
        
    return th + dt * dtheta


def kuramoto_sakaguchi_euler(
    topology: str,           # "1d" or "2d"
    L: int,                  # chain length or lattice side
    omega: np.ndarray,       # natural frequencies (size L for 1d, LxL for 2d, or flattened)
    K: float,                # coupling strength
    delta: float,            # Sakaguchi phase-lag
    theta0: np.ndarray,      # initial phases (same shape as omega)
    T: float,                # final time
    Nt: int,                 # timesteps
    startsampling: int,      # start of the logarithmic scale
    multsampling: float,     # multiplicative factor for the logarithmic scale to save
    stochastic=None,         # columnar noise or time dependent noise
    seed=None                # randomness seed (if stochastic)
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
    """

    # Set the random number generator
    if stochastic is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = None
        
    theta = theta0.copy()

    if topology == "1d":
        N = L
        omega = omega.reshape(L)
        theta = theta.reshape(L)
    else:  # "2d"
        N = L * L
        omega = omega.reshape(L, L)
        theta = theta.reshape(L, L)

    # Determine the integration step
    dt = T / Nt
    t = np.linspace(0,T,Nt)

    # First estimate the amount n of logarithmic timesteps from
    # startsampling * multsampling**(n) = Nt
    logsteps = int(np.ceil(np.log(Nt/startsampling)/np.log(multsampling)))
    
    # Preallocate output
    phases = np.empty((logsteps+1, omega.size), dtype=np.float64)
    phases[0] = theta.ravel()

    # Integration step, and save logarithmically
    t_save = np.int64(np.floor([0] + [startsampling*multsampling**n for n in range(logsteps+1)]))

    save_index = 1
    for ti in range(1,Nt):
        theta = step(topology, theta, N, omega, K, delta, dt, stochastic, rng)

        if ti > t_save[save_index]:
            phases[save_index] = theta.ravel()
            save_index += 1 

    return t[t_save[:-1]], phases

