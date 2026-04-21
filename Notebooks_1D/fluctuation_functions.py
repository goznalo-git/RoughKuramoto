import numpy as np
from TracyWidom import TracyWidom
from scipy.stats import skew
import matplotlib.pyplot as plt
    
def compute_rescaled_phases_at_index(
    phases: np.ndarray,
    t: np.ndarray,
    t0_idx: int,
    t1_idx: int,
    beta: float,
) -> np.ndarray:
    """
    Compute rescaled fluctuations:
        ϕ_j = [δφ_j(t1) - δφ_j(t0)] / (t1 - t0)^β
    with δφ_j(t) = φ_j(t) - mean_x φ(t),
    using physical times t (works for log-saved snapshots).

    Parameters
    ----------
    phases : ndarray, shape (T, N)
        Unwrapped phases φ_j(t).
    t : ndarray, shape (T,)
        Physical times corresponding to the first axis of `phases`.
    t0_idx : int
        Index of the reference time t0.
    t1_idx : int
        Index of the target time t1.
    beta : float
        Rescaling exponent.

    Returns
    -------
    varphi : ndarray, shape (N,)
        Rescaled phases ϕ_j at time t1 relative to t0.
    """
    T, N = phases.shape

    t0 = t[t0_idx]
    t1 = t[t1_idx]
    dt = t1 - t0
    if dt <= 0:
        raise ValueError(f"Need t1 > t0, got Δt = {dt}.")

    # δφ(t) = φ(t) - mean_x φ(t)
    mean_t = phases.mean(axis=1, keepdims=True)   # (T, 1)
    dphi = phases - mean_t                        # (T, N)

    varphi = (dphi[t1_idx] - dphi[t0_idx]) / (dt ** beta)

    return varphi

    
def _estimate_moments_from_pdf(pdf_fn, grid_min=-20.0, grid_max=10.0, n=200_000):
    """
    Numerically estimate mean and variance from a pdf function on a grid.
    The bounds [-20, 10] are usually enough for TW; increase if needed.
    """
    x = np.linspace(grid_min, grid_max, n)
    p = pdf_fn(x)

    # Guard against tiny negative numerical noise
    p = np.clip(p, 0.0, None)

    Z = np.trapezoid(p, x)
    if not np.isfinite(Z) or Z <= 0:
        raise RuntimeError("PDF integration failed (normalization constant nonpositive).")

    p /= Z  # normalize just in case

    mu = np.trapezoid(x * p, x)
    m2 = np.trapezoid((x - mu) ** 2 * p, x)
    sigma = np.sqrt(m2)
    return mu, sigma


## Loading TW pre-saved data
from scipy.interpolate import interp1d

data = np.loadtxt('../tw_data.csv', delimiter=',', skiprows=1)
s_vals = data[:, 0]
ds = s_vals[1] - s_vals[0]
tw_interp = {}

# Create interpolators for each TW distribution
for col, beta in enumerate([1, 2, 4], start=1):
    pdf = data[:, col]
    # Calculate moments from CSV to standardize
    mu = np.sum(s_vals * pdf) * ds
    sigma = np.sqrt(np.sum(s_vals**2 * pdf) * ds - mu**2)
    # Map to standardized coordinates
    tw_interp[beta] = interp1d((s_vals - mu) / sigma, pdf * sigma, 
                               kind='cubic', bounds_error=False, fill_value=0.0)


def normalized_tw_pdf(x, beta, grid_min=-20.0, grid_max=10.0, n_grid=200_000):
    """
    Standardized Tracy–Widom PDF (mean 0, variance 1) for beta=1,2,4.

    Returns f_std(x) where if Y ~ TW_beta, then X = (Y - mu)/sigma has PDF:
        f_std(x) = sigma * f_Y(sigma*x + mu)
    """
    TW = TracyWidom(beta=beta)

    # Build pdf function
    pdf_fn = lambda y: TW.pdf(y)

    mu, sigma = _estimate_moments_from_pdf(
        pdf_fn, grid_min=grid_min, grid_max=grid_max, n=n_grid
    )
    return TW.pdf(sigma * x + mu) * sigma


def plot_fluctuations_pdf(avg_df, L, T, tx, Ks, save=False):

    fig, ax = plt.subplots(2, 2, figsize=(10,10))
    # fig.suptitle("Fluctuation statistics under different noise types, for $L=$" + str(L))
    
    fig.supxlabel(r"$\varphi$")
    fig.supylabel(r"$P(\varphi)$")
    
    for i, (typenoise, typelabel) in enumerate(zip(["TimeDep", "Columnar"], ["Time-dependent", "Columnar"])):
    
        for j, (delta, deltaname) in enumerate(zip([0, np.arctan(5)], ["0", "atan5"])):
    
            # select rows for this noise type and delta
            mask = mask = (
                (avg_df["typenoise"] == typenoise)
                & (avg_df["L"] == L)
                & (avg_df["T"] == T)
                & (avg_df["deltaname"] == deltaname)
                & ((avg_df["K"] == Ks[0]) | (avg_df["K"] == Ks[1]))
            )
            sub = avg_df[mask]
    
            # in case there are multiple (K, L) combos, plot them all
            for _, row in sub.iterrows():
                t = row["t"]            
                varphis_full = row["varphis_full"]
    
                varphis_norm = (varphis_full - varphis_full.mean()) / varphis_full.std()
    
                print(f"Skewness ({typenoise}, {deltaname}):", skew(varphis_norm))
                
                # Data histogram
                nbins = 200
                hist, edges = np.histogram(varphis_norm, bins=nbins, density=True)
                centers = 0.5 * (edges[:-1] + edges[1:])
    
                ax[j,i].plot(centers, hist, marker='x', linestyle='None', label=r'Normalized $P(\varphi_i)$', markersize=6)
    
                # Gaussian N(0,1)
                xx = np.linspace(centers.min(), centers.max(), 1500)
                gauss_pdf = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * xx**2)
                ax[j,i].plot(xx, gauss_pdf, label='Gaussian N(0,1)')
    
                # Tracy–Widom PDFs
                ax[j,i].plot(xx, tw_interp[1](xx), label='GOE-TW')
                ax[j,i].plot(xx, tw_interp[2](xx), label='GUE-TW')
                ax[j,i].plot(xx, tw_interp[4](xx), label='GSE-TW')
    
                # ax[j,i].text(0.8, 0.95, fr'$t_\star={np.round(t_cross,1)}$', fontsize=10, transform=ax[j,i].transAxes)
    
                ax[j,i].set_xlim(-1.5,1.5)
                ax[j,i].set_ylim(0.2,0.41)
                
    ax[0,0].set_title(f"Time-dependent ($K=${Ks[0]})")
    ax[0,1].set_title(f"Columnar ($K=${Ks[1]})")      
    ax[0,0].set_ylabel(r"EW  ($\delta=0$)", fontsize=12)
    ax[1,0].set_ylabel(r"KPZ  ($\delta=\arctan5$)", fontsize=12)
    
    ax[0,0].legend()
    ax[0,1].legend()
    ax[1,0].legend()
    ax[1,1].legend()
    
    plt.tight_layout()
    if save:
        plt.savefig(f"Figures/Fluctuations/fluctuationPDFs_{L}.png", dpi=300)
    plt.show()


def plot_log_fluctuations_pdf(avg_df, L, T, tx, Ks, save=False):

    fig, ax = plt.subplots(2, 2, figsize=(10,10))
    # fig.suptitle("Fluctuation statistics under different noise types, for $L=$" + str(L))
    
    fig.supxlabel(r"$\varphi$")
    fig.supylabel(r"$\log P(\varphi)$")
    
    for i, (typenoise, typelabel) in enumerate(zip(["TimeDep", "Columnar"], ["Time-dependent", "Columnar"])):
    
        for j, (delta, deltaname) in enumerate(zip([0, np.arctan(5)], ["0", "atan5"])):
    
            # select rows for this noise type and delta
            mask = mask = (
                (avg_df["typenoise"] == typenoise)
                & (avg_df["L"] == L)
                & (avg_df["T"] == T)
                & (avg_df["deltaname"] == deltaname)
                & ((avg_df["K"] == Ks[0]) | (avg_df["K"] == Ks[1]))
            )
            sub = avg_df[mask]
    
            # in case there are multiple (K, L) combos, plot them all
            for _, row in sub.iterrows():
                t = row["t"]            
                varphis_full = row["varphis_full"]
    
                varphis_norm = (varphis_full - varphis_full.mean()) / varphis_full.std()
    
                # Data histogram
                nbins = 200
                hist, edges = np.histogram(varphis_norm, bins=nbins, density=True)
                centers = 0.5 * (edges[:-1] + edges[1:])
    
                ax[j,i].plot(centers, hist, marker='x', linestyle='None', label=r'Normalized $P(\varphi_i)$', markersize=6)
    
                # Gaussian N(0,1)
                xx = np.linspace(centers.min(), centers.max(), 1500)
                gauss_pdf = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * xx**2)
                ax[j,i].semilogy(xx, gauss_pdf, label='Gaussian N(0,1)')
    
                # Tracy–Widom PDFs
                ax[j,i].semilogy(xx, tw_interp[1](xx), label='GOE-TW')
                ax[j,i].semilogy(xx, tw_interp[2](xx), label='GUE-TW')
                ax[j,i].semilogy(xx, tw_interp[4](xx), label='GSE-TW')
    
                # ax[j,i].set_xlim(-1.5,1.5)
                ax[j,i].set_ylim(1e-4,0.5)
                
    ax[0,0].set_title(f"Time-dependent ($K=${Ks[0]})")
    ax[0,1].set_title(f"Columnar ($K=${Ks[1]})")      
    ax[0,0].set_ylabel(r"EW  ($\delta=0$)", fontsize=12)
    ax[1,0].set_ylabel(r"KPZ  ($\delta=\arctan5$)", fontsize=12)
    
    ax[0,0].legend()
    ax[0,1].legend()
    ax[1,0].legend()
    ax[1,1].legend()
                
    ax[0,0].set_xlim(-6,6)
    ax[0,1].set_xlim(-5,5)
    ax[1,0].set_xlim(-6,6)
    ax[1,1].set_xlim(-5,5)
    
    plt.tight_layout()
    if save:
        plt.savefig(f"Figures/Fluctuations/fluctuationLogPDFs_{L}.png", dpi=300)
    plt.show()