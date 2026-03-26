import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy.integrate import quad
import scienceplots

plt.style.use(["science","no-latex"])

### TO CLEAN ### 
# All the for loops over rows, since we now filter K, L, delta before (there should only be one row) 
# Remove M
# Remove L = row["L"] and K = row["K"]


# Scaling exponents
z_dict = {"TimeDep": {"EW": 2, "KPZ": 3/2}, "Columnar": {"EW": 2, "KPZ": 1.36}}
alpha_dict = {"TimeDep": {"EW": 1/2, "KPZ": 1/2}, "Columnar": {"EW": 3/2, "KPZ": 1.07}}
alpha_s_dict = {"TimeDep": {"EW": 1/2, "KPZ": 1/2}, "Columnar": {"EW": 3/2, "KPZ": 1.4}}  # anomalous scaling, only for columnar noise
alpha_loc_dict = {"TimeDep": {"EW": 1/2, "KPZ": 1/2}, "Columnar": {"EW": 1, "KPZ": 0.96}} # anomalous scaling, only for columnar noise
beta_dict = {"TimeDep": {"EW": alpha_dict["TimeDep"]["EW"]/z_dict["TimeDep"]["EW"], "KPZ": alpha_dict["TimeDep"]["KPZ"]/z_dict["TimeDep"]["KPZ"]}, 
             "Columnar": {"EW": 0.34, "KPZ": 0.93}}
delta_to_label = {"0": "EW", "atan5": "KPZ"}

# Markers for different system sizes
markerdict = {125: "^", 250: "x", 500: "o", 1000: "s"}

#####################
##### ROUGHNESS #####
#####################

def plot_roughness_collapse(avg_df, T, tx, Ks=[1,40], save=False):

    fig, ax = plt.subplots(2, 2, figsize=(12,10))
    # fig.suptitle("Roughness collapse under different noise types")

    fig.supxlabel(r"$\log (t/L^z)$")
    fig.supylabel(r"$\log (W(L,t) / L^\alpha)$")

    for i, (typenoise, typelabel) in enumerate(zip(["TimeDep", "Columnar"], ["Time-dependent noise", "Columnar noise"])
    ):

        for j, (delta, deltaname) in enumerate(zip([0, np.arctan(5)], ["0", "atan5"])):

            # select rows for this noise type and delta
            mask = mask = (
                (avg_df["typenoise"] == typenoise)
                & (avg_df["deltaname"] == deltaname)
                & (avg_df["T"] == T)
                & ((avg_df["K"] == Ks[0]) | (avg_df["K"] == Ks[1]))
                & (avg_df["L"] < 1000)
            )
            sub = avg_df[mask]

            # in case there are multiple (K, L) combos, plot them all
            for _, row in sub.iterrows():
                t = row["t"]
                mean_roughness = row["mean_roughness"]
                L = row["L"]
                K = row["K"]

                # Exponents
                z = z_dict[typenoise][delta_to_label[deltaname]]
                alpha = alpha_dict[typenoise][delta_to_label[deltaname]]

                # Collapse variables
                tLz = t / L**z
                WLa = mean_roughness / L**alpha
                
                ax[j,i].scatter(tLz, WLa, label=r"$L=$" + str(L), marker=markerdict[L], alpha=0.7)

            ax[j,i].set_xscale('log')
            ax[j,i].set_yscale('log')

    ax[0,0].set_title(f"Time-dependent ($K=${Ks[0]})")
    ax[0,1].set_title(f"Columnar ($K=${Ks[1]})")      
    ax[0,0].set_ylabel(r"EW  ($\delta=0$)", fontsize=12)
    ax[1,0].set_ylabel(r"KPZ  ($\delta=\arctan5$)", fontsize=12)

    # Scalings
    # ax[0,0].loglog(t[1:-8:] / L**(z_dict["TimeDep"]["EW"]), (t[1:-8:] / L ** (alpha_dict["TimeDep"]["EW"])) **(beta_dict["TimeDep"]["EW"]) /190,
    #             label=r"$\beta_{EW}=1/4$", linestyle="--", color="k")
    # ax[1,0].loglog(t[1:-8:] / L**(z_dict["TimeDep"]["KPZ"]), (t[1:-8:] / L ** (alpha_dict["TimeDep"]["KPZ"])) **(beta_dict["TimeDep"]["KPZ"]) /100,
    #             label=r"$\beta_{KPZ}=1/3$", linestyle="--", color="k")

    # ax[0,1].loglog(t[1:-12:] / L**(z_dict["Columnar"]["EW"]), (t[1:-12:] / L ** (alpha_dict["Columnar"]["EW"])) **(beta_dict["Columnar"]["EW"]) /80,
    #             label=r"$\beta_{EW}=3/4$", linestyle="--", color="k")
    # ax[1,1].loglog(t[1:-12:] / L**(z_dict["Columnar"]["KPZ"]), (t[1:-12:] / L ** (alpha_dict["Columnar"]["KPZ"])) **(beta_dict["Columnar"]["KPZ"]) /20,
    #             label=r"$\beta_{KPZ}\approx 0.7867$", linestyle="--", color="k")

    ax[0,0].legend()
    ax[0,1].legend()
    ax[1,0].legend()
    ax[1,1].legend()

    plt.tight_layout()
    if save:
        plt.savefig("Figures/Collapse/roughness_collapse.png", dpi=300)
    plt.show()


#############################
##### HEIGHT-DIFFERENCE #####
#############################

def plot_heightdifference_collapse(avg_df, L, T, tx, Ks=[1,40], t_intervals=3, save=False):
    
    fig, ax = plt.subplots(2, 2, figsize=(12,10))
    # fig.suptitle("Height-difference correlation collapse under different noise types, for $L=$" + str(L))

    fig.supxlabel(r"$\log (r/ t^{1/z})$")
    fig.supylabel(r"$\log (G(r,t)/ t^{2\alpha/z})$")

    for i, (typenoise, typelabel) in enumerate(zip(["TimeDep", "Columnar"], ["Time-dependent", "Columnar"])
    ):

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
                mean_heightheight = row["mean_heightheight"]
                L = row["L"]
                K = row["K"]

                # Exponents
                z = z_dict[typenoise][delta_to_label[deltaname]]
                alpha = alpha_dict[typenoise][delta_to_label[deltaname]]

                t_cross = tx[typenoise][delta_to_label[deltaname]] * L**z
                
                for ti in range(len(t))[::t_intervals]:

                    if t[ti] > 0 and t[ti] < t_cross:

                        rlen = mean_heightheight.shape[1] # we plot up to half of this since the system is periodic
                        rrange = np.array(range(rlen))

                        # Collapse variables
                        rt1z = rrange[:rlen//2] / t[ti]**(1/z)
                        Gt2az = mean_heightheight[ti,:rlen//2] / t[ti]**(2*alpha/z)

                        ax[j,i].scatter(rt1z, Gt2az, label=f"$t=${np.round(t[ti],2)}", marker=markerdict[L], alpha=0.7)
                    
                ax[j,i].set_yscale('log')
                ax[j,i].set_xscale('log')
                
                ax[j,i].text(0.8, 0.05, fr'$t_\star={np.round(t_cross,1)}$', fontsize=10, transform=ax[j,i].transAxes)

    ax[0,0].set_title(f"Time-dependent ($K=${Ks[0]})")
    ax[0,1].set_title(f"Columnar ($K=${Ks[1]})")      
    ax[0,0].set_ylabel(r"EW  ($\delta=0$)", fontsize=12)
    ax[1,0].set_ylabel(r"KPZ  ($\delta=\arctan5$)", fontsize=12)

    # scaling as r^(2\alpha) = r^1 for alpha = 1/2 (EW/KPZ)
    ax[0,0].loglog(rt1z[1:int(rlen/64)]/2, (rt1z[1:int(rlen/64)]/2)**(2*alpha_loc_dict["TimeDep"]["EW"]) /140, label=r"$\alpha_{EW}=1/2$", linestyle="--", color="k")
    ax[1,0].loglog(rt1z[1:int(rlen/64)]/8, (rt1z[1:int(rlen/64)]/8)**(2*alpha_loc_dict["TimeDep"]["KPZ"])/20, label=r"$\alpha_{KPZ}=1/2$", linestyle="--", color="k")

    # scaling as r^(2\alpha) = r^2 for alpha_loc approx 1 (EW/KPZ)
    ax[0,1].loglog(rt1z[2:int(rlen/32)], rt1z[2:int(rlen/32)]**(2*alpha_loc_dict["Columnar"]["EW"])/800, label=r"$\alpha_{loc}^{EW}=1$", linestyle="--", color="k")
    ax[1,1].loglog(rt1z[2:int(rlen/64)], rt1z[2:int(rlen/64)]**(2*alpha_loc_dict["Columnar"]["KPZ"])/80, label=r"$\alpha_{loc}^{KPZ}=0.96$", linestyle="--", color="k")

    ax[0,0].legend()
    ax[0,1].legend()
    ax[1,0].legend()
    ax[1,1].legend()

    plt.tight_layout()
    if save:
        plt.savefig(f"Figures/Collapse/heighdifference_collapse_{L}.png", dpi=300)
    plt.show()


############################
##### STRUCTURE FACTOR #####
############################

def plot_structurefactor_collapse(avg_df, L, T, tx, Ks=[1,40], t_intervals=3, save=False):

    fig, ax = plt.subplots(2, 2, figsize=(12,10))
    # fig.suptitle("Structure factor collapse under different noise types, for $L=$" + str(L))

    fig.supxlabel(r"$\log(y)=\log (k t^{1/z})$")
    fig.supylabel(r"$\log (S(k,t) k^{2\alpha+1})$")

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
                mean_structurefactor = row["mean_structurefactor"]
                L = row["L"]
                K = row["K"]

                # Exponents
                z = z_dict[typenoise][delta_to_label[deltaname]]
                alpha = alpha_dict[typenoise][delta_to_label[deltaname]]

                t_cross = tx[typenoise][delta_to_label[deltaname]] * L**(z_dict[typenoise][delta_to_label[deltaname]])

                for ti in range(len(t))[::t_intervals]:

                    if t[ti] > 0 and t[ti] < t_cross:

                        klen = mean_structurefactor.shape[1] # we only plot up to half due to nyquist
                        krange = 2*np.pi*np.fft.fftfreq(klen)

                        # Collapse variables
                        kt1z = krange[:int(klen/2)] * t[ti]**(1/z)
                        Sk2a = mean_structurefactor[ti,:int(klen/2)] * krange[:int(klen/2)]**(2*alpha+1)

                        ax[j,i].scatter(kt1z, Sk2a, label=f"$t=${np.round(t[ti],2)}", marker=markerdict[L], alpha=0.7)

                ax[j,i].set_yscale('log')
                ax[j,i].set_xscale('log')
                
                ax[j,i].text(0.05, 0.95, fr'$t_\star={np.round(t_cross,1)}$', fontsize=10, transform=ax[j,i].transAxes)

    ax[0,0].set_title(f"Time-dependent ($K=${Ks[0]})")
    ax[0,1].set_title(f"Columnar ($K=${Ks[1]})")    
    ax[0,0].set_ylabel(r"EW  ($\delta=0$)", fontsize=12)
    ax[1,0].set_ylabel(r"KPZ  ($\delta=\arctan5$)", fontsize=12)
    
    # initial scaling as y^(2*alpha+1) = k^2 for alpha = 1/2 (EW/KPZ)
    kt1z_init = kt1z/10
    ax[0,0].loglog(kt1z_init[1:int(klen/64)], kt1z_init[1:int(klen/64)]**(2*alpha_dict["TimeDep"]["EW"]+1) / 50, label=r"$\alpha^{EW}=1/2$", linestyle="--", color="k")
    ax[1,0].loglog(kt1z_init[1:int(klen/32)], kt1z_init[1:int(klen/32)]**(2*alpha_dict["TimeDep"]["KPZ"]+1) / 30, label=r"$\alpha^{KPZ}=1/2$", linestyle="--", color="k")

    # initial scaling as y^(2*alpha+1) for alpha approx 3/2 (EW) or 1.07 (KPZ)
    ax[0,1].loglog(kt1z_init[1:int(klen/200)], kt1z_init[1:int(klen/200)]**(2*alpha_dict["Columnar"]["EW"]+1) / 1, label=r"$\alpha^{EW}=3/2$", linestyle="--", color="k")
    ax[1,1].loglog(kt1z_init[1:int(klen/128)], kt1z_init[1:int(klen/128)]**(2*alpha_dict["Columnar"]["KPZ"]+1) / 1, label=r"$\alpha^{KPZ}=1.07$", linestyle="--", color="k")

    # intermediate scaling as y^(2*(alpha-alpha_s)) = k^2 for alpha = 1/2 (EW/KPZ)
    ax[0,0].loglog(kt1z[3:int(klen/8)], kt1z[3:int(klen/8)]**(2*alpha_dict["TimeDep"]["EW"]-2*alpha_s_dict["TimeDep"]["EW"]) / 300, label=r"$\alpha_s^{EW}=1/2$", linestyle=":", color="k")
    ax[1,0].loglog(kt1z[8:int(klen/2)], kt1z[8:int(klen/2)]**(2*alpha_dict["TimeDep"]["KPZ"]-2*alpha_s_dict["TimeDep"]["KPZ"]) / 50, label=r"$\alpha_s^{KPZ}=1/2$", linestyle=":", color="k")

    # intermediate scaling as y^(2*(alpha-alpha_s)) for alpha_s approx 3/2 (EW) or 1.4 (KPZ)
    ax[0,1].loglog(kt1z[2:int(klen/16)], kt1z[2:int(klen/16)]**(2*alpha_dict["Columnar"]["EW"]-2*alpha_s_dict["Columnar"]["EW"]) / 10000, label=r"$\alpha_s^{EW}=3/2$", linestyle=":", color="k")
    ax[1,1].loglog(kt1z[4:int(klen/32)], kt1z[4:int(klen/32)]**(2*alpha_dict["Columnar"]["KPZ"]-2*alpha_s_dict["Columnar"]["KPZ"]) / 200, label=r"$\alpha_s^{KPZ}=1.4$", linestyle=":", color="k")

    ax[0,0].text(0.1, 0.6, r"$y^{2\alpha_s+1}$", fontsize=12, transform=ax[0,0].transAxes)
    ax[0,1].text(0.1, 0.7, r"$y^{2\alpha_s+1}$", fontsize=12, transform=ax[0,1].transAxes)
    ax[1,0].text(0.1, 0.6, r"$y^{2\alpha_s+1}$", fontsize=12, transform=ax[1,0].transAxes)
    ax[1,1].text(0.1, 0.7, r"$y^{2\alpha_s+1}$", fontsize=12, transform=ax[1,1].transAxes)

    ax[0,0].text(0.5, 0.75, r"$y^{2(\alpha-\alpha_s)}$", fontsize=12, transform=ax[0,0].transAxes)
    ax[0,1].text(0.5, 0.7, r"$y^{2(\alpha-\alpha_s)}$", fontsize=12, transform=ax[0,1].transAxes)
    ax[1,0].text(0.5, 0.75, r"$y^{2(\alpha-\alpha_s)}$", fontsize=12, transform=ax[1,0].transAxes)
    ax[1,1].text(0.4, 0.7, r"$y^{2(\alpha-\alpha_s)}$", fontsize=12, transform=ax[1,1].transAxes)

    ax[0,0].legend()
    ax[0,1].legend()
    ax[1,0].legend()
    ax[1,1].legend()

    plt.tight_layout()
    if save:
        plt.savefig(f"Figures/Collapse/structurefactor_collapse_{L}.png", dpi=300)
    plt.show()


############################
##### PHASE COVARIANCE #####
############################


def covariance_constants(tti, C_Phi_t, beta, z, fun, x0=0.5):
    a_1 = C_Phi_t[0] / (tti**(2*beta) * fun[0])
    target = a_1 * tti**(2*beta) * fun[x0]

    # find the FIRST r where C drops below target (C is decreasing initially)
    C = C_Phi_t[:len(C_Phi_t)//2]
    rr = np.arange(C.size)
    
    # If target is above C(0) or below the min, no meaningful crossing in-range
    if target >= C[0] or target <= C.min():
        return np.nan, np.nan
    
    # first index where C <= target
    idx = np.argmax(C <= target)  # returns 0 if never true; guarded by check above
    
    # bracket between idx-1 (above target) and idx (below target)
    i0 = idx - 1
    i1 = idx
    
    C0, C1 = C[i0], C[i1]
    r0, r1 = float(i0), float(i1)
    
    # linear interpolation
    r_intermediate = r0 + (target - C0) * (r1 - r0) / (C1 - C0)

    a_2 = (tti**(1/z)) * x0 / r_intermediate
    return a_1, a_2


with open("../airy_1_values.txt", 'r') as f:
    Airy_1 = {float(line.split(" ")[0]): float(line.split(" ")[1]) for line in f.readlines()}


def larkin_inv_ft(x, eps=1e-6, rtol=1e-10, atol=1e-12, limit=300, sigma=0.102):
    """
    Computes  F^{-1}[ ((1 - e^{-k^2})^2 / k^4) ](x)
    using the convention:
        F^{-1}[g](x) = (1/2π)∫_{-∞}^{∞} e^{ikx} g(k) dk
                     = (1/π) ∫_0^∞ cos(kx) g(k) dk   (g even)

    Parameters
    ----------
    x : float
        Evaluation point.
    eps : float
        Threshold for using the small-k series to avoid 0/0.
    rtol, atol : float
        quad tolerances.
    limit : int
        quad subinterval limit.

    Returns
    -------
    float
        Value of the inverse transform at x.
    """
    x = x/1.35

    def g(kappa):
        ak = abs(kappa)
        if ak < eps:
            k2 = kappa*kappa
            return 1.0 - k2 + (7.0/12.0)*k2*k2  # series at κ=0
        return (1.0 - np.exp(-kappa*kappa))**2 / (kappa**4)

    val, _ = quad(lambda kappa: np.cos(2*np.pi*kappa*x) * g(kappa), 0.0, np.inf,
                  epsrel=rtol, epsabs=atol, limit=limit)
    return 4*sigma*val  # (1/2π) over R -> (1/π) cosine integral


def plot_phasecovariance_collapse(avg_df, L, T, tx, Ks=[1,40], t_intervals=3, nu=40.0, dx=1.0, save=False):

    fig, ax = plt.subplots(2, 2, figsize=(10,10))
    # fig.suptitle("Phase covariance collapse under different noise types, for $L=$" + str(L))

    fig.supxlabel(r"$a_2 r/ t^{1/z}$")
    fig.supylabel(r"$C(r,t)/ t^{2\beta} a_1$")

    a_1_dict = {}
    a_2_dict = {}

    for i, (typenoise, typelabel) in enumerate(zip(["TimeDep", "Columnar"], ["Time-dependent", "Columnar"])
    ):

        a_1_dict[typenoise] = {}
        a_2_dict[typenoise] = {}

        for j, (delta, deltaname) in enumerate(zip([0, np.arctan(5)], ["0", "atan5"])):

            a_1_dict[typenoise][deltaname] = []
            a_2_dict[typenoise][deltaname] = []
            
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
                mean_phasecovariance = row["mean_phasecovariance"]
                L = row["L"]
                K = row["K"]

                # Exponents
                z = z_dict[typenoise][delta_to_label[deltaname]]
                alpha = alpha_dict[typenoise][delta_to_label[deltaname]]
                beta = beta_dict[typenoise][delta_to_label[deltaname]]
            
                t_cross = tx[typenoise][delta_to_label[deltaname]] * L**(z_dict[typenoise][delta_to_label[deltaname]])

                for ti in range(len(t))[::t_intervals]:

                    if t[ti] > 0 and t[ti] < t_cross:

                        rlen = mean_phasecovariance.shape[1] # we plot up to half of this since the system is periodic
                        rrange = np.array(range(rlen))

                        a_1, a_2 = covariance_constants(t[ti], mean_phasecovariance[ti,:], beta, z, fun=Airy_1) 
                        a_1_dict[typenoise][deltaname].append(a_1)
                        a_2_dict[typenoise][deltaname].append(a_2)

                        # Collapse variables
                        a2rt1z = rrange[:rlen//2] * a_2 / t[ti]**(1/z_dict[typenoise][delta_to_label[deltaname]])
                        Ct2ba1 = mean_phasecovariance[ti,:rlen//2] / (a_1 * t[ti]**(2*beta_dict[typenoise][delta_to_label[deltaname]]))

                        ax[j,i].scatter(a2rt1z, Ct2ba1, label=f"$t=${np.round(t[ti],2)}", marker=markerdict[L], alpha=0.7)

                ax[j,i].text(0.8, 0.95, fr'$t_\star={np.round(t_cross,1)}$', fontsize=10, transform=ax[j,i].transAxes)

                # Airy_1 and F^{-1}(Larkin) values 
                ax[j,i].plot(list(Airy_1.keys()), list(Airy_1.values()),
                             lw=2, color="white", linestyle="--", label=r"Airy$_1$",
                             path_effects=[pe.Stroke(linewidth=3, foreground='k'), pe.Normal()])
                ax[j,i].plot(np.linspace(0, 1, 400), [larkin_inv_ft(rr) for rr in np.linspace(0, 1, 400)],
                             lw=2, color="white", label="Larkin",
                             path_effects=[pe.Stroke(linewidth=3, foreground='k'), pe.Normal()]) 

    ax[0,0].set_xlim(0,1)
    ax[0,1].set_xlim(0,1)
    ax[1,0].set_xlim(0,1)
    ax[1,1].set_xlim(0,1)

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
        plt.savefig(f"Figures/Collapse/phasecovariance_collapse_{L}.png", dpi=300)
    plt.show()

    return a_1_dict, a_2_dict