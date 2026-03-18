import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import scienceplots

plt.style.use(["science","no-latex"])

import sys
sys.path.append('..')
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
             "Columnar": {"EW": alpha_dict["Columnar"]["EW"]/z_dict["Columnar"]["EW"], "KPZ": alpha_dict["Columnar"]["KPZ"]/z_dict["Columnar"]["KPZ"]}}
delta_to_label = {"0": "EW", "atan5": "KPZ"}

# Markers for different system sizes
markerdict = {125: "^", 250: "x", 500: "o", 1000: "s"}

# Vertical lines' colors for different system sizes (roughness evolution plot)
vlinecolordict = {125: "tab:blue", 250: "tab:green", 500: "tab:orange", 1000: "tab:red"}

#####################
##### ROUGHNESS #####
#####################

def plot_roughness_evolution(avg_df, T, tx, Ks=[1,40], save=False):

    fig, ax = plt.subplots(2, 2, figsize=(12,10))
    # fig.suptitle("Roughness evolution under different noise types")

    fig.supxlabel(r"$\log t$")
    fig.supylabel(r"$\log W(L,t)$")

    for i, (typenoise, typelabel) in enumerate(zip(["TimeDep", "Columnar"], ["Time-dependent noise", "Columnar noise"])
    ):
        print("#", typenoise)
        for j, (delta, deltaname) in enumerate(zip([0, np.arctan(5)], ["0", "atan5"])):

            print("##", deltaname)
            # select rows for this noise type and delta
            mask = mask = (
                (avg_df["typenoise"] == typenoise)
                & (avg_df["deltaname"] == deltaname)
                & (avg_df["T"] == T)
                & ((avg_df["K"] == Ks[0]) | (avg_df["K"] == Ks[1]))
                # & (avg_df["L"] < 1000)
            )
            sub_df = avg_df[mask]

            # in case there are multiple (K, L) combos, plot them all
            for _, row in sub_df.iterrows():
                t = row["t"]
                mean_roughness = row["mean_roughness"]
                L = row["L"]
                K = row["K"]
                
                t_cross = tx[typenoise][delta_to_label[deltaname]] * L**(z_dict[typenoise][delta_to_label[deltaname]])
                
                print("\t", fr"$t_\star$({L}) = {t_cross}.")

                ax[j,i].scatter(t, mean_roughness, label=r"$L=$" + str(L), marker=markerdict[L])

                ax[j,i].vlines(t_cross, ymin=np.min(mean_roughness), ymax=np.max(mean_roughness),
                                colors=vlinecolordict[L], linestyles='dotted', label=r"$t_\star=$"+f"{np.round(t_cross,1)}")
                
            ax[j,i].set_xscale('log')
            ax[j,i].set_yscale('log')

    ax[0,0].set_title(f"Time-dependent ($K=${Ks[0]})")
    ax[0,1].set_title(f"Columnar ($K=${Ks[1]})")      
    ax[0,0].set_ylabel(r"EW  ($\delta=0$)", fontsize=12)
    ax[1,0].set_ylabel(r"KPZ  ($\delta=\arctan5$)", fontsize=12)

    # Scalings
    ax[0,0].loglog(t[1:-5:], t[1:-5:]**(beta_dict["TimeDep"]["EW"]) /15, label=r"$\beta_{EW}=1/4$", linestyle="--", color="k")
    ax[1,0].loglog(t[1:-5:], t[1:-5:]**(beta_dict["TimeDep"]["KPZ"]) /10, label=r"$\beta_{KPZ}=1/3$", linestyle="--", color="k")

    ax[0,1].loglog(t[1:-12:], t[1:-12:]**(beta_dict["Columnar"]["EW"]) / 5, label=r"$\beta_{EW}=3/4$", linestyle="--", color="k")
    ax[1,1].loglog(t[1:-12:], t[1:-12:]**(beta_dict["Columnar"]["KPZ"]) / 4, label=r"$\beta_{KPZ}\approx 0.7867$", linestyle="--", color="k")

    ax[0,0].legend()
    ax[0,1].legend()
    ax[1,0].legend()
    ax[1,1].legend()

    plt.tight_layout()
    if save:
        plt.savefig("Figures/Evolution/roughness_evolution.png", dpi=300)
    plt.show()


#############################
##### HEIGHT-DIFFERENCE #####
#############################

def plot_heightdifference_evolution(avg_df, L, T, tx, Ks=[1,40], t_intervals=3, save=False):
    
    fig, ax = plt.subplots(2, 2, figsize=(12,10))
    # fig.suptitle("Height-difference correlation evolution under different noise types, for $L=$" + str(L))

    fig.supxlabel(r"$\log r$")
    fig.supylabel(r"$\log G(r,t)$")

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
                
                t_cross = tx[typenoise][delta_to_label[deltaname]] * L**(z_dict[typenoise][delta_to_label[deltaname]])

                for ti in range(len(t))[::t_intervals]:

                    if t[ti] > 0 and t[ti] < t_cross:

                        rlen = mean_heightheight.shape[1] # we plot up to half of this since the system is periodic
                        rrange = np.array(range(rlen))
                        ax[j,i].scatter(rrange[:rlen//2], mean_heightheight[ti,:rlen//2], label=f"$t=${np.round(t[ti],2)}", marker=markerdict[L])

            ax[j,i].set_xscale('log')
            ax[j,i].set_yscale('log')

            ax[j,i].text(0.8, 0.05, fr'$t_\star={np.round(t_cross,1)}$', fontsize=10, transform=ax[j,i].transAxes)

    ax[0,0].set_title(f"Time-dependent ($K=${Ks[0]})")
    ax[0,1].set_title(f"Columnar ($K=${Ks[1]})")     
    ax[0,0].set_ylabel(r"EW  ($\delta=0$)", fontsize=12)
    ax[1,0].set_ylabel(r"KPZ  ($\delta=\arctan5$)", fontsize=12)

    # scaling as r^(2\alpha) = r^1 for alpha = 1/2 (EW/KPZ)
    ax[0,0].loglog(rrange[1:int(rlen/16)], rrange[1:int(rlen/16)]**(2*alpha_loc_dict["TimeDep"]["EW"])/140, label=r"$\alpha_{EW}=1/2$", linestyle="--", color="k")
    ax[1,0].loglog(rrange[1:int(rlen/16)], rrange[1:int(rlen/16)]**(2*alpha_loc_dict["TimeDep"]["KPZ"])/20, label=r"$\alpha_{KPZ}=1/2$", linestyle="--", color="k")

    # scaling as r^(2\alpha) = r^2 for alpha_loc approx 1 (EW/KPZ)
    ax[0,1].loglog(rrange[1:int(rlen/16)], rrange[1:int(rlen/16)]**(2*alpha_loc_dict["Columnar"]["EW"])/50, label=r"$\alpha_{loc}^{EW}=1$", linestyle="--", color="k")
    ax[1,1].loglog(rrange[1:int(rlen/32)], rrange[1:int(rlen/32)]**(2*alpha_loc_dict["Columnar"]["KPZ"])/40, label=r"$\alpha_{loc}^{KPZ}=0.96$", linestyle="--", color="k")

    ax[0,0].legend()
    ax[0,1].legend()
    ax[1,0].legend()
    ax[1,1].legend()

    plt.tight_layout()
    if save:
        plt.savefig(f"Figures/Evolution/heighdifference_evolution_{L}.png", dpi=300)
    plt.show()


############################
##### STRUCTURE FACTOR #####
############################

def analytical_Larkin_structurefactor(k, t, sigma, nu, d=1):
    """
    Analytical structure factor for the Larkin model (linear theory),
    equation (28) in the 2023 paper.
    """

    S_phi = ((2 * np.pi)**d * 2 * sigma / (nu**2 * k**4)) * (1 - np.exp(- nu * k**2 * t))**2

    return S_phi


def plot_structurefactor_evolution(avg_df, L, T, tx, Ks=[1,40], t_intervals=3, analytical=False, sigma=1, save=False):

    fig, ax = plt.subplots(2, 2, figsize=(12,10))
    # fig.suptitle("Structure factor under different noise types, for $L=$" + str(L))

    fig.supxlabel(r"$\log k$")
    fig.supylabel(r"$\log S(k,t)$")

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

                t_cross = tx[typenoise][delta_to_label[deltaname]] * L**(z_dict[typenoise][delta_to_label[deltaname]])

                for ti in range(len(t))[::t_intervals]:

                    if t[ti] > 0 and t[ti] < t_cross:

                        klen = mean_structurefactor.shape[1]
                        krange = 2*np.pi*np.fft.fftfreq(klen)
                        
                        # we only plot up to half due to nyquist
                        ax[j,i].scatter(krange[:int(klen/2)], mean_structurefactor[ti,:int(klen/2)], label=f"$t=${np.round(t[ti],2)}", marker=markerdict[L])
                    
                ax[j,i].set_yscale('log')
                ax[j,i].set_xscale('log')
                
                ax[j,i].text(0.05, 0.05, fr'$t_\star={np.round(t_cross,1)}$', fontsize=10, transform=ax[j,i].transAxes)

    ax[0,0].set_title(f"Time-dependent ($K=${Ks[0]})")
    ax[0,1].set_title(f"Columnar ($K=${Ks[1]})")    
    ax[0,0].set_ylabel(r"EW  ($\delta=0$)", fontsize=12)
    ax[1,0].set_ylabel(r"KPZ  ($\delta=\arctan5$)", fontsize=12)

    # scaling as k^(-2*alpha-1) = k^2 for alpha = 1/2 (EW/KPZ)
    ax[0,0].loglog(krange[3:int(klen/4)], krange[3:int(klen/4)]**(-2*alpha_s_dict["TimeDep"]["EW"]-1) / 100, label=r"$\alpha_s^{EW}=1/2$", linestyle="--", color="k")
    ax[1,0].loglog(krange[3:int(klen/4)], krange[3:int(klen/4)]**(-2*alpha_s_dict["TimeDep"]["KPZ"]-1) / 10, label=r"$\alpha_s^{KPZ}=1/2$", linestyle="--", color="k")

    # # scaling as k^(-2*alpha_s-1) for alpha_s approx 3/2 (EW) or 1.4 (KPZ)
    ax[0,1].loglog(krange[3:int(klen/4)], krange[3:int(klen/4)]**(-2*alpha_s_dict["Columnar"]["EW"]-1) / 1000, label=r"$\alpha_s^{EW}=3/2$", linestyle="--", color="k")
    ax[1,1].loglog(krange[10:int(klen/4)], krange[10:int(klen/4)]**(-2*alpha_s_dict["Columnar"]["KPZ"]-1) / 50, label=r"$\alpha_s^{KPZ}=1.4$", linestyle="--", color="k")

    ax[1,0].set_ylim(0,1e3)
    ax[1,1].set_ylim(0,1e4)

    ax[0,0].text(0.5, 0.6, r"$k^{-2\alpha_s+1}$", fontsize=12, transform=ax[0,0].transAxes)
    ax[1,0].text(0.5, 0.65, r"$k^{-2\alpha_s+1}$", fontsize=12, transform=ax[1,0].transAxes)
    ax[0,1].text(0.5, 0.6, r"$k^{-2\alpha_s+1}$", fontsize=12, transform=ax[0,1].transAxes)
    ax[1,1].text(0.6, 0.6, r"$k^{-2\alpha_s+1}$", fontsize=12, transform=ax[1,1].transAxes)

    handles, labels = ax[0,1].get_legend_handles_labels()

    # Add analytical Larkin structure factor for comparison to Columnar EW  case
    if analytical == True:
        for ti in range(len(t))[::t_intervals]:
            t_cross = tx["Columnar"]["EW"] * L**(z_dict["Columnar"]["EW"])
            if t[ti] > 0 and t[ti] < t_cross:
                Sphi = analytical_Larkin_structurefactor(krange, t[ti], sigma=sigma, nu=Ks[1], d=1)
                ax[0,1].loglog(krange[1:int(klen/4)], Sphi[1:int(klen/4)], linestyle=":")#, label=f"Anaytical (Larkin) $t={np.round(t[ti],1)}$")
                        
        handles.append(Line2D([0], [0], color='black', linestyle=':'))
        labels.append('Analytical (Larkin)')

    ax[0,0].legend()
    ax[1,0].legend()
    ax[0,1].legend(handles=handles, labels=labels)
    ax[1,1].legend()

    plt.tight_layout()
    if save:
        plt.savefig(f"Figures/Evolution/structurefactor_evolution_{L}.png", dpi=300)
    plt.show()

############################
##### PHASE COVARIANCE #####
############################

def plot_phasecovariance_evolution(avg_df, L, T, tx, Ks=[1,40], t_intervals=3, save=False):

    fig, ax = plt.subplots(2, 2, figsize=(10,10))
    # fig.suptitle("Phase covariance evolution under different noise types, for $L=$" + str(L))

    fig.supxlabel(r"$r$")
    fig.supylabel(r"$C(r,t)$")

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
                mean_phasecovariance = row["mean_phasecovariance"]
                L = row["L"]
                K = row["K"]

                t_cross = tx[typenoise][delta_to_label[deltaname]] * L**(z_dict[typenoise][delta_to_label[deltaname]])

                for ti in range(len(t))[::t_intervals]:

                    if t[ti] > 0 and t[ti] < t_cross:
                        rlen = mean_phasecovariance.shape[1] # we plot up to half of this since the system is periodic
                        rrange = np.array(range(rlen))
                        ax[j,i].scatter(rrange[:rlen//2], mean_phasecovariance[ti,:rlen//2], label=f"$t=${np.round(t[ti],2)}", marker=markerdict[L])
                    
                ax[j,i].text(0.8, 0.95, fr'$t_\star={np.round(t_cross,1)}$', fontsize=10, transform=ax[j,i].transAxes)

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
        plt.savefig(f"Figures/Evolution/phasecovariance_evolution_{L}.png", dpi=300)
    plt.show()
