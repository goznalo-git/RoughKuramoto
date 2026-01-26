import numpy as np
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(["science","no-latex"])


# Scaling exponents
z_dict = {"TimeDep": {"EW": 2, "KPZ": 3/2}, "Columnar": {"EW": 2, "KPZ": 1.36}}
alpha_dict = {"TimeDep": {"EW": 1/2, "KPZ": 1/2}, "Columnar": {"EW": 3/2, "KPZ": 1.07}}
alpha_s_dict = {"TimeDep": {"EW": 1/2, "KPZ": 1/2}, "Columnar": {"EW": 3/2, "KPZ": 1.4}}  # anomalous scaling, only for columnar noise
alpha_loc_dict = {"TimeDep": {"EW": 1/2, "KPZ": 1/2}, "Columnar": {"EW": 1, "KPZ": 0.96}} # anomalous scaling, only for columnar noise
beta_dict = {"TimeDep": {"EW": alpha_dict["TimeDep"]["EW"]/z_dict["TimeDep"]["EW"], "KPZ": alpha_dict["TimeDep"]["KPZ"]/z_dict["TimeDep"]["KPZ"]}, 
             "Columnar": {"EW": alpha_dict["Columnar"]["EW"]/z_dict["Columnar"]["EW"], "KPZ": alpha_dict["Columnar"]["KPZ"]/z_dict["Columnar"]["KPZ"]}}
delta_to_label = {"0": "EW", "atan5": "KPZ"}

# Markers for different system sizes
markerdict = {500: "o", 250: "x", 125: "^", 1000: "s"}


######################################
##### ROUGHNESS AND ITS COLLAPSE #####
######################################

def plot_roughness_evolution(avg_df, T, tx): 

    fig, ax = plt.subplots(2, 2, figsize=(10,10))
    fig.suptitle("Roughness evolution under different noise types")

    fig.supxlabel(r"$\log t$")
    fig.supylabel(r"$\log W$")

    for i, (typenoise, typelabel) in enumerate(zip(["TimeDep", "Columnar"], ["Time-dependent noise", "Columnar noise"])
    ):

        for j, (delta, deltaname) in enumerate(zip([0, np.arctan(5)], ["0", "atan5"])):

            # select rows for this noise type and delta
            mask = mask = (
                (avg_df["typenoise"] == typenoise)
                & (avg_df["deltaname"] == deltaname)
                & ((avg_df["K"] == 1) | (avg_df["K"] == 20))
                & (avg_df["T"] == T)
                & (avg_df["L"] < 1000)
            )
            sub_df = avg_df[mask]

            # in case there are multiple (K, L) combos, plot them all
            for _, row in sub_df.iterrows():
                t = row["t"]
                mean_roughness = row["mean_roughness"]
                L = row["L"]
                K = row["K"]
                M = row["M"]
                
                t_cross = tx[typenoise][delta_to_label[deltaname]] * L**(z_dict[typenoise][delta_to_label[deltaname]])

                ax[j,i].scatter(t, mean_roughness, label=r"$L=$" + str(L), marker=markerdict[L])

                ax[j,i].vlines(t_cross, ymin=np.min(mean_roughness), ymax=np.max(mean_roughness),
                                colors='gray', linestyles='dotted', label=r"$t_\star=$"+f"{np.round(t_cross,1)}")

            ax[j,i].set_title(typelabel+ r" ($\delta$="+f"{deltaname}, K={K}, {M} seeds)")

            ax[j,i].set_xscale('log')
            ax[j,i].set_yscale('log')



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
    plt.show()


def plot_roughness_collapse(avg_df, T):

    fig, ax = plt.subplots(2, 2, figsize=(10,10))
    fig.suptitle("Roughness collapse under different noise types")

    fig.supxlabel(r"$\log (t/L^z)$")
    fig.supylabel(r"$\log (W / L^\alpha)$")

    for i, (typenoise, typelabel) in enumerate(zip(["TimeDep", "Columnar"], ["Time-dependent noise", "Columnar noise"])
    ):

        for j, (delta, deltaname) in enumerate(zip([0, np.arctan(5)], ["0", "atan5"])):

            # select rows for this noise type and delta
            mask = mask = (
                (avg_df["typenoise"] == typenoise)
                & (avg_df["deltaname"] == deltaname)
                & ((avg_df["K"] == 1) | (avg_df["K"] == 20))
                & (avg_df["T"] == T)
                & (avg_df["L"] < 1000)
            )
            sub = avg_df[mask]

            # in case there are multiple (K, L) combos, plot them all
            for _, row in sub.iterrows():
                t = row["t"]
                mean_roughness = row["mean_roughness"]
                L = row["L"]
                K = row["K"]
                M = row["M"]

                # Collapse variables
                tLz = t / L**(z_dict[typenoise][delta_to_label[deltaname]])
                WLa = mean_roughness / L**(alpha_dict[typenoise][delta_to_label[deltaname]])

                if deltaname == "atan5":
                    WLa *= 5
                
                ax[j,i].scatter(tLz, WLa, label=r"$\delta=$" + deltaname + r", $L=$" + str(L), marker=markerdict[L])
            
            ax[j,i].set_title(typelabel+ r" ($\delta$="+f"{deltaname}, K={K}, {M} seeds)")

            ax[j,i].set_xscale('log')
            ax[j,i].set_yscale('log')

    # Scalings
    ax[0,0].loglog(t[1:-8:] / L**(z_dict["TimeDep"]["EW"]), (t[1:-8:] / L ** (alpha_dict["TimeDep"]["EW"])) **(beta_dict["TimeDep"]["EW"]) /125,
                label=r"$\beta_{EW}=1/4$", linestyle="--", color="k")
    ax[1,0].loglog(t[1:-8:] / L**(z_dict["TimeDep"]["KPZ"]), (t[1:-8:] / L ** (alpha_dict["TimeDep"]["KPZ"])) **(beta_dict["TimeDep"]["KPZ"]) /12,
                label=r"$\beta_{KPZ}=1/3$", linestyle="--", color="k")

    ax[0,1].loglog(t[1:-12:] / L**(z_dict["Columnar"]["EW"]), (t[1:-12:] / L ** (alpha_dict["Columnar"]["EW"])) **(beta_dict["Columnar"]["EW"]) /50,
                label=r"$\beta_{EW}=3/4$", linestyle="--", color="k")
    ax[1,1].loglog(t[1:-12:] / L**(z_dict["Columnar"]["KPZ"]), (t[1:-12:] / L ** (alpha_dict["Columnar"]["KPZ"])) **(beta_dict["Columnar"]["KPZ"]) /3,
                label=r"$\beta_{KPZ}\approx 0.7867$", linestyle="--", color="k")

    ax[0,0].legend()
    ax[0,1].legend()
    ax[1,0].legend()
    ax[1,1].legend()

    plt.tight_layout()
    plt.show()



##############################################
##### HEIGHT-DIFFERENCE AND ITS COLLAPSE #####
##############################################

def plot_heightdifference_evolution(avg_df, L, T, tx):
    
    fig, ax = plt.subplots(2, 2, figsize=(10,10))
    fig.suptitle("Height-difference correlation evolution under different noise types, for $L=$" + str(L))

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
                & ((avg_df["K"] == 1) | (avg_df["K"] == 20))
            )
            sub = avg_df[mask]

            # in case there are multiple (K, L) combos, plot them all
            for _, row in sub.iterrows():
                t = row["t"]            
                mean_heightheight = row["mean_heightheight"]
                L = row["L"]
                K = row["K"]
                M = row["M"]

                t_cross = tx[typenoise][delta_to_label[deltaname]] * L**(z_dict[typenoise][delta_to_label[deltaname]])

                for ti in range(mean_heightheight.shape[0])[::4]:

                    if t[ti] > 0 and t[ti] < t_cross:

                        rlen = mean_heightheight.shape[1] # we plot up to half of this since the system is periodic
                        rrange = np.array(range(rlen))
                        ax[j,i].scatter(rrange[:int(rlen/2)], mean_heightheight[ti,:int(rlen/2)], label=f"$t=${np.round(t[ti],2)}", marker=markerdict[L])

                ax[j,i].set_title(typelabel+ r" ($\delta$="+f"{deltaname}, K={K}, {M} seeds). " + r"$t_\star$=" + f"{np.round(t_cross,1)}")

            ax[j,i].set_xscale('log')
            ax[j,i].set_yscale('log')

    # scaling as r^(2\alpha) = r^1 for alpha = 1/2 (EW/KPZ)
    ax[0,0].loglog(rrange[1:int(rlen/16)], rrange[1:int(rlen/16)]**(2*alpha_loc_dict["TimeDep"]["EW"])/140, label=r"$\alpha_{EW}=1/2$", linestyle="--", color="k")
    ax[1,0].loglog(rrange[1:int(rlen/16)], rrange[1:int(rlen/16)]**(2*alpha_loc_dict["TimeDep"]["KPZ"])/20, label=r"$\alpha_{KPZ}=1/2$", linestyle="--", color="k")

    # scaling as r^(2\alpha) = r^2 for alpha_loc approx 1 (EW/KPZ)
    ax[0,1].loglog(rrange[1:int(rlen/16)], rrange[1:int(rlen/16)]**(2*alpha_loc_dict["Columnar"]["EW"])/20, label=r"$\alpha_{loc}^{EW}=1$", linestyle="--", color="k")
    ax[1,1].loglog(rrange[1:int(rlen/16)], rrange[1:int(rlen/16)]**(2*alpha_loc_dict["Columnar"]["KPZ"])/20, label=r"$\alpha_{loc}^{KPZ}=0.96$", linestyle="--", color="k")

    ax[0,0].legend()
    ax[0,1].legend()
    ax[1,0].legend()
    ax[1,1].legend()

    plt.tight_layout()
    plt.show()


def plot_heightdifference_collapse(avg_df, L, T, tx):
    
    fig, ax = plt.subplots(2, 2, figsize=(10,10))
    fig.suptitle("Height-difference correlation collapse under different noise types, for $L=$" + str(L))

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
                & ((avg_df["K"] == 1) | (avg_df["K"] == 20))
            )
            sub = avg_df[mask]

            # in case there are multiple (K, L) combos, plot them all
            for _, row in sub.iterrows():
                t = row["t"]
                mean_heightheight = row["mean_heightheight"]
                L = row["L"]
                K = row["K"]
                M = row["M"]

                t_cross = tx[typenoise][delta_to_label[deltaname]] * L**(z_dict[typenoise][delta_to_label[deltaname]])
                
                for ti in range(mean_heightheight.shape[0])[::4]:

                    if t[ti] > 0 and t[ti] < t_cross:

                        rlen = mean_heightheight.shape[1] # we plot up to half of this since the system is periodic
                        rrange = np.array(range(rlen))

                        # Collapse variables
                        rt1z = rrange[:int(rlen/2)] / ti**(1/z_dict[typenoise][delta_to_label[deltaname]])
                        Gt2az = mean_heightheight[ti,:int(rlen/2)] / ti**(2*alpha_loc_dict[typenoise][delta_to_label[deltaname]]/z_dict[typenoise][delta_to_label[deltaname]])

                        ax[j,i].scatter(rt1z, Gt2az, label=f"$t=${np.round(t[ti],2)}", marker=markerdict[L])
                    
                ax[j,i].set_title(typelabel+ r" ($\delta$="+f"{deltaname}, K={K}, {M} seeds). " + r"$t_\star$=" + f"{np.round(t_cross,1)}")

                ax[j,i].set_yscale('log')
                ax[j,i].set_xscale('log')

    # scaling as r^(2\alpha) = r^1 for alpha = 1/2 (EW/KPZ)
    ax[0,0].loglog(rt1z[1:int(rlen/16)], rt1z[1:int(rlen/16)]**(2*alpha_loc_dict["TimeDep"]["EW"]) /140, label=r"$\alpha_{EW}=1/2$", linestyle="--", color="k")
    ax[1,0].loglog(rt1z[1:int(rlen/16)], rt1z[1:int(rlen/16)]**(2*alpha_loc_dict["TimeDep"]["KPZ"])/15, label=r"$\alpha_{KPZ}=1/2$", linestyle="--", color="k")

    # scaling as r^(2\alpha) = r^2 for alpha_loc approx 1 (EW/KPZ)
    ax[0,1].loglog(rt1z[1:int(rlen/16)], rt1z[1:int(rlen/16)]**(2*alpha_loc_dict["Columnar"]["EW"])/15, label=r"$\alpha_{loc}^{EW}=1$", linestyle="--", color="k")
    ax[1,1].loglog(rt1z[1:int(rlen/16)], rt1z[1:int(rlen/16)]**(2*alpha_loc_dict["Columnar"]["KPZ"])/10, label=r"$\alpha_{loc}^{KPZ}=0.96$", linestyle="--", color="k")

    ax[0,0].legend()
    ax[0,1].legend()
    ax[1,0].legend()
    ax[1,1].legend()

    plt.tight_layout()
    plt.show()




#############################################
##### STRUCTURE FACTOR AND ITS COLLAPSE #####
#############################################

def plot_structurefactor_evolution(avg_df, L, T, tx):

    fig, ax = plt.subplots(2, 2, figsize=(10,10))
    fig.suptitle("Structure factor under different noise types, for $L=$" + str(L))

    fig.supxlabel(r"$\log k$")
    fig.supylabel(r"$\log S(k,t)$")

    for i, (typenoise, typelabel) in enumerate(zip(["TimeDep", "Columnar"], ["Time-dependent", "Columnar"])
    ):

        for j, (delta, deltaname) in enumerate(zip([0, np.arctan(5)], ["0", "atan5"])):

            # select rows for this noise type and delta
            mask = mask = (
                (avg_df["typenoise"] == typenoise)
                & (avg_df["L"] == L)
                & (avg_df["T"] == T)
                & (avg_df["deltaname"] == deltaname)
                & ((avg_df["K"] == 1) | (avg_df["K"] == 20))
            )
            sub = avg_df[mask]

            # in case there are multiple (K, L) combos, plot them all
            for _, row in sub.iterrows():
                t = row["t"]            
                mean_structurefactor = row["mean_structurefactor"]
                L = row["L"]
                K = row["K"]
                M = row["M"]

                t_cross = tx[typenoise][delta_to_label[deltaname]] * L**(z_dict[typenoise][delta_to_label[deltaname]])

                for ti in range(mean_structurefactor.shape[0])[::4]:

                    if t[ti] > 0 and t[ti] < t_cross:

                        klen = mean_structurefactor.shape[1] 
                        krange = np.array(range(klen))
                        # we only plot up to half due to nyquist
                        ax[j,i].scatter(krange[:int(klen/2)], mean_structurefactor[ti,:int(klen/2)], label=f"$t=${np.round(t[ti],2)}", marker=markerdict[L])
                    
                ax[j,i].set_title(typelabel+ r" ($\delta$="+f"{deltaname}, K={K}, {M} seeds). " + r"$t_\star$=" + f"{np.round(t_cross,1)}")

                ax[j,i].set_yscale('log')
                ax[j,i].set_xscale('log')

    # scaling as k^(-2*alpha-1) = k^2 for alpha = 1/2 (EW/KPZ)
    ax[0,0].loglog(krange[3:int(klen/4)], krange[3:int(klen/4)]**(-2*alpha_s_dict["TimeDep"]["EW"]-1) * 100, label=r"$\alpha_s^{EW}=1/2$", linestyle="--", color="k")
    ax[1,0].loglog(krange[3:int(klen/4)], krange[3:int(klen/4)]**(-2*alpha_s_dict["TimeDep"]["KPZ"]-1) * 1000, label=r"$\alpha_s^{KPZ}=1/2$", linestyle="--", color="k")

    # # scaling as k^(-2*alpha_s-1) for alpha_s approx 3/2 (EW) or 1.4 (KPZ)
    ax[0,1].loglog(krange[3:int(klen/4)], krange[3:int(klen/4)]**(-2*alpha_s_dict["Columnar"]["EW"]-1) * 200000, label=r"$\alpha_s^{EW}=3/2$", linestyle="--", color="k")
    ax[1,1].loglog(krange[5:int(klen/4)], krange[5:int(klen/4)]**(-2*alpha_s_dict["Columnar"]["KPZ"]-1) * 1000000, label=r"$\alpha_s^{KPZ}=1.4$", linestyle="--", color="k")

    ax[1,0].set_ylim(0,1e3)
    ax[1,1].set_ylim(0,1e5)

    ax[0,0].legend()
    ax[0,1].legend()
    ax[1,0].legend()
    ax[1,1].legend()

    plt.tight_layout()
    plt.show()


def plot_structurefactor_collapse(avg_df, L, T, tx):

    fig, ax = plt.subplots(2, 2, figsize=(10,10))
    fig.suptitle("Structure factor collapse under different noise types, for $L=$" + str(L))

    fig.supxlabel(r"$\log (k t^{1/z})$")
    fig.supylabel(r"$\log (S(k,t) k^{2\alpha+1})$")

    for i, (typenoise, typelabel) in enumerate(zip(["TimeDep", "Columnar"], ["Time-dependent", "Columnar"])
    ):

        for j, (delta, deltaname) in enumerate(zip([0, np.arctan(5)], ["0", "atan5"])):

            # select rows for this noise type and delta
            mask = mask = (
                (avg_df["typenoise"] == typenoise)
                & (avg_df["L"] == L)
                & (avg_df["T"] == T)
                & (avg_df["deltaname"] == deltaname)
                & ((avg_df["K"] == 1) | (avg_df["K"] == 20))
            )
            sub = avg_df[mask]

            # in case there are multiple (K, L) combos, plot them all
            for _, row in sub.iterrows():
                t = row["t"]            
                mean_structurefactor = row["mean_structurefactor"]
                L = row["L"]
                K = row["K"]
                M = row["M"]

                t_cross = tx[typenoise][delta_to_label[deltaname]] * L**(z_dict[typenoise][delta_to_label[deltaname]])

                for ti in range(mean_structurefactor.shape[0])[::4]:

                    if t[ti] > 0 and t[ti] < t_cross:

                        klen = mean_structurefactor.shape[1] # we only plot up to half due to nyquist
                        krange = np.array(range(klen))

                        # Collapse variables
                        kt1z = krange[:int(klen/2)] * ti**(1/z_dict[typenoise][delta_to_label[deltaname]])
                        Sk2a = mean_structurefactor[ti,:int(klen/2)] * krange[:int(klen/2)]**(2*alpha_dict[typenoise][delta_to_label[deltaname]]+1)

                        ax[j,i].scatter(kt1z, Sk2a, label=f"$t=${np.round(t[ti],2)}", marker=markerdict[L])

                ax[j,i].set_title(typelabel+ r" ($\delta$="+f"{deltaname}, K={K}, {M} seeds). " + r"$t_\star$=" + f"{np.round(t_cross,1)}")

                ax[j,i].set_yscale('log')
                ax[j,i].set_xscale('log')

    # scaling as y^(2*alpha+1) = k^2 for alpha = 1/2 (EW/KPZ)
    ax[0,0].loglog(kt1z[2:int(klen/16)], kt1z[2:int(klen/16)]**(2*alpha_dict["TimeDep"]["EW"]-2*alpha_s_dict["TimeDep"]["EW"]) * 50, label=r"$\alpha_s^{EW}=1/2$", linestyle="--", color="k")
    ax[1,0].loglog(kt1z[3:int(klen/8)], kt1z[3:int(klen/8)]**(2*alpha_dict["TimeDep"]["KPZ"]-2*alpha_s_dict["TimeDep"]["KPZ"]) * 400, label=r"$\alpha_s^{KPZ}=1/2$", linestyle="--", color="k")

    # # scaling as k^(-2*alpha_s-1) for alpha_s approx 3/2 (EW) or 1.4 (KPZ)
    ax[0,1].loglog(kt1z[2:int(klen/16)], kt1z[2:int(klen/16)]**(2*alpha_dict["Columnar"]["EW"]-2*alpha_s_dict["Columnar"]["EW"]) * 50000, label=r"$\alpha_s^{EW}=3/2$", linestyle="--", color="k")
    ax[1,1].loglog(kt1z[5:int(klen/16)], kt1z[5:int(klen/16)]**(2*alpha_dict["Columnar"]["KPZ"]-2*alpha_s_dict["Columnar"]["KPZ"]) * 1000000, label=r"$\alpha_s^{KPZ}=1.4$", linestyle="--", color="k")

    # ax[1,0].set_ylim(0,1e3)

    ax[0,0].legend()
    ax[0,1].legend()
    ax[1,0].legend()
    ax[1,1].legend()

    plt.tight_layout()
    plt.show()


#############################################
##### PHASE COVARIANCE AND ITS COLLAPSE #####
#############################################