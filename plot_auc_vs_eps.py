import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import erf
from sklearn.metrics import r2_score, mean_squared_error


def erf_func(x, a, b, c):
    return a * erf(b * x) + c

def goodness_of_fit(x, y, func, params):
    y_pred = func(x, *params)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    return r2, rmse

def gaussian_auc(epsilon, auc_clean, c):
    return 0.5 + (auc_clean - 0.5) * 0.5 * (1 + erf(c * epsilon / np.sqrt(2)))


def plot_auc_vs_epsilon(csv_file):
    # df = pd.read_csv(csv_file, header=None, names=["run", "noise", "epsilon", "delta", "roc_auc", "acc", "seed",
    #                                                "sep", "sensitivity", "top_1_accuracy", "top_k_accuracy"])
    df = pd.read_csv(csv_file, header=None, names=["run", "noise", "rho", "delta", "roc_auc", "acc", "seed",
                                                   "sep", "sensitivity", "top_1_accuracy", "top_k_accuracy", "epsilon", "tw"])
    max_epsilon = df["epsilon"][df["epsilon"] != np.inf].astype(float).max() + 200
    # max_plotting_epsilon = 40
    # df = df[df["epsilon"].astype(float) <= max_plotting_epsilon]
    df["epsilon"] = df["epsilon"].replace(np.inf, max_epsilon).astype(float)
    df["epsilon_label"] = df["epsilon"].replace(max(df["epsilon"]), r"$\infty$")

    grouped_roc_auc = df.groupby("epsilon")["roc_auc"].agg(["mean", "std"]).reset_index()
    grouped_top_k = df.groupby("epsilon")["top_k_accuracy"].agg(["mean", "std"]).reset_index()
    grouped_top_1 = df.groupby("epsilon")["top_1_accuracy"].agg(["mean", "std"]).reset_index()

    # Grouped stuff for tables.
    grouped = df.groupby(['rho', 'epsilon']).agg(
        roc_auc_mean=('roc_auc', 'mean'),
        roc_auc_std=('roc_auc', 'std'),
        atk1_mean=('top_1_accuracy', 'mean'),
        atk1_std=('top_1_accuracy', 'std'),
        atk10_mean=('top_k_accuracy', 'mean'),
        atk10_std=('top_k_accuracy', 'std')
    ).reset_index()
    grouped['roc_auc±std'] = grouped['roc_auc_mean'].round(3).astype(str) + '±' + grouped['roc_auc_std'].round(3).astype(str)
    grouped['atk1±std'] = grouped['atk1_mean'].round(3).astype(str) + '±' + grouped['atk1_std'].round(3).astype(str)
    grouped['atk10±std'] = grouped['atk10_mean'].round(3).astype(str) + '±' + grouped['atk10_std'].round(3).astype(str)
    # grouped['atk1±std'] = grouped['atk1_mean'].apply(lambda x: f"{x:.2e}") + '±' + grouped['atk1_std'].apply(lambda x: f"{x:.2e}")
    # grouped['atk10±std'] = grouped['atk10_mean'].apply(lambda x: f"{x:.2e}") + '±' + grouped['atk10_std'].apply(lambda x: f"{x:.2e}")
    table = grouped[['rho', 'epsilon', 'roc_auc±std', 'atk1±std', 'atk10±std']]

    fig, ax1 = plt.subplots(figsize=(6, 4))
    BIGGER_SIZE = 12

    # Left y-axis.
    ax1.errorbar(grouped_roc_auc["epsilon"], grouped_roc_auc["mean"], yerr=grouped_roc_auc["std"], fmt='o-', capsize=5, label="AUC", color="blue")
    ax1.set_xlabel(r"$\epsilon$")

    # Hacky thing to get the last label to look pretty. Like, very hacky.
    ax1.set_xscale('log')
    xticks = list(ax1.get_xticks()) + [grouped_roc_auc["epsilon"].iloc[-1]]
    xticklabels = [tick.get_text() for tick in ax1.get_xticklabels()] + [df["epsilon_label"].unique()[-1]]
    finite_ticks = [tick for tick in ax1.get_xticks() if tick < max_epsilon] + [grouped_roc_auc["epsilon"].iloc[-1]]
    finite_labels = [tick.get_text() for tick in ax1.get_xticklabels()]
    finite_labels = finite_labels[:len(finite_ticks)-1] + [r"$\infty$"]
    ax1.set_xticks(finite_ticks)
    ax1.set_xticklabels(finite_labels)
    ax1.set_xlim(left=3.5*1e-4)

    ax1.set_ylabel("AUC", color="blue", fontsize=BIGGER_SIZE)
    ax1.tick_params(axis='y', labelcolor="blue", labelsize=BIGGER_SIZE)
    ax1.tick_params(axis='x', labelsize=BIGGER_SIZE)
    ax1.set_xlabel(r"$\epsilon$", fontsize=BIGGER_SIZE)
    # ax1.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Second y-axis.
    ax2 = ax1.twinx()
    ax2.set_yscale('log')
    ax2.errorbar(grouped_top_k["epsilon"], grouped_top_k["mean"], yerr=grouped_top_k["std"], fmt='s', capsize=3, label="Top-10 Atk Acc", color="red")
    ax2.errorbar(grouped_top_1["epsilon"], grouped_top_1["mean"], yerr=grouped_top_1["std"], fmt='^', capsize=3, label="Top-1 Atk Acc", color="red")
    ax2.set_ylabel("Top-10 and Top-1 Attacks Accuracy", color="red", fontsize=BIGGER_SIZE)
    ax2.set_ylim(top=3)
    ax2.tick_params(axis='y', labelcolor="red", labelsize=BIGGER_SIZE)

    # Pretty stuff.
    plt.title(r"MOLHIV - AUC and attack accuracy vs $\epsilon$", fontsize=BIGGER_SIZE)
    fig.tight_layout()



    # Some curve fits, dropping infs and regrouping.
    df = df[df["epsilon"] != np.inf]
    grouped_roc_auc = df.groupby("epsilon")["roc_auc"].agg(["mean", "std"]).reset_index()
    params, _ = curve_fit(erf_func, grouped_roc_auc["epsilon"], grouped_roc_auc["mean"], maxfev=10000)
    # x_fit = np.linspace(min(grouped_roc_auc["epsilon"]), max(grouped_roc_auc["epsilon"]), 100000)
    x_fit = np.geomspace(min(grouped_roc_auc["epsilon"]), max(grouped_roc_auc["epsilon"]), 1000)
    y_fit = erf_func(x_fit, *params)
    # plt.plot(x_fit, y_fit, label="Fit")
    ax1.plot(x_fit, y_fit, label="AUC fit with erf", color="green")

    a, b, c = params
    print(f"Offset (c): {c:.3f}, Amplitude (a): {a:.3f}, Slope (b): {b:.3f}")

    r2, rmse = goodness_of_fit(grouped_roc_auc['epsilon'], grouped_roc_auc["mean"], erf_func, params)
    print(f"R^2: {r2:.4f}, RMSE: {rmse:.4f}")

    # Add text box for fanciness.
    # textstr = f"$R^2$: {r2:.2f}\nRMSE: {rmse:.2f}"
    # plt.gca().text(0.98, 0.02, textstr, transform=plt.gca().transAxes, fontsize=10,
    #            verticalalignment='bottom', horizontalalignment='right',
    #            bbox=dict(boxstyle="round", facecolor="white", edgecolor="black", alpha=1.0))
    #
    # c = 0.02/(4 * 0.3124 * np.sqrt(np.log(1.25 / 1e-5)))
    # y_theory = gaussian_auc(x_fit, 0.74, 0.157)
    # print(y_theory)
    # plt.plot(x_fit, y_theory, label="Theoretical")

    # theory_slope = 3.5/(4 * 0.3124 * np.sqrt(np.log(1.25 / 1e-5)))
    # print(f"Theory slope: {theory_slope:.3f}")

    # plt.grid(True, which="both", linestyle="--", linewidth=0.5, zorder=1)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    legend = ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left",
                        frameon=True, facecolor="white", edgecolor="black")

    plt.savefig(csv_file.replace(".csv", ".pdf"))

    plt.show()


def plot_tw(csv_file):
    # df = pd.read_csv(csv_file, header=None, names=["run", "noise", "epsilon", "delta", "roc_auc", "acc", "seed",
    #                                                "sep", "sensitivity", "top_1_accuracy", "top_k_accuracy"])
    df = pd.read_csv(csv_file, header=None, names=["run", "noise", "rho", "delta", "roc_auc", "acc", "seed",
                                                   "sep", "sensitivity", "top_1_accuracy", "top_k_accuracy", "epsilon",
                                                   "tw"])


    grouped_roc_auc = df.groupby("tw")["roc_auc"].agg(["mean", "std"]).reset_index()
    grouped_top_k = df.groupby("tw")["top_k_accuracy"].agg(["mean", "std"]).reset_index()
    grouped_top_1 = df.groupby("tw")["top_1_accuracy"].agg(["mean", "std"]).reset_index()

    fig, ax1 = plt.subplots(figsize=(6, 4))
    BIGGER_SIZE = 12

    # Left y-axis.
    ax1.errorbar(grouped_roc_auc["tw"], grouped_roc_auc["mean"], yerr=grouped_roc_auc["std"], fmt='o-', capsize=5, label="AUC", color="blue")
    ax1.scatter(df["tw"], df['roc_auc'], color="blue", s=10, zorder=2, alpha=0.5)
    ax1.set_xlabel("Maximum treewidth")

    # Hacky thing to get the last label to look pretty. Like, very hacky.
    # ax1.set_xscale('log')
    # xticks = list(ax1.get_xticks()) + [grouped_roc_auc["epsilon"].iloc[-1]]
    # xticklabels = [tick.get_text() for tick in ax1.get_xticklabels()] + [df["epsilon_label"].unique()[-1]]
    # finite_ticks = [tick for tick in ax1.get_xticks() if tick < max_epsilon] + [grouped_roc_auc["epsilon"].iloc[-1]]
    # finite_labels = [tick.get_text() for tick in ax1.get_xticklabels()]
    # finite_labels = finite_labels[:len(finite_ticks)-1] + [r"$\infty$"]
    # ax1.set_xticks(finite_ticks)
    # ax1.set_xticklabels(finite_labels)
    # ax1.set_xlim(left=3.5*1e-4)

    ax1.set_ylabel("AUC", color="blue", fontsize=BIGGER_SIZE)
    ax1.tick_params(axis='y', labelcolor="blue", labelsize=BIGGER_SIZE)
    ax1.tick_params(axis='x', labelsize=BIGGER_SIZE)
    ax1.set_xlabel("Maximum treewidth", fontsize=BIGGER_SIZE)
    ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    # ax1.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Second y-axis.
    ax2 = ax1.twinx()
    # ax2.set_yscale('log')
    # ax2.errorbar(grouped_top_k["tw"], grouped_top_k["mean"], yerr=grouped_top_k["std"], fmt='s', capsize=3, label="Top-10 Atk Acc", color="red")
    ax2.errorbar(grouped_top_1["tw"], grouped_top_1["mean"], yerr=grouped_top_1["std"], fmt='^', capsize=3, label="Top-1 Atk Acc", color="red")
    ax2.set_ylabel("Top-1 Attacks Accuracy", color="red", fontsize=BIGGER_SIZE)
    # ax2.set_ylim(top=3)
    ax2.tick_params(axis='y', labelcolor="red", labelsize=BIGGER_SIZE)

    # Pretty stuff.
    plt.title(r"MOLHIV - AUC and attack accuracy vs maximum treewidth", fontsize=BIGGER_SIZE)
    fig.tight_layout()



    # Some curve fits, dropping infs and regrouping.
    # df = df[df["epsilon"] != np.inf]
    # grouped_roc_auc = df.groupby("epsilon")["roc_auc"].agg(["mean", "std"]).reset_index()
    # params, _ = curve_fit(erf_func, grouped_roc_auc["epsilon"], grouped_roc_auc["mean"], maxfev=10000)
    # x_fit = np.linspace(min(grouped_roc_auc["epsilon"]), max(grouped_roc_auc["epsilon"]), 100000)
    # x_fit = np.geomspace(min(grouped_roc_auc["epsilon"]), max(grouped_roc_auc["epsilon"]), 1000)
    # y_fit = erf_func(x_fit, *params)
    # plt.plot(x_fit, y_fit, label="Fit")
    # ax1.plot(x_fit, y_fit, label="AUC fit with erf", color="green")

    # a, b, c = params
    # print(f"Offset (c): {c:.3f}, Amplitude (a): {a:.3f}, Slope (b): {b:.3f}")

    # r2, rmse = goodness_of_fit(grouped_roc_auc['epsilon'], grouped_roc_auc["mean"], erf_func, params)
    # print(f"R^2: {r2:.4f}, RMSE: {rmse:.4f}")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    legend = ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper center",
                        frameon=True, facecolor="white", edgecolor="black")

    plt.savefig(csv_file.replace(".csv", ".pdf"))

    plt.show()


# plot_tw("results/classifier_on_homcounts/molhiv_tw_output_20250613_152622_knn_MOLHIV_16_50_all.csv")
# plot_auc_vs_epsilon("results/classifier_on_homcounts/molhiv_output_20250602_101112_knn_MOLHIV_16_50_all_tw1.csv")