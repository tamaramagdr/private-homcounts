import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy.optimize import curve_fit
from scipy.special import erf
from sklearn.metrics import r2_score, mean_squared_error
from matplotlib.transforms import Bbox

def nudge_legend(legend, dx=0, dy=0, units="axes"):
    """
    Nudge a legend by (dx, dy).
    """
    bbox = legend.get_bbox_to_anchor()
    transform = legend.get_bbox_to_anchor()._transform  # Keep the same transform.

    if units == "pixels":
        fig = legend.axes.figure
        dx /= fig.dpi * fig.get_size_inches()[0]
        dy /= fig.dpi * fig.get_size_inches()[1]
        units = "axes"

    # Nudge in axes coordinates.
    new_bbox = Bbox.from_bounds(
        bbox.x0 + dx, bbox.y0 + dy, bbox.width, bbox.height
    )
    legend.set_bbox_to_anchor(new_bbox, transform=transform)

def erf_func(x, a, b, c):
    return a * erf(b * x) + c

def goodness_of_fit(x, y, func, params):
    y_pred = func(x, *params)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    return r2, rmse

def gaussian_auc(epsilon, auc_clean, c):
    return 0.5 + (auc_clean - 0.5) * 0.5 * (1 + erf(c * epsilon / np.sqrt(2)))


def plot_sbm_results(csv_file, csv_file_2, features_csv_file=None, attack_csv_file=None, eps_correction=0,
                        legend_position='upper left', legend_bbox=None, auc_fit_line=True,
                        plot_accuracy=False,
                        title=None, no_second_axis=True):
    df = pd.read_csv(csv_file, header=None, names=["run", "noise", "rho", "delta", "roc_auc", "acc", "seed",
                                                   "sep", "sensitivity", "top_1_accuracy", "top_k_accuracy", "epsilon", "tw"])
    max_epsilon = df["epsilon"][df["epsilon"] != np.inf].astype(float).max() + 200
    df["epsilon"] = df["epsilon"].replace(np.inf, max_epsilon).astype(float)
    df["epsilon"] = df["epsilon"] + eps_correction
    df["epsilon_label"] = df["epsilon"].replace(max(df["epsilon"]), r"$\infty$")

    grouped_roc_auc = df.groupby("epsilon")["roc_auc"].agg(["mean", "std"]).reset_index()
    grouped_acc = df.groupby("epsilon")["acc"].agg(["mean", "std"]).reset_index()
    grouped_top_k = df.groupby("epsilon")["top_k_accuracy"].agg(["mean", "std"]).reset_index()
    grouped_top_1 = df.groupby("epsilon")["top_1_accuracy"].agg(["mean", "std"]).reset_index()

    # Grouped stuff for tables.
    grouped = df.groupby(['rho', 'epsilon']).agg(
        roc_auc_mean=('roc_auc', 'mean'),
        roc_auc_std=('roc_auc', 'std'),
        acc_mean=('acc', 'mean'),
        acc_std=('acc', 'std'),
        atk1_mean=('top_1_accuracy', 'mean'),
        atk1_std=('top_1_accuracy', 'std'),
        atk10_mean=('top_k_accuracy', 'mean'),
        atk10_std=('top_k_accuracy', 'std')
    ).reset_index()
    grouped['roc_auc±std'] = grouped['roc_auc_mean'].round(3).astype(str) + '±' + grouped['roc_auc_std'].round(3).astype(str)
    grouped['acc±std'] = grouped['acc_mean'].round(3).astype(str) + '±' + grouped['acc_std'].round(3).astype(str)
    grouped['atk1±std'] = grouped['atk1_mean'].round(3).astype(str) + '±' + grouped['atk1_std'].round(3).astype(str)
    grouped['atk10±std'] = grouped['atk10_mean'].round(3).astype(str) + '±' + grouped['atk10_std'].round(3).astype(str)

    df2 = pd.read_csv(csv_file_2, header=None, names=["run", "noise", "rho", "delta", "roc_auc", "acc", "seed",
                                                   "sep", "sensitivity", "top_1_accuracy", "top_k_accuracy", "epsilon", "tw"])
    max_epsilon = df2["epsilon"][df2["epsilon"] != np.inf].astype(float).max() + 200
    df2["epsilon"] = df2["epsilon"].replace(np.inf, max_epsilon).astype(float)
    df2["epsilon"] = df2["epsilon"] + eps_correction
    df2["epsilon_label"] = df2["epsilon"].replace(max(df2["epsilon"]), r"$\infty$")

    grouped2_roc_auc = df2.groupby("epsilon")["roc_auc"].agg(["mean", "std"]).reset_index()
    grouped2_acc = df2.groupby("epsilon")["acc"].agg(["mean", "std"]).reset_index()
    grouped2_top_k = df2.groupby("epsilon")["top_k_accuracy"].agg(["mean", "std"]).reset_index()
    grouped2_top_1 = df2.groupby("epsilon")["top_1_accuracy"].agg(["mean", "std"]).reset_index()

    # grouped2 stuff for tables.
    grouped2 = df2.groupby(['rho', 'epsilon']).agg(
        roc_auc_mean=('roc_auc', 'mean'),
        roc_auc_std=('roc_auc', 'std'),
        acc_mean=('acc', 'mean'),
        acc_std=('acc', 'std'),
        atk1_mean=('top_1_accuracy', 'mean'),
        atk1_std=('top_1_accuracy', 'std'),
        atk10_mean=('top_k_accuracy', 'mean'),
        atk10_std=('top_k_accuracy', 'std')
    ).reset_index()
    grouped2['roc_auc±std'] = grouped2['roc_auc_mean'].round(3).astype(str) + '±' + grouped2['roc_auc_std'].round(3).astype(str)
    grouped2['acc±std'] = grouped2['acc_mean'].round(3).astype(str) + '±' + grouped2['acc_std'].round(3).astype(str)
    grouped2['atk1±std'] = grouped2['atk1_mean'].round(3).astype(str) + '±' + grouped2['atk1_std'].round(3).astype(str)
    grouped2['atk10±std'] = grouped2['atk10_mean'].round(3).astype(str) + '±' + grouped2['atk10_std'].round(3).astype(str)


    fig, ax1 = plt.subplots(figsize=(6, 4))
    BIGGER_SIZE = 14

    blue_color = "#984EA3"
    brown_color = "#A65628"
    # brown_color = "#F781BF"
    blue_color = "#F781BF"
    red_color = "#FF7F00"

    # Left y-axis.
    ax1.errorbar(grouped_acc["epsilon"], grouped_acc["mean"], yerr=grouped_acc["std"], fmt='o-', capsize=5, label="Cycles", color=blue_color)
    ax1.set_xlabel(r"$\epsilon$")

    # Hacky thing to get the last label to look pretty. Like, very hacky.
    ax1.set_xscale('log')
    finite_ticks = [tick for tick in ax1.get_xticks() if tick < max_epsilon] + [grouped_roc_auc["epsilon"].iloc[-1]]
    finite_labels = [tick.get_text() for tick in ax1.get_xticklabels()]
    finite_labels = finite_labels[:len(finite_ticks)-1] + [r"$\infty$"]
    ax1.set_xticks(finite_ticks)
    ax1.set_xticklabels(finite_labels)
    ax1.set_xlim(left=3.5*1e-4)
    if eps_correction>0:
        ax1.set_xlim(left=eps_correction)

    ax1.set_ylabel("Accuracy", fontsize=BIGGER_SIZE)
    ax1.tick_params(axis='y', labelsize=BIGGER_SIZE)
    ax1.tick_params(axis='x', labelsize=BIGGER_SIZE)
    ax1.set_xlabel(r"$\epsilon$", fontsize=BIGGER_SIZE)

    # Second y-axis.
    if not no_second_axis:
        ax2 = ax1.twinx()
        ax2.set_yscale('log')
        ax2.errorbar(grouped_top_k["epsilon"], grouped_top_k["mean"], yerr=grouped_top_k["std"], fmt='s', capsize=3, label="Top-10 Atk Acc", color=red_color)
        ax2.errorbar(grouped_top_1["epsilon"], grouped_top_1["mean"], yerr=grouped_top_1["std"], fmt='^', capsize=3, label="Top-1 Atk Acc", color=red_color)
        ax2.set_ylabel("Top-10 and Top-1 Attacks Accuracy", color=red_color, fontsize=BIGGER_SIZE)
        ax2.set_ylim(top=3)
        ax2.tick_params(axis='y', labelcolor="red", labelsize=BIGGER_SIZE)
    else:
        ax1.errorbar(grouped2_acc["epsilon"], grouped2_acc["mean"], yerr=grouped2_acc["std"], fmt='s-', capsize=5, label="Trees", color=brown_color)

    # Pretty stuff.
    plt.title(r"SBM - accuracy vs $\epsilon$", fontsize=BIGGER_SIZE)
    if title is not None:
        plt.title(title, fontsize=BIGGER_SIZE)
    fig.tight_layout()


    if not no_second_axis:
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        # Move the legend to the right by position_correction
        legend = ax1.legend(handles1 + handles2, labels1 + labels2,
                            loc=legend_position,
                            bbox_to_anchor=legend_bbox,
                            frameon=True, facecolor="white", edgecolor="black")
    else:
        handles1, labels1 = ax1.get_legend_handles_labels()
        # Move the legend to the right by position_correction
        legend = ax1.legend(handles1, labels1,
                            loc=legend_position,
                            bbox_to_anchor=legend_bbox,
                            frameon=True, facecolor="white", edgecolor="black")


    ax1.grid(color="lightgrey", linestyle="dashed")
    ax1.set_facecolor('xkcd:white')

    plt.savefig(csv_file.replace(".csv", ".pdf"))

    plt.show()


def plot_auc_vs_epsilon(csv_file, features_csv_file=None, attack_csv_file=None, eps_correction=0,
                        legend_position='upper left', legend_bbox=None, auc_fit_line=True,
                        plot_accuracy=False,
                        title=None, no_second_axis=False, mae=False):
    df = pd.read_csv(csv_file, header=None, names=["run", "noise", "rho", "delta", "roc_auc", "acc", "seed",
                                                   "sep", "sensitivity", "top_1_accuracy", "top_k_accuracy", "epsilon", "tw"])
    max_epsilon = df["epsilon"][df["epsilon"] != np.inf].astype(float).max() + 200
    min_epsilon = df["epsilon"][df["epsilon"] != np.inf].astype(float).min()
    df["epsilon"] = df["epsilon"].replace(np.inf, max_epsilon).astype(float)
    df["epsilon"] = df["epsilon"] + eps_correction
    df["epsilon_label"] = df["epsilon"].replace(max(df["epsilon"]), r"$\infty$")

    if attack_csv_file is not None:
        """
        If attack results are provided, replace the attack accuracies with those from the attack results.
        """
        attack_df = pd.read_csv(attack_csv_file, header=None,
                                names=["run", "noise", "rho", "delta", "roc_auc", "acc", "seed",
                                       "sep", "sensitivity", "top_1_accuracy", "top_k_accuracy", "epsilon", "tw"])
        attack_df["epsilon"] = attack_df["epsilon"].replace(np.inf, max_epsilon).astype(float)
        attack_df["epsilon"] = attack_df["epsilon"] + eps_correction
        attack_df = attack_df[["epsilon", "top_1_accuracy", "top_k_accuracy"]]
        df = df.drop(columns=["top_1_accuracy", "top_k_accuracy"]).merge(attack_df, on="epsilon", how="left")

    grouped_roc_auc = df.groupby("epsilon")["roc_auc"].agg(["mean", "std"]).reset_index()
    grouped_acc = df.groupby("epsilon")["acc"].agg(["mean", "std"]).reset_index()
    grouped_top_k = df.groupby("epsilon")["top_k_accuracy"].agg(["mean", "std"]).reset_index()
    grouped_top_1 = df.groupby("epsilon")["top_1_accuracy"].agg(["mean", "std"]).reset_index()

    # Grouped stuff for tables.
    grouped = df.groupby(['rho', 'epsilon']).agg(
        roc_auc_mean=('roc_auc', 'mean'),
        roc_auc_std=('roc_auc', 'std'),
        acc_mean=('acc', 'mean'),
        acc_std=('acc', 'std'),
        atk1_mean=('top_1_accuracy', 'mean'),
        atk1_std=('top_1_accuracy', 'std'),
        atk10_mean=('top_k_accuracy', 'mean'),
        atk10_std=('top_k_accuracy', 'std')
    ).reset_index()
    grouped['roc_auc±std'] = grouped['roc_auc_mean'].round(3).astype(str) + '±' + grouped['roc_auc_std'].round(3).astype(str)
    grouped['acc±std'] = grouped['acc_mean'].round(3).astype(str) + '±' + grouped['acc_std'].round(3).astype(str)
    grouped['atk1±std'] = grouped['atk1_mean'].round(3).astype(str) + '±' + grouped['atk1_std'].round(3).astype(str)
    grouped['atk10±std'] = grouped['atk10_mean'].round(3).astype(str) + '±' + grouped['atk10_std'].round(3).astype(str)
    table = grouped[['rho', 'epsilon', 'roc_auc±std', 'atk1±std', 'atk10±std']]

    fig, ax1 = plt.subplots(figsize=(6, 4))
    BIGGER_SIZE = 14

    blue_color = "#984EA3"  # purple
    red_color = "#FF7F00"
    green_color = "#4DAF4A"

    # Left y-axis.
    if plot_accuracy:
        ax1.errorbar(grouped_acc["epsilon"], grouped_acc["mean"], yerr=grouped_acc["std"], fmt='o-', capsize=5, label="Accuracy", color=blue_color)
    elif mae:
        ax1.errorbar(grouped_acc["epsilon"], grouped_acc["mean"], yerr=grouped_acc["std"], fmt='o-', capsize=5, label="MAE", color=blue_color)
    else:
        ax1.errorbar(grouped_roc_auc["epsilon"], grouped_roc_auc["mean"], yerr=grouped_roc_auc["std"], fmt='o-', capsize=5, label="AUC", color=blue_color)

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
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    if eps_correction>0:
        ax1.set_xlim(left=eps_correction)

    if plot_accuracy:
        ax1.set_ylabel("Accuracy", color=blue_color, fontsize=BIGGER_SIZE)
    elif mae:
        ax1.set_ylabel("MAE", color=blue_color, fontsize=BIGGER_SIZE)
    else:
        ax1.set_ylabel("AUC", color=blue_color, fontsize=BIGGER_SIZE)
    ax1.tick_params(axis='y', labelcolor=blue_color, labelsize=BIGGER_SIZE)
    ax1.tick_params(axis='x', labelsize=BIGGER_SIZE)
    ax1.set_xlabel(r"$\epsilon$", fontsize=BIGGER_SIZE)

    # Second y-axis.
    if not no_second_axis:
        ax2 = ax1.twinx()
        ax2.set_yscale('log')
        ax2.errorbar(grouped_top_k["epsilon"], grouped_top_k["mean"], yerr=grouped_top_k["std"], fmt='s', capsize=3, label="Top-10 Atk Acc", color=red_color)
        ax2.errorbar(grouped_top_1["epsilon"], grouped_top_1["mean"], yerr=grouped_top_1["std"], fmt='^', capsize=3, label="Top-1 Atk Acc", color=red_color)
        ax2.set_ylabel("Top-10 and Top-1 Attacks Accuracy", color=red_color, fontsize=BIGGER_SIZE)
        ax2.set_ylim(top=3)
        ax2.tick_params(axis='y', labelcolor=red_color, labelsize=BIGGER_SIZE)

    # Pretty stuff.
    plt.title(r"MOLHIV - AUC and attack accuracy vs $\epsilon$", fontsize=BIGGER_SIZE)
    if title is not None:
        plt.title(title, fontsize=BIGGER_SIZE)
    fig.tight_layout()


    # Some curve fits, dropping infs and regrouping.
    if auc_fit_line:
        df = df[df["epsilon"] != np.inf]
        grouped_roc_auc = df.groupby("epsilon")["roc_auc"].agg(["mean", "std"]).reset_index()
        params, _ = curve_fit(erf_func, grouped_roc_auc["epsilon"], grouped_roc_auc["mean"], maxfev=10000)
        x_fit = np.geomspace(min(grouped_roc_auc["epsilon"]), max(grouped_roc_auc["epsilon"]), 1000)
        y_fit = erf_func(x_fit, *params)
        ax1.plot(x_fit, y_fit, label="AUC fit with erf", color=green_color)

        a, b, c = params
        print(f"Offset (c): {c:.3f}, Amplitude (a): {a:.3f}, Slope (b): {b:.3f}")

        r2, rmse = goodness_of_fit(grouped_roc_auc['epsilon'], grouped_roc_auc["mean"], erf_func, params)
        print(f"R^2: {r2:.4f}, RMSE: {rmse:.4f}")

    if features_csv_file is not None:
        df_features = pd.read_csv(features_csv_file,
                                  header=None,
                                  names=["run", "noise", "rho", "delta", "roc_auc", "acc", "seed",
                                         "sep", "sensitivity", "top_1_accuracy", "top_k_accuracy", "epsilon", "tw"])
        auc_clean = df_features['roc_auc'].mean()
        std_clean = df_features['roc_auc'].std()
        line_limits = (ax1.get_xlim()[0], ax1.get_xlim()[1])
        print(f"Feature-only AUC: {auc_clean:.3f} ± {std_clean:.3f}")
        if mae:
            ax1.hlines(auc_clean, xmin=line_limits[0], xmax=line_limits[1], colors='purple', linestyles='dashed', label='Feature-only MAE')
        else:
            ax1.hlines(auc_clean, xmin=line_limits[0], xmax=line_limits[1], colors='purple', linestyles='dashed', label='Feature-only AUC')
        ax1.fill_betweenx([auc_clean - std_clean, auc_clean + std_clean], x1=line_limits[0], x2=line_limits[1], color='purple', alpha=0.2)

    if not no_second_axis:
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        # Move the legend to the right by position_correction
        legend = ax1.legend(handles1 + handles2, labels1 + labels2,
                            loc=legend_position,
                            bbox_to_anchor=legend_bbox,
                            frameon=True, facecolor="white", edgecolor="black")
    else:
        handles1, labels1 = ax1.get_legend_handles_labels()
        legend = ax1.legend(handles1, labels1,
                            loc=legend_position,
                            bbox_to_anchor=legend_bbox,
                            frameon=True, facecolor="white", edgecolor="black")




    ax1.grid(color="lightgrey", linestyle="dashed")
    ax2.grid(False)
    ax1.set_facecolor('xkcd:white')
    ax2.set_facecolor('xkcd:white')
    plt.savefig(csv_file.replace(".csv", ".pdf"))

    plt.show()


def plot_tw(csv_file, save=True):
    df = pd.read_csv(csv_file, header=None, names=["run", "noise", "rho", "delta", "roc_auc", "acc", "seed",
                                                   "sep", "sensitivity", "top_1_accuracy", "top_k_accuracy", "epsilon",
                                                   "tw"])


    grouped_roc_auc = df.groupby("tw")["roc_auc"].agg(["mean", "std"]).reset_index()
    grouped_top_k = df.groupby("tw")["top_k_accuracy"].agg(["mean", "std"]).reset_index()
    grouped_top_1 = df.groupby("tw")["top_1_accuracy"].agg(["mean", "std"]).reset_index()

    fig, ax1 = plt.subplots(figsize=(6, 4))
    BIGGER_SIZE = 14

    blue_color = "#A65628" # Brown
    red_color = "#FF7F00"

    # Left y-axis.
    ax1.errorbar(grouped_roc_auc["tw"], grouped_roc_auc["mean"], yerr=grouped_roc_auc["std"], fmt='o-', capsize=5, label="AUC", color=blue_color)
    ax1.scatter(df["tw"], df['roc_auc'], color=blue_color, s=10, zorder=2, alpha=0.5)
    ax1.set_xlabel("Maximum treewidth")

    ax1.set_ylabel("AUC", color=blue_color, fontsize=BIGGER_SIZE)
    ax1.tick_params(axis='y', labelcolor=blue_color, labelsize=BIGGER_SIZE)
    ax1.tick_params(axis='x', labelsize=BIGGER_SIZE)
    ax1.set_xlabel("Maximum treewidth", fontsize=BIGGER_SIZE)
    ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Second y-axis.
    ax2 = ax1.twinx()
    ax2.errorbar(grouped_top_1["tw"]+0.05, grouped_top_1["mean"], yerr=grouped_top_1["std"], fmt='^', capsize=3, label="Top-1 Atk Acc", color=red_color)
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax2.set_ylabel("Top-1 Attacks Accuracy", color=red_color, fontsize=BIGGER_SIZE)
    ax2.tick_params(axis='y', labelcolor=red_color, labelsize=BIGGER_SIZE)

    # Pretty stuff.
    plt.title(r"MOLHIV - AUC and attack accuracy vs maximum treewidth", fontsize=BIGGER_SIZE)
    fig.tight_layout()

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    legend = ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right",
                        frameon=True, facecolor="white", edgecolor="black")


    ax1.grid(color="lightgrey", linestyle="dashed")
    ax2.grid(False)
    ax1.set_facecolor('xkcd:white')
    ax2.set_facecolor('xkcd:white')
    if save:
        plt.savefig(csv_file.replace(".csv", ".pdf"))

    plt.show()

results_folder = "results/classifier_on_homdensities/"

plt.style.use('seaborn-v0_8')
plot_auc_vs_epsilon(csv_file=f"{results_folder}r_MOLHIV_16_50_all_tw1.csv")
plot_auc_vs_epsilon(csv_file=f"{results_folder}r_MOLBACE_16_50_all_tw1.csv",
                    title=r"MOLBACE - AUC and attack accuracy vs $\epsilon$")
plot_sbm_results(csv_file_2=f"{results_folder}r_TREE_SBM_16_50_all_tw1.csv",
                 csv_file=f"{results_folder}r_SBM_16_50_all_cycles.csv")
plot_tw(csv_file=f"{results_folder}rtw_MOLHIV_16_50_tw_comparison_1.csv")
