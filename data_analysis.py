import numpy as np
import pandas as pd

def get_result_summary(csv_file_list, features_only = False, eps_correction=0, attack_csv_file=None, global_sensitivity=False):
    for csv_file in csv_file_list:
        df = pd.read_csv(csv_file, header=None, names=["run", "noise", "rho", "delta", "roc_auc", "acc", "seed",
                                                       "sep", "sensitivity", "top_1_accuracy", "top_k_accuracy", "epsilon", "tw"])
        max_epsilon = df["epsilon"][df["epsilon"] != np.inf].astype(float).max() + 200
        min_epsilon = df["epsilon"][df["epsilon"] != np.inf].astype(float).min()
        df["epsilon"] = df["epsilon"].replace(np.inf, max_epsilon).astype(float)
        df["epsilon"] = df["epsilon"] + eps_correction
        df["epsilon_label"] = df["epsilon"].replace(max(df["epsilon"]), r"$\infty$")

        if "only" in csv_file or features_only:
            # Print avg and stdev for roc_auc and acc.
            roc_auc_mean = df["roc_auc"].mean()
            roc_auc_std = df["roc_auc"].std()
            acc_mean = df["acc"].mean()
            acc_std = df["acc"].std()
            dataset_name = csv_file.split('/')[2].split('_')[1].upper()
            if "features" in csv_file and "only" not in csv_file:
                dataset_name += " + Features"
            print(f"Dataset: {dataset_name}")
            print(f"roc_auc: {roc_auc_mean:.3f} ± {roc_auc_std:.3f}     acc: {acc_mean:.3f} ± {acc_std:.3f}")
            continue

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
        if not global_sensitivity:
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
        else:
            grouped = df.groupby(['rho']).agg(
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
        grouped['atk1±std'] = grouped['atk1_mean'].round(5).astype(str) + '±' + grouped['atk1_std'].round(5).astype(str)
        grouped['atk10±std'] = grouped['atk10_mean'].round(5).astype(str) + '±' + grouped['atk10_std'].round(5).astype(str)
        # select only row with rho=0.01 and inf.
        if not global_sensitivity:
            grouped = grouped[(grouped['rho'] == 0.01) | (grouped['epsilon'] == max_epsilon)]
            # table = grouped[['rho', 'epsilon', 'roc_auc±std', 'acc±std', 'atk1±std', 'atk10±std']]
            table = grouped[['rho', 'epsilon', 'roc_auc±std', 'acc±std', 'atk1±std', 'atk10±std']]
        else:
            table = grouped[['rho', 'roc_auc±std', 'acc±std', 'atk1±std', 'atk10±std']]

        # Read name of the dataset as the first word after the second slash in the file path.
        try:
            dataset_name = csv_file.split('/')[-1].split('_')[1].upper()
        except:
            dataset_name = "Unknown Dataset"
        if "features" in csv_file and "only" not in csv_file:
            dataset_name += " + Features"
        print(f"Dataset: {dataset_name}")
        print(table)
        print("\n")


results_folder = "results/classifier_on_homdensities/"

print("MAIN")
csv_file_list = [f"{results_folder}r_MOLBACE_16_50_all_tw1.csv",
                 f"{results_folder}r_MOLBBBP_16_50_all_tw1.csv",
                 f"{results_folder}r_MOLHIV_16_50_all_tw1.csv",
                 f"{results_folder}r_MOLLIPO_16_50_all_tw1.csv",
                 f"{results_folder}r_GITHUB_STARGAZERS_16_50_all_tw1.csv",
                 f"{results_folder}r_REDDIT-BINARY_16_50_all_tw1.csv",
                 f"{results_folder}r_REDDIT-MULTI-5K_16_50_all_tw1.csv"]
get_result_summary(csv_file_list)

print("FEATURES")
csv_file_list = [f"{results_folder}rf_MOLBACE_16_50_all_tw1.csv",
                 f"{results_folder}rf_MOLBBBP_16_50_all_tw1.csv",
                 f"{results_folder}rf_MOLHIV_16_50_all_tw1.csv",
                 f"{results_folder}rf_MOLLIPO_16_50_all_tw1.csv"]
get_result_summary(csv_file_list)

print("FEATURES ONLY")
csv_file_list = [f"{results_folder}rfo_MOLBACE_16_50_all_tw1.csv",
                 f"{results_folder}rfo_MOLBBBP_16_50_all_tw1.csv",
                 f"{results_folder}rfo_MOLHIV_16_50_all_tw1.csv",
                 f"{results_folder}rfo_MOLLIPO_16_50_all_tw1.csv"]
get_result_summary(csv_file_list, features_only=True)

print("GLOBAL SENSITIVITY")
csv_file_list = [f"{results_folder}rgs_MOLBACE_16_50_all_tw1.csv",
                 f"{results_folder}rgs_MOLBBBP_16_50_all_tw1.csv",
                 f"{results_folder}rgs_MOLHIV_16_50_all_tw1.csv",
                 f"{results_folder}rgs_MOLLIPO_16_50_all_tw1.csv"]
get_result_summary(csv_file_list, global_sensitivity=True)