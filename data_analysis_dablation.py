import glob
import re

import pandas as pd


def get_result_summary(csv_file_list, features_only = False, eps_correction=0, attack_csv_file=None):
    for csv_file in csv_file_list:
        df_list = []
        # Get the names of all the csv files starting with 'csv_file' and ending with '.csv'.
        csv_names = glob.glob(f"{csv_file}*.csv")
        # For each of the files, read the two digits after _16_ and before .csv, which is the homcount size.
        for name in csv_names:
            match = re.search(r'_16_(\d{2})', name)
            size = int(match.group(1))
            df = pd.read_csv(name, header=None, names=["run", "noise", "rho", "delta", "roc_auc", "acc", "seed",
                                                       "sep", "sensitivity", "top_1_accuracy", "top_k_accuracy", "epsilon", "tw"])
            df["size"] = size
            df_list.append(df)
        df = pd.concat(df_list, ignore_index=True)
        # Compute the avg and std for roc_auc and acc for each size.
        grouped_by_size = df.groupby("size").agg(
            roc_auc_mean=('roc_auc', 'mean'),
            roc_auc_std=('roc_auc', 'std'),
            acc_mean=('acc', 'mean'),
            acc_std=('acc', 'std')
        ).reset_index()

        dataset_name = csv_file.split('/')[2].split('_')[2].upper()
        print(dataset_name)
        print(grouped_by_size)
        # Save name is the same as before last /, and then dataset name, and then _summary_by_size.csv
        save_name = '/'.join(csv_file.split('/')[:-1]) + f"/ra_ablation_{dataset_name}_eps1_summary_by_size.csv"
        # Write the grouped_by_size to a csv file named f"{csv_file}_summary_by_size.csv", with 5 digits after the decimal point.
        grouped_by_size.to_csv(f"{save_name}", index=False, float_format='%.5f')


csv_file_list = ["results/classifier_on_homdensities/r_ablation_MOLBACE",
                 "results/classifier_on_homdensities/r_ablation_MOLLIPO",
                 "results/classifier_on_homdensities/r_ablation_MOLBBBP",
                 "results/classifier_on_homdensities/r_ablation_MOLHIV"]
get_result_summary(csv_file_list)
