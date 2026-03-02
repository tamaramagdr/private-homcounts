#!/bin/bash

mkdir -p results/test

datasets=(molhiv molbace molbbbp mollipo github_stargazers reddit-binary reddit-multi-5k)

base_args=(
  --rho 0.01
  --max_treewidth 1
  --save_path ""
  --seed 1
  --run 69
  --run_attacks
)

for dataset in "${datasets[@]}"; do
  python train_on_homdensities.py \
    "${base_args[@]}" \
    --dataset "$dataset" \
    --output_file "results/test/rf_r"

  python train_on_homdensities.py \
    "${base_args[@]}" \
    --dataset "$dataset" \
    --output_file "results/test/f" \
    --use_features

  python train_on_homdensities.py \
    "${base_args[@]}" \
    --dataset "$dataset" \
    --output_file "results/test/fo" \
    --use_features \
    --features_only
done

datasets=(sbm tree_sbm)

for dataset in "${datasets[@]}"; do
  python train_on_homdensities.py \
    "${base_args[@]}" \
    --dataset "$dataset" \
    --output_file "results/test/rf_r"
done
