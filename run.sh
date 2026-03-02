#!/bin/bash

rhos=(0.00000001 0.000001 0.00001 0.0001 0.001 0.005 0.01 0.05 1 "inf")
#rhos=(0.01 "inf")
save_path=""
dataset="molbbbp"
runs=(69 420 666)
output_file="results/classifier_on_homdensities/r"
#args="--run_attacks --use_features --features_only"
#args="--use_features --features_only"
#args="--use_features"
args="--run_attacks"
#args="--run_attacks --use_cycles_and_tw"
#args="--global_sensitivity_ablation"
max_treewidth=1
#pattern_count=(10 20 30 40 50)

#for pc in "${pattern_count[@]}"; do
for run in "${runs[@]}"; do
  for seed in {1..3}; do
    echo "Running train_on_homdensities.py with seed=${seed}"
    for rho in "${rhos[@]}"; do
      echo "Running train_on_homdensities.py with rho=${rho}"
#      echo "Pattern count ablation with pc=${pc}"
      python train_on_homdensities.py --rho "$rho" --max_treewidth "$max_treewidth" --dataset "$dataset" --save_path "$save_path" --output_file "$output_file" --seed "$seed" --run "$run" $args
#        python train_on_homdensities.py --rho "$rho" --max_treewidth "$max_treewidth" --dataset "$dataset" --save_path "$save_path" --output_file "$output_file" --seed "$seed" --run "$run" $args --pattern_count_ablation "$pc"
    done
  done
done
#done
