#!/bin/bash

#rhos=(0.00001 0.05 0.1 0.2 0.3 0.5 0.7 1.0 1.5 2 3 4 5 7 10 15 20 30 40 50 "inf")
#rhos=(0.00001 0.05 1.0 10 100 "inf")
#rhos=(0.00000001 0.000001 0.00001 0.0001 0.001 0.005 0.01 0.05 1.0 "inf")
#rhos=(0.005)
rhos=(0.001)
tws=(3)
#rhos=("inf")
save_path="run_$(date +%Y%m%d_%H%M%S)"
dataset="molhiv"
runs=(666)
output_file="results/classifier_on_homcounts/here_${dataset}_tw_output_$(date +%Y%m%d_%H%M%S)"

for seed in {1..3}; do
  echo "Running train_on_homcounts.py with seed=${seed}"
  for rho in "${rhos[@]}"; do
    echo "Running train_on_homcounts.py with rho=${rho}"
    for tw in "${tws[@]}"; do
      echo "Running train_on_homcounts.py with tw=${tw}"
      for run in "${runs[@]}"; do
        echo "Running train_on_homcounts.py with run=${run}"
        python train_on_homdensities.py --rho "$rho" --dataset "$dataset" --save_path "$save_path" --output_file "$output_file" --seed "$seed" --run "$run" --max_treewidth "$tw"
      done
    done
  done
done