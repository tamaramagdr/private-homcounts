#!/bin/bash

epsilons=(0.00001 1 5 10 50 100 1000 10000 100000 "inf")
#epsilons=(100 1000 10000 "inf")
#epsilons=(0.00001 1 10)
#epsilons=("inf")
save_path="run_$(date +%Y%m%d_%H%M%S)"
output_file="results/classifier_on_homcounts/privacy_attack_$(date +%Y%m%d_%H%M%S)"

for seed in {0..2}; do
  echo "Running edge_privacy_attack.py with seed=${seed}"
  for epsilon in "${epsilons[@]}"; do
    echo "Running train_on_homcounts.py with epsilon=${epsilon}"
    python legacy_attack.py --epsilon "$epsilon" --dataset "molhiv" --save_path "$save_path" --output_file "$output_file" --seed "$seed"
  done
done