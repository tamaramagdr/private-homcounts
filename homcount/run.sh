#!/bin/bash

python pattern_extractors/hom.py --data csl --seed 69 --dloc data/graphdbs --oloc data/homcount --pattern_count 50 --run_id csl_seed69_tw1 --hom_type full_kernel --hom_size 8 --max_treewidth=1
python pattern_extractors/hom.py --data csl --seed 420 --dloc data/graphdbs --oloc data/homcount --pattern_count 50 --run_id csl_seed420_tw1 --hom_type full_kernel --hom_size 8 --max_treewidth=1
python pattern_extractors/hom.py --data csl --seed 69 --dloc data/graphdbs --oloc data/homcount --pattern_count 50 --run_id csl_seed69_tw1 --hom_type full_kernel --hom_size 16 --max_treewidth=1
python pattern_extractors/hom.py --data csl --seed 420 --dloc data/graphdbs --oloc data/homcount --pattern_count 50 --run_id csl_seed420_tw1 --hom_type full_kernel --hom_size 16 --max_treewidth=1
python pattern_extractors/hom.py --data csl --seed 69 --dloc data/graphdbs --oloc data/homcount --pattern_count 50 --run_id csl_seed69_tw2 --hom_type full_kernel --hom_size 8 --max_treewidth=2
python pattern_extractors/hom.py --data csl --seed 420 --dloc data/graphdbs --oloc data/homcount --pattern_count 50 --run_id csl_seed420_tw2 --hom_type full_kernel --hom_size 8 --max_treewidth=2
python pattern_extractors/hom.py --data csl --seed 69 --dloc data/graphdbs --oloc data/homcount --pattern_count 50 --run_id csl_seed69_tw2 --hom_type full_kernel --hom_size 16 --max_treewidth=2
python pattern_extractors/hom.py --data csl --seed 420 --dloc data/graphdbs --oloc data/homcount --pattern_count 50 --run_id csl_seed420_tw2 --hom_type full_kernel --hom_size 16 --max_treewidth=2
python pattern_extractors/hom.py --data csl --seed 69 --dloc data/graphdbs --oloc data/homcount --pattern_count 50 --run_id csl_seed69_tw3 --hom_type full_kernel --hom_size 8 --max_treewidth=3
python pattern_extractors/hom.py --data csl --seed 420 --dloc data/graphdbs --oloc data/homcount --pattern_count 50 --run_id csl_seed420_tw3 --hom_type full_kernel --hom_size 8 --max_treewidth=3
python pattern_extractors/hom.py --data csl --seed 69 --dloc data/graphdbs --oloc data/homcount --pattern_count 50 --run_id csl_seed69_tw3 --hom_type full_kernel --hom_size 16 --max_treewidth=3
python pattern_extractors/hom.py --data csl --seed 420 --dloc data/graphdbs --oloc data/homcount --pattern_count 50 --run_id csl_seed420_tw3 --hom_type full_kernel --hom_size 16 --max_treewidth=3

python pattern_extractors/hom.py --data zinc_subset --seed 69 --dloc data/graphdbs --oloc data/homcount --pattern_count 50 --run_id zinc_seed69_tw1 --hom_type full_kernel --hom_size 16 --max_treewidth=1
python pattern_extractors/hom.py --data zinc_subset --seed 420 --dloc data/graphdbs --oloc data/homcount --pattern_count 50 --run_id zinc_seed420_tw1 --hom_type full_kernel --hom_size 16 --max_treewidth=1
python pattern_extractors/hom.py --data zinc_subset --seed 69 --dloc data/graphdbs --oloc data/homcount --pattern_count 50 --run_id zinc_seed69_tw2 --hom_type full_kernel --hom_size 8 --max_treewidth=2
python pattern_extractors/hom.py --data zinc_subset --seed 420 --dloc data/graphdbs --oloc data/homcount --pattern_count 50 --run_id zinc_seed420_tw2 --hom_type full_kernel --hom_size 8 --max_treewidth=2
python pattern_extractors/hom.py --data zinc_subset --seed 69 --dloc data/graphdbs --oloc data/homcount --pattern_count 50 --run_id zinc_seed69_tw2 --hom_type full_kernel --hom_size 16 --max_treewidth=2
python pattern_extractors/hom.py --data zinc_subset --seed 420 --dloc data/graphdbs --oloc data/homcount --pattern_count 50 --run_id zinc_seed420_tw2 --hom_type full_kernel --hom_size 16 --max_treewidth=2

python pattern_extractors/hom.py --data ogbg-molhiv --seed 69 --dloc data/graphdbs --oloc data/homcount --pattern_count 50 --run_id molhiv_seed69_tw1 --hom_type full_kernel --hom_size 16 --max_treewidth=1
python pattern_extractors/hom.py --data ogbg-molhiv --seed 420 --dloc data/graphdbs --oloc data/homcount --pattern_count 50 --run_id molhiv_seed420_tw1 --hom_type full_kernel --hom_size 16 --max_treewidth=1
python pattern_extractors/hom.py --data ogbg-molhiv --seed 69 --dloc data/graphdbs --oloc data/homcount --pattern_count 50 --run_id molhiv_seed69_tw1 --hom_type full_kernel --hom_size 16 --max_treewidth=2
python pattern_extractors/hom.py --data ogbg-molhiv --seed 420 --dloc data/graphdbs --oloc data/homcount --pattern_count 50 --run_id molhiv_seed420_tw1 --hom_type full_kernel --hom_size 16 --max_treewidth=2
python pattern_extractors/hom.py --data ogbg-molhiv --seed 69 --dloc data/graphdbs --oloc data/homcount --pattern_count 50 --run_id molhiv_seed69_tw1 --hom_type full_kernel --hom_size 8 --max_treewidth=1
python pattern_extractors/hom.py --data ogbg-molhiv --seed 420 --dloc data/graphdbs --oloc data/homcount --pattern_count 50 --run_id molhiv_seed420_tw1 --hom_type full_kernel --hom_size 8 --max_treewidth=1
python pattern_extractors/hom.py --data ogbg-molhiv --seed 69 --dloc data/graphdbs --oloc data/homcount --pattern_count 50 --run_id molhiv_seed69_tw1 --hom_type full_kernel --hom_size 8 --max_treewidth=2
python pattern_extractors/hom.py --data ogbg-molhiv --seed 420 --dloc data/graphdbs --oloc data/homcount --pattern_count 50 --run_id molhiv_seed420_tw1 --hom_type full_kernel --hom_size 8 --max_treewidth=2
