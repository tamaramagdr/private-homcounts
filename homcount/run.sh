#!/bin/bash

python pattern_extractors/hom.py --data ogbg-molhiv --seed 69 --dloc data/graphdbs --oloc data/homcount --pattern_count 50 --run_id timingmolhiv_seed69_tw1 --hom_type full_kernel --hom_size 16 --max_treewidth=1