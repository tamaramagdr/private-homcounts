# Expressive and Private Graph Representations
This repository contains the code for the ICLR 2026 paper "On The Trade-Off Between Expressivity and Privacy in Graph Representation Learning", available on [OpenReview](https://openreview.net/pdf?id=XXLDvwMwbe).

## How to use
Clone the repository as usual and install the requirements.
To include the `homcount` submodule, clone with the `--recurse-submodules` flag.

The repository uses [Git Large File Storage (Git LFS)](https://git-lfs.github.com/) to manage large files.
Make sure you have Git LFS installed and then run:

```bash
git lfs install
git lfs pull
```

This downloads the pre-computed homomorphism counts in the `data/homomorphism_counts` folder.

The main script to compute the private and expressive graph representations and train a classifier is `train_on_homdensities.py`,
and can be run as:

```
python train_on_homdensities.py --rho "$rho" --dataset "$dataset" --save_path "$save_path" --output_file "$output_file" --seed "$seed" --run "$run"
```
where `dataset` is in `[molbbbp, molbace, molhiv, mollipo, github-stargazers, reddit-binary, reddit-multi-5k, sbm, tree_sbm]`.

Please note that the versions in the `requirements.txt` may need some adjustment as there is a bug with `torch>=2.6.0` and `ogb==1.3.6.`, and smaller versions may be necessary.

To run the GIN baseline, use `python gin_baseline_ogbg.py` and append `--rr` for [Randomized Response](https://ceur-ws.org/Vol-1558/paper35.pdf) and `--rr --deg_preserving` for [degree preserving Randomized Response](https://www.tdp.cat/issues21/tdp.a521a23.pdf).

The `results/` folder contains results and visualizations for the experiments.
Note that these results are obtained using a slightly improved version of the code, that uses the number of nodes as a 
bound for the maximum degree of the graphs in case this provides a tighter bound than domain knowledge.
We thank the reviewers for their comments on this.
The previous computation can be recovered by modifying the `compute_local_bounded_degree_sensitivity` function in `noise_and_sensitivity.py`.
The results reported in the paper are stored in `results/classifier_on_homdensities/paper_results/`.

## Submodules and credits
The repository contains a copy of the `homcount` repository by [pwelke](https://github.com/pwelke/homcount), and relies on the `HomSub` submodule also forked from [pwelke](https://github.com/pwelke/homsub).

## Citation

If you use our code please cite us as

```
@inproceedings{
  indri2026on,
  title={On the trade-off between expressivity and privacy in graph representation learning},
  author={Patrick Indri and Tamara Drucks and Thomas G{\"a}rtner},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=XXLDvwMwbe}
}
```
