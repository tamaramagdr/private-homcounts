# private-homcounts 
Private and expressive graph representation learning.

## How to use
Clone the repository as usual and install the requirements.

The repository [Git Large File Storage (Git LFS)](https://git-lfs.github.com/) to manage large files.
Make sure you have Git LFS installed and then run:

```bash
git lfs install
git lfs pull
```

This downloads the pre-computed homomorphism counts in the `data/homomorphism_counts` folder.

The main script to compute the private and expressive graph representations and train a classifier is `train_on_homcounts.py`,
and can be run as:

```
python train_on_homdensities.py --rho "$rho" --dataset "$dataset" --save_path "$save_path" --output_file "$output_file" --seed "$seed" --run "$run"
```
where `dataset` is in `[molhiv, molbace, moltox21]`.

Please note that the versions in the `requirements.txt` may need some adjustment as there is a bug with `torch>=2.6.0` and `ogb==1.3.6.`, and smaller versions may be necessary.

## Submodules and credits
The repository contains a copy of the `homcount` repository by [pwelke](https://github.com/pwelke/homcount), and relies on the `homsub` submodule also forked from [pwelke](https://github.com/pwelke/homsub).
