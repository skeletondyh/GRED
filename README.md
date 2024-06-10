# GRED

This repo contains the official code for our ICML 2024 paper: **Recurrent Distance Filtering for Graph Representation Learning** (https://arxiv.org/abs/2312.01538)

## Setup

The code is developed using JAX and Flax (since we want to use `associative_scan`):

```
python==3.10.12
jaxlib==0.4.14+cuda11.cudnn86
flax==0.7.2
optax==0.1.7   # optimizer lib
numpy==1.25.2
scipy==1.11.2
scikit-learn==1.3.0
```
Please refer to the [JAX](https://jax.readthedocs.io/en/latest/installation.html) and [Flax](https://github.com/google/flax?tab=readme-ov-file#quick-install) installation pages.

To keep the data preprocessing consistent with other baselines, we load the datasets using `torch-geometric==2.3.1` and convert them into NumPy arrays.
You need `torch-geometric` to run `preprocess.py` and `preprocess_peptides.py`, but you don't need it to run training scripts.

## Data Preprocessing

To prepare MNIST, CIFAR10, ZINC, PATTERN, and CLUSTER, please run:
```
python preprocess.py
```
To prepare Peptides-func and Peptides-struct, please run:
```
python preprocess_peptides.py
```

## Training

For MNIST and CIFAR10:
```
python train_pixel.py --name MNIST --num_layers 4 --num_hops 3 --dim_h 128 --dim_v 96
python train_pixel.py --name CIFAR10 --num_layers 8 --num_hops 5 --dim_h 96 --dim_v 64
```

For ZINC:
```
python train_zinc.py
```
For CLUSTER and PATTERN:
```
python train_sbm.py --name CLUSTER --num_layers 16 --dim_h 64 --dim_v 64 --weight_decay 0.2 --r_min 0.9
python train_sbm.py --name PATTERN --num_layers 10 --dim_h 72 --dim_v 64 --weight_decay 0.1 --r_min 0.5
```

For Peptides-func and Peptides-struct:
```
python train_peptides_func.py
python train_peptides_struct.py
```
---

If you have any questions regarding the code, please feel free to raise an issue.
If you find our paper helpful in your research, please consider citing it:
```
@inproceedings{ding2024gred,
title={Recurrent Distance Filtering for Graph Representation Learning},
author={Yuhui Ding and Antonio Orvieto and Bobby He and Thomas Hofmann},
booktitle={Forty-first International Conference on Machine Learning},
year={2024}
}
```
