# UGIES: Unified Graph-based Importance Evaluation System

Official implementation of **"Selecting Samples on Graphs: A Unified Dataset Pruning Framework for Lossless Training Acceleration"** (ICML 2026).

> Dongyue Wu, Zilin Guo, Xiaoyu Li, Jiajia Liu, Jingdong Chen, Nong Sang, Changxin Gao

UGIES is a unified graph-based dataset pruning framework that models the training dataset as a weighted graph, where **node weights** encode intrinsic sample importance (e.g., entropy, loss) and **edge weights** encode extrinsic pairwise interactions (e.g., cosine similarity). Dataset pruning is then cast as a **Maximum Weight Clique Problem (MWCP)**, solved efficiently via a principled greedy algorithm with formal approximation guarantees. UGIES reduces training time by over **40%** without sacrificing accuracy on ImageNet-1k with ResNet-50.

## Highlights

- **Unified framework**: Jointly models intrinsic and extrinsic sample importance via a graph-based formulation.
- **Theoretical guarantees**: The unified objective is provably submodular under mild conditions, enabling a greedy solver with bounded approximation ratio.
- **Flexible instantiation**: Supports diverse importance metrics (entropy, loss, gradient norm, etc.) while preserving guarantees.
- **Lossless acceleration**: Achieves 43.2% total time reduction on ImageNet-1k (ResNet-50, 50% pruning) with no accuracy loss.

## Requirements

- Python >= 3.8
- PyTorch >= 2.0
- torchvision >= 0.15
- CUDA-enabled GPU

### Install dependencies

```bash
pip install -r requirements.txt
```

| Package | Version | Purpose |
|---------|---------|---------|
| torch | >= 2.0 | Core deep learning framework |
| torchvision | >= 0.15 | Models, datasets, transforms |
| numpy | - | Numerical computation |
| scipy | - | Sparse matrix operations for graph |
| pandas | - | DataFrame processing for clustering |
| faiss-cpu | - | Fast KMeans clustering |
| tqdm | - | Progress bar |

## Data Preparation

### ImageNet-1k

Download [ImageNet-1k](http://image-net.org/) and organize as:

```
/path/to/imagenet/
├── train/
│   ├── n01440764/
│   └── ...
└── val/
    ├── n01440764/
    └── ...
```

### CIFAR-10/100

CIFAR-10/100 will be automatically downloaded by torchvision.

## Repository Structure

```
UGIES/
├── graph_construction/            # Preprocessing pipeline
│   ├── inference.py               # Extract embeddings from pretrained models
│   ├── capture_feature.py         # Feature extraction hook utility
│   ├── cluster.py                 # FAISS KMeans (Structured Graph Sparsification)
│   ├── build_graph.py             # Build similarity graph (standard)
│   ├── build_graph_fast.py        # Build similarity graph (optimized)
│   ├── cifar_cluster.py           # CIFAR-100 clustering
│   ├── cifar_build_graph.py       # CIFAR-100 cluster-based graph
│   └── cifar_build_full_graph.py  # CIFAR-100 full graph
├── training/
│   ├── imagenet/                  # ImageNet training
│   │   ├── train_graph_static_prob.py  # UGIES epoch-wise pruning (main)
│   │   ├── train_online.py        # UGIES + PFB online acceleration
│   │   ├── train.py               # Baseline (no pruning)
│   │   └── train_unsup.py         # InfoBatch baseline
│   └── cifar/                     # CIFAR-100 training
│       ├── cifar_graph_prune.py        # UGIES cluster-based graph
│       └── cifar_full_graph_prune.py   # UGIES full graph
├── models/                        # Swin Transformer, ResNet
├── scripts/                       # Example shell scripts
├── PIPELINE.md                    # Detailed preprocessing docs
├── requirements.txt
└── README.md
```

## Method Overview

UGIES defines a unified importance score for each sample:

**I(x_i | S) = a * I_in(x_i) + I_ex(x_i | S)**

- **I_in(x_i)**: Intrinsic importance (e.g., prediction entropy), independent of other samples.
- **I_ex(x_i | S)**: Extrinsic importance from pairwise interactions, promoting diversity.
- **a**: Coefficient balancing intrinsic vs. extrinsic contributions.

The pipeline builds a sparsified graph via **Structured Graph Sparsification** (per-class KMeans clustering), then performs epoch-wise greedy selection based on unified importance.

## Graph Construction (Preprocessing)

See [PIPELINE.md](PIPELINE.md) for detailed documentation.

### ImageNet

```bash
# Step 1: Extract embeddings
python graph_construction/inference.py \
    --data-path /path/to/imagenet \
    --model swin_t \
    --output-dir /path/to/output/embeddings \
    --resume /path/to/pretrained_checkpoint.pth \
    --test-only

# Step 2: Structured Graph Sparsification (per-class KMeans)
python graph_construction/cluster.py \
    --input /path/to/output/embeddings \
    --output /path/to/output/clusters

# Step 3: Build similarity graph
python graph_construction/build_graph.py \
    --input /path/to/output/clusters \
    --output /path/to/output/graphs
```

### CIFAR-100

```bash
# Cluster embeddings
python graph_construction/cifar_cluster.py \
    --input /path/to/cifar100_embeddings.pt \
    --output /path/to/cifar100_clusters \
    --n-clusters 10 --kmeans-iter 15

# Build graph
python graph_construction/cifar_build_graph.py \
    --input /path/to/cifar100_clusters \
    --output /path/to/cifar100_graphs

# Alternative: full graph (without clustering)
python graph_construction/cifar_build_full_graph.py \
    --input /path/to/cifar100_embeddings.pt \
    --output /path/to/cifar100_full_graph.pkl \
    --batch-size 5000
```

## Training

### CIFAR-100 with ResNet-18

**Baseline** (no pruning):
```bash
python training/cifar/cifar_graph_prune.py \
    --model r18 --num_epoch 200 --batch-size 128 \
    --max-lr 0.05 --dataloader-ratio 1.0
```

**UGIES** (our method):
```bash
python training/cifar/cifar_full_graph_prune.py \
    --graph-path /path/to/cifar100/full_graph.pkl \
    --dataloader-ratio 0.625 --score-weight 1.0 \
    --model r18 --num_epoch 200 --batch-size 128 --max-lr 0.05
```

### ImageNet-1k with Swin-T

All experiments use 4 GPUs. Adjust `--nproc_per_node` and `--batch-size` for your setup.

**Baseline** (no pruning):
```bash
torchrun --nproc_per_node=4 training/imagenet/train.py \
    --model swin_t --data-path /path/to/imagenet \
    --epochs 300 --batch-size 256 --opt adamw --lr 0.001 \
    --weight-decay 0.05 --norm-weight-decay 0.0 \
    --bias-weight-decay 0.0 --transformer-embedding-decay 0.0 \
    --lr-scheduler cosineannealinglr --lr-min 0.00001 \
    --lr-warmup-method linear --lr-warmup-epochs 20 --lr-warmup-decay 0.01 \
    --amp --label-smoothing 0.1 --mixup-alpha 0.8 \
    --clip-grad-norm 5.0 --cutmix-alpha 1.0 --random-erase 0.25 \
    --interpolation bicubic --auto-augment ta_wide \
    --model-ema --ra-sampler --ra-reps 4 --val-resize-size 224 \
    --output-dir /path/to/output
```

**UGIES epoch-wise pruning** (main method):
```bash
torchrun --nproc_per_node=4 training/imagenet/train_graph_static_prob.py \
    --model swin_t --data-path /path/to/imagenet \
    --graph-dir /path/to/graphs \
    --epochs 300 --batch-size 256 --opt adamw --lr 0.001 \
    --weight-decay 0.05 --norm-weight-decay 0.0 \
    --bias-weight-decay 0.0 --transformer-embedding-decay 0.0 \
    --lr-scheduler cosineannealinglr --lr-min 0.00001 \
    --lr-warmup-method linear --lr-warmup-epochs 20 --lr-warmup-decay 0.01 \
    --amp --label-smoothing 0.1 --mixup-alpha 0.8 \
    --clip-grad-norm 5.0 --cutmix-alpha 1.0 --random-erase 0.25 \
    --interpolation bicubic --auto-augment ta_wide \
    --model-ema --ra-sampler --ra-reps 4 --val-resize-size 224 \
    --dataloader-ratio 0.625 --score-weight 0.2 \
    --output-dir /path/to/output
```

**UGIES + PFB online acceleration** (variant):
```bash
torchrun --nproc_per_node=4 training/imagenet/train_online.py \
    --model swin_t --data-path /path/to/imagenet \
    --epochs 300 --batch-size 256 --opt adamw --lr 0.001 \
    --weight-decay 0.05 --norm-weight-decay 0.0 \
    --bias-weight-decay 0.0 --transformer-embedding-decay 0.0 \
    --lr-scheduler cosineannealinglr --lr-min 0.00001 \
    --lr-warmup-method linear --lr-warmup-epochs 20 --lr-warmup-decay 0.01 \
    --amp --label-smoothing 0.1 --mixup-alpha 0.8 \
    --clip-grad-norm 5.0 --cutmix-alpha 1.0 --random-erase 0.25 \
    --interpolation bicubic --auto-augment ta_wide \
    --model-ema --ra-sampler --ra-reps 4 --val-resize-size 224 \
    --use-online --ratio 0.4 --start-end 16 260 \
    --output-dir /path/to/output
```

## Key Parameters

### UGIES Epoch-wise Pruning (`train_graph_static_prob.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--dataloader-ratio` | required | Selected data fraction per epoch, i.e., (1-p) |
| `--score-weight` | required | Coefficient a for intrinsic vs. extrinsic balance |
| `--graph-dir` | required | Directory with pre-built graph files |

### Online Pruning with PFB (`train_online.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--use-online` | `False` | Enable online pruning with PFB |
| `--ratio` | `0.5` | Pruning ratio |
| `--kernel` | `64` | KDE kernel centers |
| `--kernel-channel` | `128` | KDE kernel feature dim |
| `--alpha` | `0.01` | EMA decay for kernel updates |
| `--start-end` | `[16, 260]` | Pruning start/end epochs |

## Results

### CIFAR-100 with ResNet-18

| Pruning Ratio | 30% | 50% | 70% |
|---------------|-----|-----|-----|
| Full Data | 78.2 | 78.2 | 78.2 |
| InfoBatch | 78.2 | 78.1 | 76.5 |
| DivBS | 78.5 | 78.2 | 77.2 |
| **UGIES (Ours)** | **78.9** | **78.6** | **77.6** |

### ImageNet-1k with ResNet-50 (Efficiency)

| Method | Acc(%) | Total Time(h) | Reduction |
|--------|--------|---------------|-----------|
| Full Data | 76.4 | 13.9 | - |
| InfoBatch | 76.6 | 10.2 | 26.6% |
| PFB | 76.4 | 8.7 | 37.4% |
| **UGIES (Ours)** | **76.5** | **7.9** | **43.2%** |

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{wu2026ugies,
  title={Selecting Samples on Graphs: A Unified Dataset Pruning Framework for Lossless Training Acceleration},
  author={Wu, Dongyue and Guo, Zilin and Li, Xiaoyu and Liu, Jiajia and Chen, Jingdong and Sang, Nong and Gao, Changxin},
  booktitle={Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  year={2026}
}
```

## Acknowledgements

The CIFAR training code is built upon [InfoBatch](https://github.com/NUS-HPC-AI-LAB/InfoBatch) (ICLR 2024 Oral). The ImageNet training scripts are adapted from the [torchvision reference](https://github.com/pytorch/vision/tree/main/references/classification). The online pruning variant builds upon PFB (ICCV 2025).
