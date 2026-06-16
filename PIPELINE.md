# ImageNet Embedding 提取与图构建流程

本文档记录了 ImageNet 数据集上提取 embedding、聚类和构建图的脚本及其使用方法。

## 流程概览

```
inference.py    →    cluster.py         →    build_graph.py
   ↓                      ↓                         ↓
[提取 embedding]    [按类KMeans聚类]         [构建相似度图]
   ↓                      ↓                         ↓
  .pkl/.json          parquet 文件            .graph.pkl
```

---

## 1. Embedding 提取脚本

### `graph_construction/inference.py`

对 ImageNet 样本进行推理，提取模型 embedding 层特征。

**主要功能：**
- 加载 ImageNet 数据集（支持分布式多 GPU）
- 使用模型（ResNet, Swin Transformer 等）进行推理
- 通过 `capture_feature.py` 提取 embedding 层特征
- 保存每个样本的详细信息

**输出字段：**
| 字段 | 说明 |
|------|------|
| `image_name` | 图片路径 |
| `logits` | 模型输出 logits |
| `loss` | 交叉熵损失 |
| `feature` | embedding 向量 |
| `true_class` | 真实类别 ID |
| `pred_class` | 预测类别 ID |
| `correct` | 是否预测正确 |

**使用示例：**
```bash
python graph_construction/inference.py \
    --data-path /path/to/imagenet \
    --model swin_t \
    --output-dir /path/to/output \
    --resume /path/to/checkpoint.pth \
    --test-only
```

---

## 2. 聚类脚本

### `graph_construction/cluster.py`

基于 embedding 对每个类内部的样本进行 KMeans 聚类。

**主要功能：**
- 读取 inference 输出的 pkl 文件
- 按 `true_class` 分组
- 使用 Faiss 进行 KMeans 聚类
- 保存聚类结果为 parquet 文件

**使用示例：**
```bash
python graph_construction/cluster.py \
    --input /path/to/embeddings \
    --output /path/to/clusters
```

**参数说明：**
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `feature_column` | `"feature"` | embedding 列名 |
| `n_clusters_per_class` | 25 | 每个类聚成几个子簇 |
| `max_samples_per_class` | 5000 | 每类最多使用的样本数 |
| `kmeans_iter` | 25 | KMeans 迭代次数 |
| `use_gpu` | False | 是否使用 GPU 加速 |
| `max_rows_per_file` | 10000 | 每个 parquet 最大行数 |

**输出结构：**
```
output_dir/
└── class_{id}/
    ├── centroids.npy          # 聚类中心
    ├── cluster_mapping.json   # 聚类映射信息
    └── clusters/
        ├── cluster_0000_00000.parquet
        ├── cluster_0000_00001.parquet
        └── ...
```

---

## 3. 图构建脚本

### `graph_construction/build_graph.py`

根据聚类结果构建样本间的相似度图。

**主要功能：**
- 加载聚类结果（parquet 文件）
- 计算样本间的余弦相似度/距离
- 构建全连接图
- 保存图数据为 pkl 文件

**图数据结构：**
```python
{
    'nodes': [...],           # 节点列表
    'edges': [...],           # 边列表 (i, j, weight)
    'adj_matrix': csr_matrix, # 邻接矩阵（稀疏）
    'num_nodes': N,
    'avg_loss': float,
    'within_cluster_feature_variance': float,
}
```

**使用示例：**
```bash
python graph_construction/build_graph.py \
    --input /path/to/clusters \
    --output /path/to/graphs \
    --feature_col feature
```

**相关文件：**
- `build_graph_fast.py`: 优化版本（使用 numpy 矩阵乘法加速）

---

## 4. 其他相关文件

| 文件 | 说明 |
|------|------|
| `graph_construction/capture_feature.py` | 特征提取工具，被 inference.py 调用 |
| `training/imagenet/train_graph_static_prob.py` | 使用图信息进行静态概率训练 |
| `training/imagenet/train_online.py` | PFB 在线剪枝训练 |

---

## 5. 典型使用流程

```bash
# Step 1: 提取 embedding
python graph_construction/inference.py \
    --data-path /path/to/imagenet \
    --model swin_t \
    --output-dir /path/to/output/embeddings \
    --resume checkpoint.pth \
    --test-only

# Step 2: 聚类
python graph_construction/cluster.py \
    --input /path/to/output/embeddings \
    --output /path/to/output/clusters

# Step 3: 构建图
python graph_construction/build_graph.py \
    --input /path/to/output/clusters \
    --output /path/to/output/graphs
```

---

## 6. 输出文件

生成的 graph 文件（`*.graph.pkl`）将保存在指定的输出目录中，可直接用于 `train_graph_static_prob.py` 的 `--graph-dir` 参数。
