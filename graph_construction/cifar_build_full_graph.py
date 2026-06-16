"""
CIFAR100 全量图构建脚本（不按类别聚类）
直接基于所有样本的 embedding 构建完整的全连接图

优化版本：
- 直接计算 edge scores，不构建完整的邻接矩阵
- 大幅减少内存占用和存储时间

用法:
    python cifar_build_full_graph.py \
        --input /path/to/train_embeddings.pt \
        --output /path/to/full_graph.pkl \
        --batch-size 5000
"""

import os
import pickle
import numpy as np
import torch
import argparse
import time
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Any


def format_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def load_embeddings(embedding_path: str, verbose: bool = True):
    """
    加载 CIFAR100 embedding 文件

    返回:
        indices: np.ndarray [N]
        labels: np.ndarray [N]
        embeddings: np.ndarray [N, D]
    """
    if verbose:
        print(f"Loading embeddings from {embedding_path}...")

    data = torch.load(embedding_path, map_location='cpu', weights_only=False)

    indices = data['indices'].numpy() if isinstance(data['indices'], torch.Tensor) else data['indices']
    labels = data['labels'].numpy() if isinstance(data['labels'], torch.Tensor) else data['labels']
    embeddings = data['embeddings'].numpy() if isinstance(data['embeddings'], torch.Tensor) else data['embeddings']

    if verbose:
        print(f"Loaded {len(indices)} samples, embedding dim: {embeddings.shape[1]}")
        print(f"Number of classes: {len(np.unique(labels))}")

    return indices, labels, embeddings


def compute_edge_scores_from_embeddings(
    embeddings: np.ndarray,
    indices: np.ndarray,
    batch_size: int = 5000,
    verbose: bool = True
) -> Dict[int, float]:
    """
    直接从 embeddings 计算每个节点的平均边权分数
    不构建完整的邻接矩阵，分批计算以节省内存

    参数:
        embeddings: [N, D] 特征矩阵
        indices: [N] 样本索引
        batch_size: 每批处理的样本数
        verbose: 是否显示进度

    返回:
        scores: Dict[idx -> avg_edge_weight]
    """
    N = embeddings.shape[0]

    if verbose:
        print(f"Computing edge scores for {N} samples...")
        print(f"Method: batch computation without full adjacency matrix")

    t0 = time.perf_counter()

    # 归一化
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    X = embeddings / norms

    # 使用累积方式计算每个节点的边权和与边数
    edge_weight_sum = np.zeros(N, dtype=np.float64)
    edge_count = np.zeros(N, dtype=np.int64)

    num_batches = (N + batch_size - 1) // batch_size
    iterator = range(num_batches)
    if verbose:
        iterator = tqdm(iterator, desc="Computing edge scores")

    for batch_idx in iterator:
        start_i = batch_idx * batch_size
        end_i = min((batch_idx + 1) * batch_size, N)

        # 计算该批与所有样本的距离
        batch_sim = X[start_i:end_i] @ X.T  # [batch, N]
        np.clip(batch_sim, -1.0, 1.0, out=batch_sim)
        batch_dist = 1.0 - batch_sim  # 余弦距离

        # 统计每个节点的边权和与边数
        for local_i, global_i in enumerate(range(start_i, end_i)):
            # 只计算上三角部分（global_i < j）
            for j in range(global_i + 1, N):
                weight = float(batch_dist[local_i, j])
                edge_weight_sum[global_i] += weight
                edge_count[global_i] += 1
                # 由于是对称的，j 节点也需要统计这条边
                edge_weight_sum[j] += weight
                edge_count[j] += 1

    # 计算平均边权
    scores = {}
    for i in range(N):
        if edge_count[i] > 0:
            scores[int(indices[i])] = float(edge_weight_sum[i] / edge_count[i])
        else:
            scores[int(indices[i])] = 0.0

    t1 = time.perf_counter()
    if verbose:
        print(f"Edge scores computed in {format_duration(t1 - t0)}")

    return scores


def build_full_graph_optimized(
    indices: np.ndarray,
    labels: np.ndarray,
    embeddings: np.ndarray,
    batch_size: int = 5000,
    save_edges: bool = False,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    构建全量图（优化版本：不存储完整邻接矩阵）

    参数:
        indices: 样本索引 [N]
        labels: 样本标签 [N]
        embeddings: 特征矩阵 [N, D]
        batch_size: 分批计算大小
        save_edges: 是否保存边列表（通常不需要）
        verbose: 是否显示进度

    返回:
        graph_data: 图数据字典，包含 nodes, edge_scores, num_nodes 等
    """
    N = len(indices)
    D = embeddings.shape[1]

    if verbose:
        print(f"Building optimized full graph for {N} samples...")
        print(f"NOT storing full adjacency matrix - saving memory and time")

    t_total = time.perf_counter()

    # 1. 归一化
    t0 = time.perf_counter()
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    X = embeddings / norms

    feature_mean = X.mean(axis=0)
    within_var = float(((X - feature_mean) ** 2).mean())

    if verbose:
        t1 = time.perf_counter()
        print(f"Step 1: Normalization done in {format_duration(t1 - t0)}")

    # 2. 构建节点列表
    t0 = time.perf_counter()
    nodes = []
    for i in range(N):
        nodes.append({
            "idx": int(indices[i]),
            "label": int(labels[i]),
        })

    if verbose:
        t1 = time.perf_counter()
        print(f"Step 2: Built {N} nodes in {format_duration(t1 - t0)}")

    # 3. 计算边权（使用累积方式，不构建完整矩阵）
    t0 = time.perf_counter()
    edge_weight_sum = np.zeros(N, dtype=np.float64)
    edge_count = np.zeros(N, dtype=np.int64)

    # 如果需要保存边列表
    edges = [] if save_edges else None
    all_weights = []

    num_batches = (N + batch_size - 1) // batch_size
    iterator = range(num_batches)
    if verbose:
        iterator = tqdm(iterator, desc="Step 3: Computing edge weights")

    for batch_idx in iterator:
        start_i = batch_idx * batch_size
        end_i = min((batch_idx + 1) * batch_size, N)

        # 计算该批与所有样本的距离
        batch_sim = X[start_i:end_i] @ X.T  # [batch, N]
        np.clip(batch_sim, -1.0, 1.0, out=batch_sim)
        batch_dist = 1.0 - batch_sim

        # 统计每个节点的边权和与边数
        for local_i, global_i in enumerate(range(start_i, end_i)):
            for j in range(global_i + 1, N):  # 只处理上三角
                weight = float(batch_dist[local_i, j])

                # 累积边权
                edge_weight_sum[global_i] += weight
                edge_count[global_i] += 1
                edge_weight_sum[j] += weight
                edge_count[j] += 1

                # 可选：保存边列表
                if save_edges:
                    edges.append((global_i, j, weight))
                    all_weights.append(weight)

    if verbose:
        t1 = time.perf_counter()
        print(f"Step 3: Edge weights computed in {format_duration(t1 - t0)}")
        total_edges = N * (N - 1) // 2
        print(f"Total edges (in full graph): {total_edges}")

    # 4. 计算每个节点的平均边权
    t0 = time.perf_counter()
    edge_scores = {}
    for i in range(N):
        if edge_count[i] > 0:
            edge_scores[int(indices[i])] = float(edge_weight_sum[i] / edge_count[i])
        else:
            edge_scores[int(indices[i])] = 0.0

    if verbose:
        t1 = time.perf_counter()
        print(f"Step 4: Edge scores computed in {format_duration(t1 - t0)}")

    # 5. 构建返回字典（不包含 adj_matrix）
    graph_data = {
        "nodes": nodes,
        "num_nodes": N,
        "within_cluster_feature_variance": within_var,
        "edge_scores": edge_scores,  # 预计算好的 edge scores
    }

    # 可选：保存边列表（通常不需要，会增加文件大小）
    if save_edges:
        graph_data["edges"] = edges
        graph_data["cosine_distance_matrix_upper_triangle"] = np.array(all_weights, dtype=np.float32)

    # 注意：不存储 adj_matrix

    t_end = time.perf_counter()
    if verbose:
        print(f"\nTotal graph building time: {format_duration(t_end - t_total)}")

    return graph_data


def main(
    input_path: str,
    output_path: str,
    batch_size: int = 5000,
    save_edges: bool = False,
):
    """
    主函数

    参数:
        input_path: embedding .pt 文件路径
        output_path: 输出 graph.pkl 文件路径
        batch_size: 分批计算大小
        save_edges: 是否保存边列表（会增加文件大小）
    """
    print("=" * 60)
    print("CIFAR100 Full Graph Building (Optimized - No Adj Matrix)")
    print("=" * 60)

    t_total = time.perf_counter()

    # 1. 加载 embedding
    t0 = time.perf_counter()
    indices, labels, embeddings = load_embeddings(input_path)
    t1 = time.perf_counter()
    print(f"[Time] Loading embeddings: {format_duration(t1 - t0)}")

    # 2. 构建全量图（优化版本）
    t0 = time.perf_counter()
    graph_data = build_full_graph_optimized(
        indices=indices,
        labels=labels,
        embeddings=embeddings,
        batch_size=batch_size,
        save_edges=save_edges,
        verbose=True
    )
    t1 = time.perf_counter()
    print(f"[Time] Building graph: {format_duration(t1 - t0)}")

    # 3. 保存
    t0 = time.perf_counter()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving graph to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(graph_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    t1 = time.perf_counter()
    print(f"[Time] Saving graph: {format_duration(t1 - t0)}")

    # 打印统计信息
    print("\n===== Summary =====")
    print(f"Total nodes: {graph_data['num_nodes']}")
    print(f"Edge scores computed: {len(graph_data['edge_scores'])}")
    print(f"Output file: {output_path}")

    # 检查文件大小
    file_size = output_path.stat().st_size / 1024**2  # MB
    print(f"File size: {file_size:.2f} MB")

    t_end = time.perf_counter()
    print(f"\n[Time] TOTAL: {format_duration(t_end - t_total)}")
    print("\n🎉 Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIFAR100 全量图构建（优化版，不存储邻接矩阵）")
    parser.add_argument("--input", type=str, required=True, help="embedding .pt 文件路径")
    parser.add_argument("--output", type=str, required=True, help="输出 graph.pkl 文件路径")
    parser.add_argument("--batch-size", type=int, default=5000, help="分批计算大小")
    parser.add_argument("--save-edges", action="store_true", help="保存边列表（会增加文件大小）")

    args = parser.parse_args()

    main(
        input_path=args.input,
        output_path=args.output,
        batch_size=args.batch_size,
        save_edges=args.save_edges,
    )