"""
CIFAR100 图构建脚本
基于聚类结果构建全连接图，使用索引代替文件名

输入: 聚类结果目录 (class_{id}/clusters/cluster_*.parquet)
输出: 图文件 (class_{id}/cluster_*.graph.pkl)

用法:
    python cifar_build_graph.py \
        --input /path/to/train_cluster/ \
        --output /path/to/train_graph/ \
        --workers 4
"""

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import csr_matrix
import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from collections import defaultdict
from typing import List, Dict, Any


def format_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def build_complete_graph_with_cosine_weights(
    df_cluster: pd.DataFrame,
    feature_col: str = "feature"
) -> Dict[str, Any]:
    """
    为单个 cluster 构建全连接图

    参数:
        df_cluster: 包含该 cluster 所有样本的 DataFrame
        feature_col: 特征列名

    返回:
        图数据字典，包含 nodes, edges, adj_matrix 等
    """
    N = len(df_cluster)
    if N == 0:
        raise ValueError("Empty cluster")

    # 提取特征
    features = np.stack(df_cluster[feature_col].values).astype(np.float32, copy=False)

    # 归一化
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    X = features / norms

    # 计算余弦相似度矩阵
    cos_sim = X @ X.T
    np.clip(cos_sim, -1.0, 1.0, out=cos_sim)
    cos_dist = 1.0 - cos_sim

    # 构建节点列表（使用索引作为标识符）
    nodes = []
    for idx, row in enumerate(df_cluster.itertuples(index=False)):
        nodes.append({
            "idx": int(getattr(row, "index")),  # 原始 CIFAR100 数据集索引
            "label": int(getattr(row, "label")),
        })

    # 构建边列表（上三角）
    iu, ju = np.triu_indices(N, k=1)
    w = cos_dist[iu, ju].astype(np.float32, copy=False)
    edges = list(zip(iu.tolist(), ju.tolist(), w.tolist()))

    # 构建邻接矩阵
    row = np.concatenate([iu, ju])
    col = np.concatenate([ju, iu])
    data = np.concatenate([w, w])
    adj_matrix = csr_matrix((data, (row, col)), shape=(N, N))

    # 统计信息
    feature_mean = features.mean(axis=0)
    within_var = float(((features - feature_mean) ** 2).mean())

    return {
        "nodes": nodes,
        "edges": edges,
        "adj_matrix": adj_matrix,
        "num_nodes": N,
        "within_cluster_feature_variance": within_var,
        "cosine_distance_matrix_upper_triangle": w,
    }


def load_cluster_data(class_dir: Path) -> Dict[int, pd.DataFrame]:
    """
    加载一个类别的所有 cluster 数据
    """
    clusters_dir = class_dir / "clusters"
    if not clusters_dir.exists():
        return {}

    parquet_files = sorted(clusters_dir.glob("cluster_*.parquet"))
    clusters = {}

    for file in parquet_files:
        try:
            stem = file.stem
            cluster_id = int(stem.split("_")[1])
            df = pd.read_parquet(file)

            # 将 feature 从 list 转回 numpy array
            if 'feature' in df.columns:
                df['feature'] = df['feature'].apply(lambda x: np.array(x, dtype=np.float32) if isinstance(x, list) else x)

            clusters.setdefault(cluster_id, []).append(df)
        except Exception as e:
            print(f"⚠️ 无法读取 {file}: {e}")

    # 合并同一 cluster 的多个文件
    for cid in list(clusters.keys()):
        clusters[cid] = pd.concat(clusters[cid], ignore_index=True)

    return clusters


def save_graph(graph_data: Dict, save_path: Path):
    """保存图数据为 pickle 文件"""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(graph_data, f, protocol=pickle.HIGHEST_PROTOCOL)


def _process_one_cluster(args):
    """子进程：构图 + 保存"""
    class_id, cluster_id, df_cluster, feature_col, output_root = args
    t0 = time.perf_counter()

    graph_data = build_complete_graph_with_cosine_weights(df_cluster, feature_col=feature_col)

    save_dir = Path(output_root) / f"class_{class_id}"
    save_path = save_dir / f"cluster_{cluster_id:04d}.graph.pkl"
    save_graph(graph_data, save_path)

    t1 = time.perf_counter()
    return class_id, cluster_id, graph_data["num_nodes"], (t1 - t0)


def main(
    input_clustering_root: str,
    output_graph_root: str,
    feature_col: str = "feature",
    max_workers: int = None,
    print_slowest: int = 10
):
    """
    主函数：基于聚类结果构建图

    参数:
        input_clustering_root: 聚类结果根目录
        output_graph_root: 图输出根目录
        feature_col: 特征列名
        max_workers: 最大并行进程数
        print_slowest: 打印最慢的 N 个 cluster
    """
    print("=" * 60)
    print("CIFAR100 Graph Building Task")
    print("=" * 60)

    input_root = Path(input_clustering_root)
    output_root = Path(output_graph_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # 扫描所有 class 目录
    class_dirs = [d for d in input_root.iterdir() if d.is_dir() and d.name.startswith("class_")]
    if not class_dirs:
        raise FileNotFoundError(f"未找到以 'class_' 开头的子目录: {input_root}")

    # 构建任务列表
    tasks = []
    class_task_count = {}
    scan_t0 = time.perf_counter()

    for class_dir in tqdm(sorted(class_dirs), desc="Loading cluster data"):
        class_id = int(class_dir.name.split("_")[1])
        clusters = load_cluster_data(class_dir)
        if not clusters:
            continue
        class_task_count[class_id] = len(clusters)
        for cluster_id, df_cluster in clusters.items():
            tasks.append((class_id, cluster_id, df_cluster, feature_col, str(output_root)))

    scan_t1 = time.perf_counter()
    if not tasks:
        print("没有任何 cluster 任务")
        return

    if max_workers is None:
        max_workers = min(4, os.cpu_count() or 1)

    print(f"🔍 Scan done: classes={len(class_task_count)}, clusters={len(tasks)} "
          f"(scan time {format_duration(scan_t1 - scan_t0)}), workers={max_workers}")

    # 并行执行
    run_t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_process_one_cluster, t) for t in tasks]

        pbar = tqdm(total=len(futures), desc="Building graphs", dynamic_ncols=True)
        for fut in as_completed(futures):
            class_id, cluster_id, n, elapsed = fut.result()
            pbar.update(1)
            pbar.set_postfix_str(f"last=class{class_id} c{cluster_id:04d} N={n}")
        pbar.close()
    run_t1 = time.perf_counter()

    print("\n===== Summary =====")
    print(f"Total time: {format_duration(run_t1 - run_t0)}")
    print(f"Total clusters: {len(tasks)}")
    print(f"Avg time/cluster: {(run_t1 - run_t0) / len(tasks):.3f}s")
    print(f"\n🎉 Done! Graphs saved to: {output_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIFAR100 图构建脚本")
    parser.add_argument("--input", type=str, required=True, help="聚类结果根目录")
    parser.add_argument("--output", type=str, required=True, help="图输出根目录")
    parser.add_argument("--feature-col", type=str, default="feature", help="特征列名")
    parser.add_argument("--workers", type=int, default=None, help="并行进程数")
    parser.add_argument("--print-slowest", type=int, default=10, help="打印最慢的 N 个 cluster")

    args = parser.parse_args()

    t0 = time.perf_counter()
    main(
        input_clustering_root=args.input,
        output_graph_root=args.output,
        feature_col=args.feature_col,
        max_workers=args.workers,
        print_slowest=args.print_slowest,
    )
    t1 = time.perf_counter()
    print(f"\nProgram total time: {format_duration(t1 - t0)}")