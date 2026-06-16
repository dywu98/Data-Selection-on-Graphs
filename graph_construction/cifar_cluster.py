"""
CIFAR100 聚类脚本
基于 embedding 文件进行 KMeans 聚类，适配 CIFAR100 的索引格式

输入: torch.save 的 .pt 文件，包含 indices, labels, embeddings
输出: class_{id}/clusters/cluster_*.parquet 文件

用法:
    python cifar_faiss_cluster.py \
        --input /path/to/train_embeddings.pt \
        --output /path/to/output_cluster_dir \
        --n-clusters 10 \
        --feature-dim 512
"""

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import faiss
from tqdm import tqdm
import time
import argparse
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional


def format_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def load_embeddings(embedding_path: str, verbose: bool = True) -> pd.DataFrame:
    """
    加载 CIFAR100 embedding 文件并转为 DataFrame

    参数:
        embedding_path: .pt 文件路径
        verbose: 是否打印信息

    返回:
        pd.DataFrame，包含 index, label, feature 列
    """
    if verbose:
        print(f"Loading embeddings from {embedding_path}...")

    data = torch.load(embedding_path, map_location='cpu')

    indices = data['indices'].numpy() if isinstance(data['indices'], torch.Tensor) else data['indices']
    labels = data['labels'].numpy() if isinstance(data['labels'], torch.Tensor) else data['labels']
    embeddings = data['embeddings'].numpy() if isinstance(data['embeddings'], torch.Tensor) else data['embeddings']

    if verbose:
        print(f"Loaded {len(indices)} samples, embedding dim: {embeddings.shape[1]}")
        print(f"Number of classes: {len(np.unique(labels))}")

    # 转为 list 存储（方便后续处理）
    df = pd.DataFrame({
        'index': indices,
        'label': labels,
        'feature': list(embeddings)  # 每行一个 numpy array
    })

    return df


def faiss_kmeans(data: np.ndarray, k: int, niter: int = 20, gpu: bool = False, seed: int = 42):
    """
    对输入数据进行 KMeans 聚类
    """
    data = data.astype(np.float32)
    n_samples, dim = data.shape

    kmeans = faiss.Kmeans(
        d=dim,
        k=k,
        niter=niter,
        nredo=1,
        spherical=False,
        verbose=True,
        gpu=gpu,
        seed=seed
    )
    kmeans.train(data)

    centroids = kmeans.centroids.copy()

    index = faiss.IndexFlatL2(dim)
    if gpu:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    index.add(centroids)
    _, labels = index.search(data, 1)
    labels = labels.reshape(-1)

    return centroids, labels


def process_one_class(
    class_id: int,
    df_class: pd.DataFrame,
    n_clusters: int,
    kmeans_iter: int,
    use_gpu: bool,
    output_base_dir: str,
    max_rows_per_file: int,
):
    """
    对单个类进行聚类并保存结果
    """
    try:
        N = len(df_class)

        if N < n_clusters:
            return (class_id, False, f"skip: N={N} < k={n_clusters}")

        # 提取特征
        features = np.stack(df_class['feature'].values).astype(np.float32)

        # KMeans 聚类
        centroids, labels = faiss_kmeans(
            data=features,
            k=n_clusters,
            niter=kmeans_iter,
            gpu=use_gpu,
            seed=42
        )

        # 保存结果
        output_base_dir = Path(output_base_dir)
        class_dir = output_base_dir / f"class_{class_id}"
        class_dir.mkdir(parents=True, exist_ok=True)
        clusters_dir = class_dir / "clusters"
        clusters_dir.mkdir(exist_ok=True)

        # 添加聚类标签
        df_with_label = df_class.reset_index(drop=True).copy()
        df_with_label["cluster_label"] = labels

        # 分块保存每个 cluster
        grouped = df_with_label.groupby("cluster_label")

        for label, group in grouped:
            chunk_size = max_rows_per_file
            num_chunks = (len(group) // chunk_size) + 1
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, len(group))
                chunk = group.iloc[start_idx:end_idx].reset_index(drop=True)

                # 将 feature 转为 list 以便保存
                chunk_to_save = chunk.copy()
                chunk_to_save['feature'] = chunk_to_save['feature'].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

                filename = f"cluster_{label:04d}_{i:05d}.parquet"
                chunk_to_save.to_parquet(clusters_dir / filename, index=False)

        # 保存聚类中心
        np.save(class_dir / "centroids.npy", centroids)

        # 保存映射信息
        unique, counts = np.unique(labels, return_counts=True)
        mapping = {
            "class_id": int(class_id),
            "num_clusters": len(unique),
            "centroid_shape": centroids.shape,
            "cluster_distribution": dict(zip(unique.tolist(), counts.tolist())),
        }
        with open(class_dir / "cluster_mapping.json", 'w', encoding='utf-8') as f:
            import json
            json.dump(mapping, f, indent=2, ensure_ascii=False)

        return (class_id, True, f"ok, {len(grouped)} clusters created")

    except Exception as e:
        import traceback
        return (class_id, False, f"error: {e}\n{traceback.format_exc()}")


def main(
    input_path: str,
    output_base_dir: str,
    n_clusters_per_class: int = 10,
    kmeans_iter: int = 20,
    use_gpu: bool = False,
    max_rows_per_file: int = 10000,
    target_classes: Optional[List[int]] = None,
):
    """
    主函数：对 CIFAR100 embedding 进行聚类

    参数:
        input_path: embedding .pt 文件路径
        output_base_dir: 输出目录
        n_clusters_per_class: 每个类聚成几个簇
        kmeans_iter: KMeans 迭代次数
        use_gpu: 是否使用 GPU
        max_rows_per_file: 每个 parquet 文件最大行数
        target_classes: 只处理指定的类（None 表示全部）
    """
    print("=" * 60)
    print("CIFAR100 Clustering Task")
    print("=" * 60)

    # 1. 加载 embedding
    df_all = load_embeddings(input_path, verbose=True)

    # 2. 过滤目标类（如果指定）
    if target_classes is not None:
        df_all = df_all[df_all['label'].isin(target_classes)]
        print(f"筛选后剩余 {len(df_all)} 条数据")

    # 3. 按 label 分组
    grouped = df_all.groupby('label')
    print(f"共发现 {len(grouped)} 个类别")

    # 4. 并行处理每个类
    groups = list(grouped)  # [(label, df_class), ...]

    # 只保留需要的列
    needed_cols = ['index', 'label', 'feature']
    groups = [(label, dfc[needed_cols].copy()) for label, dfc in groups]

    max_workers = min(os.cpu_count() or 1, len(groups), 32)
    print(f"使用 {max_workers} 个进程并行处理")

    futures = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for class_id, df_class in tqdm(groups, desc="Submitting jobs"):
            futures.append(ex.submit(
                process_one_class,
                class_id,
                df_class,
                n_clusters_per_class,
                kmeans_iter,
                use_gpu,
                output_base_dir,
                max_rows_per_file,
            ))

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing classes"):
            class_id, ok, msg = fut.result()
            if not ok:
                print(f"❌ class {class_id}: {msg}")
            else:
                print(f"✅ class {class_id}: {msg}")

    print(f"\n🎉 聚类完成！结果保存在: {output_base_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIFAR100 聚类脚本")
    parser.add_argument("--input", type=str, required=True, help="embedding .pt 文件路径")
    parser.add_argument("--output", type=str, required=True, help="输出目录")
    parser.add_argument("--n-clusters", type=int, default=10, help="每个类聚成几个簇")
    parser.add_argument("--kmeans-iter", type=int, default=20, help="KMeans 迭代次数")
    parser.add_argument("--gpu", action="store_true", help="是否使用 GPU")
    parser.add_argument("--max-rows", type=int, default=10000, help="每个 parquet 文件最大行数")
    parser.add_argument("--target-classes", type=int, nargs='+', default=None, help="只处理指定的类")

    args = parser.parse_args()

    t0 = time.perf_counter()
    main(
        input_path=args.input,
        output_base_dir=args.output,
        n_clusters_per_class=args.n_clusters,
        kmeans_iter=args.kmeans_iter,
        use_gpu=args.gpu,
        max_rows_per_file=args.max_rows,
        target_classes=args.target_classes,
    )
    t1 = time.perf_counter()
    print(f"Total time: {format_duration(t1 - t0)}")