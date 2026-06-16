import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import faiss
from tqdm import tqdm
import time

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Tuple
from typing import List, Optional
from multiprocessing import Pool, current_process

# ========================================
# 工具函数：读取目录下所有 .pkl 文件
# ========================================

def process_one_class(
    class_id: int,
    df_class: pd.DataFrame,
    feature_column: str,
    n_clusters_per_class: int,
    max_samples_per_class: int,
    kmeans_iter: int,
    use_gpu: bool,
    output_base_dir: str,
    max_rows_per_file: int,
    input_pkl_dir: str,
):
    """
    子进程执行：对单个类聚类并保存结果
    注意：df_class 会被 pickle 传入；建议只传必要列且不太大。
    """
    try:
        N = len(df_class)
        if max_samples_per_class and N > max_samples_per_class:
            df_class = df_class.sample(n=max_samples_per_class, random_state=42)
            N = len(df_class)

        if N < n_clusters_per_class:
            return (class_id, False, f"skip: N={N} < k={n_clusters_per_class}")

        features = np.stack(df_class[feature_column].values)
        if features.ndim != 2:
            return (class_id, False, f"bad feature shape: {features.shape}")

        centroids, labels = faiss_kmeans(
            data=features,
            k=n_clusters_per_class,
            niter=kmeans_iter,
            gpu=use_gpu,
            seed=42
        )

        save_per_class_clustering_result(
            df_class=df_class,
            class_id=class_id,
            labels=labels,
            centroids=centroids,
            output_base_dir=output_base_dir,
            max_rows_per_file=max_rows_per_file,
            extra_metadata={
                "source": input_pkl_dir,
                "feature_column": feature_column,
                "n_clusters": n_clusters_per_class,
                "total_samples": len(df_class),
                "timestamp": pd.Timestamp.now().isoformat()
            }
        )
        return (class_id, True, "ok")

    except Exception as e:
        return (class_id, False, f"error: {e}")

def read_all_pkl_files(folder_path: str, columns: List[str] = None, verbose: bool = True) -> pd.DataFrame:
    """
    读取文件夹中所有 .pkl 文件，合并为一个 DataFrame

    参数:
        folder_path: 包含 .pkl 的目录
        columns: 可选，只保留某些列（减少内存）
        verbose: 是否打印信息

    返回:
        pd.DataFrame
    """
    folder_path = Path(folder_path)
    if not folder_path.exists():
        raise FileNotFoundError(f"路径不存在: {folder_path}")

    pkl_files = sorted([f for f in folder_path.glob("*.pkl") if f.is_file()])
    if not pkl_files:
        raise FileNotFoundError(f"未找到任何 .pkl 文件: {folder_path}")

    dfs = []
    for file in tqdm(pkl_files, desc="Loading PKL files"):
        try:
            with open(file, 'rb') as f:
                data = pickle.load(f)
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, pd.DataFrame):
                df = data
            else:
                # 尝试从 dict 中取
                try:
                    df = pd.DataFrame(data)
                except Exception:
                    continue

            if columns:
                df = df[[c for c in columns if c in df.columns]]
            dfs.append(df)
        except Exception as e:
            print(f"❌ 无法读取 {file}: {e}")

    df_all = pd.concat(dfs, ignore_index=True)
    if verbose:
        print(f"✅ 成功加载 {len(df_all)} 条数据，来自 {len(pkl_files)} 个文件")
    return df_all


# # 子进程读单个文件：必须放在模块顶层
# def _load_one_pkl_process(file_str: str, columns=None):
#     file = Path(file_str)
#     with open(file, "rb") as f:
#         data = pickle.load(f)

#     if isinstance(data, list):
#         df = pd.DataFrame(data)
#     elif isinstance(data, pd.DataFrame):
#         df = data
#     else:
#         return None

#     if columns:
#         keep = [c for c in columns if c in df.columns]
#         df = df[keep]
#     return df


# def read_all_pkl_files(folder_path, columns=None, verbose=True, max_workers=None):
#     """
#     多进程读取文件夹中所有 .pkl 文件，合并为一个 DataFrame
#     """
#     folder_path = Path(folder_path)
#     if not folder_path.exists():
#         raise FileNotFoundError(f"路径不存在: {folder_path}")

#     pkl_files = sorted([f for f in folder_path.glob("*.pkl") if f.is_file()])
#     if not pkl_files:
#         raise FileNotFoundError(f"未找到任何 .pkl 文件: {folder_path}")

#     # 默认进程数：不要太大，避免把共享存储打崩 + 内存爆
#     if max_workers is None:
#         max_workers = min(8, len(pkl_files))  # 你也可以改成 os.cpu_count()
#     else:
#         max_workers = min(max_workers, len(pkl_files))

#     dfs = []
#     errors = 0

#     with ProcessPoolExecutor(max_workers=max_workers) as ex:
#         futures = {
#             ex.submit(_load_one_pkl_process, str(f), columns): f
#             for f in pkl_files
#         }

#         for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Loading PKL files (mp x{max_workers})"):
#             f = futures[fut]
#             try:
#                 df = fut.result()
#                 if df is not None:
#                     dfs.append(df)
#             except Exception as e:
#                 errors += 1
#                 print(f"❌ 无法读取 {f}: {e}")

#     if not dfs:
#         raise RuntimeError("所有 pkl 都加载失败，dfs 为空")

#     df_all = pd.concat(dfs, ignore_index=True)
#     if verbose:
#         print(f"✅ 成功加载 {len(df_all)} 条数据，来自 {len(pkl_files)} 个文件（失败 {errors} 个），workers={max_workers}")
#     return df_all


# ========================================
# Faiss 聚类函数（类内使用）
# ========================================
def faiss_kmeans(data, k, niter=20, gpu=False, seed=42):
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


# ========================================
# 保存类内聚类结果
# ========================================
def save_per_class_clustering_result(
    df_class: pd.DataFrame,
    class_id: int,
    labels: np.ndarray,
    centroids: np.ndarray,
    output_base_dir: str,
    max_rows_per_file: int = 100_000,
    extra_metadata: dict = None
):
    """
    保存单个类的聚类结果：
        output_base_dir/
          └── class_{id}/
               ├── centroids.npy
               ├── cluster_mapping.json
               └── clusters/
                    ├── cluster_0000_00000.parquet
                    └── ...
    """
    output_base_dir = Path(output_base_dir)
    class_dir = output_base_dir / f"class_{class_id}"
    class_dir.mkdir(parents=True, exist_ok=True)

    clusters_dir = class_dir / "clusters"
    clusters_dir.mkdir(exist_ok=True)

    # 添加聚类标签
    df_with_label = df_class.reset_index(drop=True).copy()
    df_with_label["cluster_label"] = labels

    # 分块保存 cluster 数据
    grouped = df_with_label.groupby("cluster_label")
    print(f"📦 Class {class_id}: 共 {len(grouped)} 个子簇")

    for label, group in grouped:
        chunk_size = max_rows_per_file
        num_chunks = (len(group) // chunk_size) + 1
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(group))
            chunk = group.iloc[start_idx:end_idx].reset_index(drop=True)
            filename = f"cluster_{label:04d}_{i:05d}.parquet"
            chunk.to_parquet(clusters_dir / filename, index=False)

    # 保存聚类中心
    np.save(class_dir / "centroids.npy", centroids)

    # 保存映射信息
    unique, counts = np.unique(labels, return_counts=True)
    mapping = {
        "class_id": int(class_id),
        "num_clusters": len(unique),
        "centroid_shape": centroids.shape,
        "cluster_distribution": dict(zip(unique.tolist(), counts.tolist())),
        "metadata": extra_metadata or {}
    }
    with open(class_dir / "cluster_mapping.json", 'w', encoding='utf-8') as f:
        import json
        json.dump(mapping, f, indent=2, ensure_ascii=False)

    print(f"✅ 类 {class_id} 聚类结果已保存至: {class_dir}")


# ========================================
# 主函数
# ========================================
def main(
    input_pkl_dir: str,
    output_base_dir: str,
    feature_column: str = "feature",
    target_classes: list = None,  # 如果只想处理某些类
    n_clusters_per_class: int = 10,  # 每个类聚成几类
    max_samples_per_class: int = None,  # 可选：限制每类最多用多少样本
    kmeans_iter: int = 20,
    use_gpu: bool = False,
    max_rows_per_file: int = 100_000
):
    print("🚀 开始按类聚类任务...")

    # 1. 加载所有 pkl
    df_all = read_all_pkl_files(
        folder_path=input_pkl_dir,
        columns=['image_name', 'true_class', 'pred_class', 'correct', 'loss', 'logits', 'feature'],
        verbose=True,
    )

    # 2. 过滤目标类
    if target_classes is not None:
        df_all = df_all[df_all['true_class'].isin(target_classes)]
        print(f"筛选后剩余 {len(df_all)} 条数据")

    # 3. 按 true_class 分组
    grouped = df_all.groupby('true_class')
    print(f"🔍 共发现 {len(grouped)} 个类别")

    # 4. 并行遍历每个类进行聚类
    groups = list(grouped)  # [(class_id, df_class), ...]
    print(f"🧵 准备并行处理 {len(groups)} 个类别")

    # 建议：只保留需要列，减少进程间 pickle 成本
    needed_cols = ['image_name', 'true_class', 'pred_class', 'correct', 'loss', 'logits', feature_column]
    groups = [(cid, dfc[needed_cols].copy()) for cid, dfc in groups]

    # 进程数：按机器核数/任务数调；太大反而会抢内存
    max_workers = min(os.cpu_count() or 1, len(groups), 8)
    print(f"⚙️ Process workers = {max_workers}")

    futures = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for class_id, df_class in tqdm(groups, desc="submitting jobs"):
            futures.append(ex.submit(
                process_one_class,
                class_id,
                df_class,
                feature_column,
                n_clusters_per_class,
                max_samples_per_class,
                kmeans_iter,
                use_gpu,
                output_base_dir,
                max_rows_per_file,
                input_pkl_dir,
            ))

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing classes (parallel)"):
            class_id, ok, msg = fut.result()
            if not ok:
                print(f"❌ class {class_id}: {msg}")
            else:
                print(f"✅ class {class_id}: {msg}")

def format_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

# ========================================
# 示例调用
# ========================================
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="directory containing pkl files from inference")
    ap.add_argument("--output", required=True, help="output directory for cluster results")
    cli_args = ap.parse_args()

    input_pkl_dir = cli_args.input
    output_base_dir = cli_args.output

    t0 = time.perf_counter()
    main(
        input_pkl_dir=input_pkl_dir,
        output_base_dir=output_base_dir,
        feature_column="feature",           # 你 pkl 中存 fc 输入的字段名
        target_classes=None,                # 设为 [0, 1, 2] 只处理前3类；None 表示全部
        n_clusters_per_class=25,            # 每类聚成 25 个子簇（可调整）
        max_samples_per_class=5000,         # 每类最多用 5000 个样本（防内存爆炸）
        kmeans_iter=25,
        use_gpu=False,                       # 若有 GPU 加速建议开启
        max_rows_per_file=10000             # 每个 parquet 最多 1w 行
    )
    t1 = time.perf_counter()
    print(f"Total inference time: {format_duration(t1 - t0)}")
