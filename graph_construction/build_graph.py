import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import argparse
import time



def build_complete_graph_with_cosine_weights(df_cluster: pd.DataFrame, feature_col: str = 'feature'):
    """
    对一个 cluster 的 DataFrame 构建全连接图

    参数:
        df_cluster: 包含同一 cluster 内样本的 DataFrame，至少有 'loss', 'image_name', feature_col
        feature_col: 特征列名，默认 'feature'

    返回:
        graph_dict: 包含节点、边、邻接矩阵等信息
    """
    N = len(df_cluster)
    if N == 0:
        raise ValueError("Empty cluster")

    # 提取特征并堆叠
    try:
        features = np.stack(df_cluster[feature_col].values)  # shape: (N, D)
    except Exception as e:
        print(f"❌ 特征堆叠失败: {e}")
        raise

    # 计算余弦相似度矩阵
    cos_sim = cosine_similarity(features)  # (N, N)
    cos_dist = 1 - cos_sim  # 余弦距离作为边权重

    # 构建节点列表
    nodes = []
    for idx, (_, row) in enumerate(df_cluster.iterrows()):
        nodes.append({
            'idx': idx,
            'image_name': row.get('image_name', f'unknown_{idx}'),
            'loss': float(row['loss']),
            'true_class': row.get('true_class'),
            'pred_class': row.get('pred_class'),
            'correct': bool(row.get('correct')),
        })

    # 构建边列表（只保留上三角，避免重复）
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            weight = float(cos_dist[i, j])
            edges.append((i, j, weight))

    # 邻接矩阵（稀疏存储）
    row = [e[0] for e in edges]
    col = [e[1] for e in edges]
    data = [e[2] for e in edges]
    adj_matrix = csr_matrix((data + data, (row + col, col + row)), shape=(N, N))  # 对称化

    # 统计信息
    avg_loss = df_cluster['loss'].mean()
    feature_mean = features.mean(axis=0)
    within_var = ((features - feature_mean) ** 2).mean()

    return {
        'nodes': nodes,
        'edges': edges,
        'adj_matrix': adj_matrix,
        'num_nodes': N,
        'avg_loss': avg_loss,
        'within_cluster_feature_variance': float(within_var),
        'cosine_distance_matrix_upper_triangle': cos_dist[np.triu_indices(N, k=1)],  # 可选存统计
    }


def load_cluster_data(class_dir: Path):
    """
    加载某个类目录下所有 cluster parquet 文件
    返回: dict[cluster_id, DataFrame]
    """
    clusters_dir = class_dir / "clusters"
    if not clusters_dir.exists():
        return {}

    parquet_files = sorted(clusters_dir.glob("cluster_*.parquet"))
    clusters = {}

    for file in parquet_files:
        try:
            # 解析 cluster ID
            stem = file.stem  # e.g., "cluster_0003_00000"
            cluster_id = int(stem.split('_')[1])  # 第二个下划线前是 cluster label
            df = pd.read_parquet(file)

            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(df)

        except Exception as e:
            print(f"⚠️ 无法读取 {file}: {e}")

    # 合并每个 cluster 的多个 chunk
    for cid in clusters:
        clusters[cid] = pd.concat(clusters[cid], ignore_index=True)

    return clusters


def save_graph(graph_data, save_path: Path):
    """安全保存图数据"""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(graph_data, f)
    print(f"✅ 图已保存: {save_path} (节点数: {graph_data['num_nodes']})")


def main(input_clustering_root: str, output_graph_root: str, feature_col: str = 'feature'):
    input_root = Path(input_clustering_root)
    output_root = Path(output_graph_root)
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"🔍 扫描聚类结果目录: {input_root}")

    class_dirs = [d for d in input_root.iterdir() if d.is_dir() and d.name.startswith("class_")]

    if not class_dirs:
        raise FileNotFoundError(f"未找到以 'class_' 开头的子目录: {input_root}")

    for class_dir in sorted(class_dirs):
        class_id = class_dir.name.split("_")[1]
        print(f"\n📌 处理类别: {class_id}")

        # Step 1: 加载该类的所有 cluster 数据
        clusters = load_cluster_data(class_dir)
        if not clusters:
            print(f"   ⚠️ 未加载到任何 cluster 数据")
            continue

        # Step 2: 为每个 cluster 构建图
        for cluster_id, df_cluster in clusters.items():
            print(f"   🛠️  构建 cluster {cluster_id} 图 (样本数: {len(df_cluster)})")
            try:
                graph_data = build_complete_graph_with_cosine_weights(df_cluster, feature_col=feature_col)

                # 保存路径: output_root/class_{id}/cluster_{id}.graph.pkl
                save_dir = output_root / f"class_{class_id}"
                save_path = save_dir / f"cluster_{cluster_id:04d}.graph.pkl"

                save_graph(graph_data, save_path)

            except Exception as e:
                print(f"   ❌ 构建 cluster {cluster_id} 失败: {e}")
                continue

    print(f"\n🎉 所有图构建完成！结果保存至: {output_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="基于聚类结果构建全连接图")
    parser.add_argument("--input", type=str, required=True, help="聚类结果根目录（包含 class_* 文件夹）")
    parser.add_argument("--output", type=str, required=True, help="图输出根目录")
    parser.add_argument("--feature_col", type=str, default="feature", help="特征列名（默认: 'feature')")

    args = parser.parse_args()
    t0 = time.perf_counter()
    main(
        input_clustering_root=args.input,
        output_graph_root=args.output,
        feature_col=args.feature_col
    )
    t1 = time.perf_counter()
    print(f"Total inference time: {format_duration(t1 - t0)}")