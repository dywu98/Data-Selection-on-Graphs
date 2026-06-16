"""
CIFAR100 全量图 GraphProbPrune Dataset
支持加载单个全量图文件（不按类别聚类）

优化版本：
- 直接使用预计算的 edge_scores，不需要加载邻接矩阵
- 加载速度大幅提升

与 cifar_graph_dataloader.py 的区别：
- 加载单个 graph.pkl 文件而非目录下的多个 class_*/cluster_*.graph.pkl
- 直接通过 idx 匹配样本
"""

import os
import torch
import numpy as np
import torchvision
import math
import time
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from collections import defaultdict
from torch.utils.data import Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler
from operator import itemgetter
from typing import Dict, Iterator, List, Optional, Union
from pathlib import Path
from tqdm import tqdm
import pickle


def load_full_graph(graph_path: str, verbose: bool = True) -> dict:
    """
    加载单个全量图文件

    参数:
        graph_path: graph.pkl 文件路径
        verbose: 是否打印加载时间

    返回:
        graph_data: 图数据字典
    """
    graph_path = Path(graph_path)
    if not graph_path.exists():
        raise FileNotFoundError(f"图文件不存在: {graph_path}")

    if verbose:
        print(f"加载图文件: {graph_path}")
        t0 = time.perf_counter()

    with open(graph_path, 'rb') as f:
        graph_data = pickle.load(f)

    if verbose:
        t1 = time.perf_counter()
        print(f"[Time] 加载图文件耗时: {t1 - t0:.2f} 秒")
        print(f"图包含 {graph_data['num_nodes']} 个节点")

    return graph_data


def create_score_array_from_full_graph(
    graph_data: dict,
    dataset_length: int,
    fill_value: float = 0.0,
    verbose: bool = True
) -> np.ndarray:
    """
    从图数据创建分数数组

    参数:
        graph_data: 图数据字典，应包含 'edge_scores' 或 'nodes' + 'adj_matrix'
        dataset_length: 数据集总长度
        fill_value: 未匹配样本的填充值
        verbose: 是否打印信息

    返回:
        score_array: np.ndarray, shape=(dataset_length,)
    """
    if verbose:
        t0 = time.perf_counter()

    score_array = np.full(dataset_length, fill_value, dtype=np.float32)

    # 优先使用预计算的 edge_scores
    if 'edge_scores' in graph_data:
        scores = graph_data['edge_scores']
        matched = 0
        for idx, score in scores.items():
            if 0 <= idx < dataset_length:
                score_array[idx] = float(score)
                matched += 1

        if verbose:
            t1 = time.perf_counter()
            print(f"[Time] 从预计算的 edge_scores 创建数组耗时: {t1 - t0:.2f} 秒")
            print(f"✅ 成功创建 score 数组，形状: {score_array.shape}")
            print(f"   - 匹配到的有效样本数: {matched}")
            print(f"   - 未匹配样本数: {dataset_length - matched} (填充为 {fill_value})")

    elif 'adj_matrix' in graph_data:
        # 兼容旧版本：从邻接矩阵计算
        if verbose:
            print("图中没有预计算的 edge_scores，从邻接矩阵计算...")

        nodes = graph_data['nodes']
        adj_matrix = graph_data['adj_matrix']
        N = len(nodes)

        W = adj_matrix.toarray() if hasattr(adj_matrix, 'toarray') else adj_matrix

        for i in tqdm(range(N), desc="Computing node scores"):
            weights = []
            for j in range(N):
                if i != j and W[i, j] > 0:
                    weights.append(W[i, j])

            if len(weights) > 0:
                avg_score = float(np.mean(weights))
            else:
                avg_score = 0.0

            idx = nodes[i]['idx']
            if 0 <= idx < dataset_length:
                score_array[idx] = avg_score

        if verbose:
            t1 = time.perf_counter()
            print(f"[Time] 从邻接矩阵计算 scores 耗时: {t1 - t0:.2f} 秒")

    else:
        raise ValueError("图数据既没有 'edge_scores' 也没有 'adj_matrix'，无法创建分数数组")

    return score_array


class CIFARFullGraphProbPrune(Dataset):
    """
    CIFAR100 全量图 GraphProbPrune Dataset

    与 CIFARGraphProbPrune 的区别：
    - 加载单个全量图文件而非目录下的多个聚类图
    - 适用于不按类别聚类的对比实验
    """

    def __init__(
        self,
        dataset,
        graph_path: str,
        ratio: float = 0.5,
        num_epoch: int = None,
        delta: float = None,
        mode: str = "prob",
        verbose: bool = True
    ):
        """
        参数:
            dataset: torchvision.datasets.CIFAR100 实例
            graph_path: 全量图文件路径 (单个 .pkl 文件)
            ratio: 剪枝比例
            num_epoch: 总训练 epoch 数
            delta: annealing 参数
            mode: 选择模式，'prob' 或 'topk'
            verbose: 是否打印加载信息
        """
        self.dataset = dataset
        self.ratio = ratio
        self.num_epoch = num_epoch
        self.delta = delta
        self.mode = mode
        self.verbose = verbose

        # 初始化分数数组
        self.node_scores = np.ones([len(self.dataset)])
        self.final_scores = np.ones([len(self.dataset)])
        self.weights = np.ones(len(self.dataset))
        self.save_num = 0

        # 计时：加载图
        t_load_start = time.perf_counter()

        # 加载全量图
        graph_data = load_full_graph(graph_path, verbose=verbose)

        # 创建分数数组
        self.edge_scores = create_score_array_from_full_graph(
            graph_data=graph_data,
            dataset_length=len(self.dataset),
            fill_value=0.0,
            verbose=verbose
        )

        t_load_end = time.perf_counter()
        self.load_time = t_load_end - t_load_start

        if verbose:
            print(f"prune selection mode: {self.mode}")
            print(f"数据集长度: {len(self.dataset)}")
            print(f"[Time] 图加载总耗时: {self.load_time:.2f} 秒")

    def __setscore__(self, indices, values):
        """更新节点的动态分数（训练过程中调用）"""
        self.node_scores[indices] = values
        self.final_scores[indices] = self.node_scores[indices] - self.edge_scores[indices]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data, target = self.dataset[index]
        weight = self.weights[index]
        return data, target, index, weight

    def prune(self):
        """
        执行剪枝，返回当前 epoch 要使用的样本索引列表
        """
        t0 = time.perf_counter()

        # 选择 final_scores 低于 99 分位数的样本作为 well_learned
        b = self.final_scores < np.percentile(self.final_scores, 99)
        well_learned_samples = np.where(b)[0]
        pruned_samples = np.where(np.invert(b))[0]

        if well_learned_samples.shape == (0,) or self.final_scores.min() == self.final_scores.max():
            print('Cut {} samples for next iteration'.format(len(self.dataset) - len(pruned_samples)))
            self.save_num += len(self.dataset) - len(pruned_samples)
            np.random.shuffle(pruned_samples)
            self.prune_time = time.perf_counter() - t0
            return pruned_samples

        # 根据模式选择样本
        prob = torch.softmax(torch.from_numpy(self.final_scores[well_learned_samples]) / 5.0, dim=0).numpy()
        k = int(self.ratio * len(well_learned_samples))

        if self.mode == 'prob':
            selected = np.random.choice(well_learned_samples, k, p=prob)
        elif self.mode == 'topk':
            topk_idx = np.argsort(prob)[-k:][::-1]
            selected = well_learned_samples[topk_idx]
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        self.reset_weights()
        if len(selected) > 0:
            self.weights[selected] = 1 / self.ratio
            pruned_samples = np.append(pruned_samples, selected)

        print('Cut {} samples for next iteration'.format(len(self.dataset) - len(pruned_samples)))
        self.save_num += len(self.dataset) - len(pruned_samples)
        np.random.shuffle(pruned_samples)

        self.prune_time = time.perf_counter() - t0
        return pruned_samples

    def pruning_sampler(self):
        return InfoBatchSampler(self, self.num_epoch, self.delta)

    def no_prune(self):
        samples = list(range(len(self.dataset)))
        np.random.shuffle(samples)
        return samples

    def mean_score(self):
        return self.final_scores.mean()

    def normal_sampler_no_prune(self):
        return InfoBatchSampler(self.no_prune)

    def get_weights(self, indexes):
        return self.weights[indexes]

    def total_save(self):
        return self.save_num

    def reset_weights(self):
        self.weights = np.ones(len(self.dataset))

    def total_time_cost(self):
        """兼容 ImageNet 版本接口"""
        return 0

    def get_load_time(self):
        """返回图加载时间"""
        return self.load_time

    def get_last_prune_time(self):
        """返回最近一次剪枝操作的时间"""
        return getattr(self, 'prune_time', 0.0)


class InfoBatchSampler:
    """InfoBatch 采样器"""

    def __init__(self, infobatch_dataset, num_epoch=float('inf'), delta=1):
        self.infobatch_dataset = infobatch_dataset
        self.seq = None
        self.stop_prune = num_epoch * delta
        self.seed = 0
        self.reset()

    def reset(self):
        np.random.seed(self.seed)
        self.seed += 1
        if self.seed > self.stop_prune:
            if self.seed <= self.stop_prune + 1:
                self.infobatch_dataset.reset_weights()
            self.seq = self.infobatch_dataset.no_prune()
        else:
            self.seq = self.infobatch_dataset.prune()
        self.ite = iter(self.seq)
        self.new_length = len(self.seq)

    def __next__(self):
        try:
            return next(self.ite)
        except StopIteration:
            self.reset()
            raise StopIteration

    def __len__(self):
        return len(self.seq)

    def __iter__(self):
        self.ite = iter(self.seq)
        return self

    def set_epoch(self, epoch):
        """兼容 DistributedSampler 接口"""
        np.random.seed(epoch)


class DatasetFromSampler(Dataset):
    """Dataset to create indexes from `Sampler`."""

    def __init__(self, sampler: Sampler):
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        return len(self.sampler)


class DistributedSamplerWrapper(DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    """

    def __init__(
        self,
        sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
    ):
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        self.sampler = sampler

    def __iter__(self) -> Iterator[int]:
        self.dataset = DatasetFromSampler(self.sampler)
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))


@torch.no_grad()
def concat_all_gather(tensor, dim=0):
    """
    Performs all_gather operation on the provided tensors.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=dim)
    return output


def is_master():
    if not torch.distributed.is_available():
        return True
    if not torch.distributed.is_initialized():
        return True
    if torch.distributed.get_rank() == 0:
        return True
    return False


def split_index(t):
    """将大索引拆分为低 15 位和高位（用于分布式训练的索引传递）"""
    low_mask = 0b111111111111111
    low = torch.tensor([x & low_mask for x in t])
    high = torch.tensor([(x >> 15) & low_mask for x in t])
    return low, high


def recombine_index(low, high):
    """重组拆分的索引"""
    original_tensor = torch.tensor([(high[i] << 15) + low[i] for i in range(len(low))])
    return original_tensor