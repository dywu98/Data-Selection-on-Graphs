"""
CIFAR100 专用的 GraphProbPrune Dataset
基于索引匹配（而非文件名），适配 CIFAR100 预加载数据集

核心差异：
- ImageNet 使用 image_name（文件路径）匹配样本
- CIFAR100 使用 idx（数据集索引）匹配样本
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


def load_all_graph_files(graph_dir: str) -> List[dict]:
    """
    读取指定目录下所有的 *.graph.pkl 文件

    返回: 列表，每个元素包含 'file_path' 和 'graph_data'
    """
    graph_dir = Path(graph_dir)
    if not graph_dir.exists():
        raise FileNotFoundError(f"图目录不存在: {graph_dir}")

    graph_files = sorted(graph_dir.rglob("*.graph.pkl"))
    if not graph_files:
        raise FileNotFoundError(f"未在 {graph_dir} 找到任何 .graph.pkl 文件")

    print(f"🔍 找到 {len(graph_files)} 个图文件")
    graph_data_list = []

    for file in tqdm(graph_files, desc="加载图文件"):
        with open(file, 'rb') as f:
            graph_data = pickle.load(f)
            graph_data_list.append({
                'file_path': str(file),
                'graph_data': graph_data
            })

    return graph_data_list


def compute_average_edge_weight_scores(nodes: List[dict], adj_matrix: np.ndarray) -> Dict[int, float]:
    """
    计算图中每个节点的 score：该节点所有连接边的权重的平均值

    参数:
        nodes: 节点列表，每个节点包含 'idx' (CIFAR100 数据集索引)
        adj_matrix: 邻接矩阵 (N, N)

    返回:
        Dict[idx -> score]
    """
    N = len(nodes)
    avg_scores = {}

    W = adj_matrix if not hasattr(adj_matrix, 'toarray') else adj_matrix.toarray()

    for i in range(N):
        weights = []
        for j in range(N):
            if i != j and W[i, j] > 0:
                weights.append(W[i, j])

        if len(weights) == 0:
            avg_score = 0.0
        else:
            avg_score = float(np.mean(weights))

        # 使用 idx 作为 key（CIFAR100 数据集索引）
        idx = nodes[i]['idx']
        avg_scores[idx] = avg_score

    return avg_scores


def collect_all_scores(graph_data_list: List[dict]) -> Dict[int, float]:
    """
    遍历所有图文件，提取每个样本的平均边权 score

    返回:
        Dict[idx -> score]
    """
    all_scores = {}

    for item in tqdm(graph_data_list, desc="计算每个图的平均边权分数"):
        graph_data = item['graph_data']
        nodes = graph_data['nodes']
        adj_matrix = graph_data['adj_matrix']

        scores_dict = compute_average_edge_weight_scores(nodes, adj_matrix)

        for idx, score in scores_dict.items():
            if idx in all_scores:
                print(f"⚠️ 重复 idx: {idx}，将覆盖前值")
            all_scores[idx] = score

    print(f"✅ 共计算出 {len(all_scores)} 个样本的分数")
    return all_scores


def create_score_array_by_index(
    scores: Dict[int, float],
    dataset_length: int,
    fill_value: float = 0.0
) -> np.ndarray:
    """
    根据 idx->score 映射创建分数数组

    参数:
        scores: Dict[idx -> score]
        dataset_length: 数据集总长度
        fill_value: 未匹配样本的填充值

    返回:
        scores: np.ndarray, shape=(dataset_length,)
    """
    score_array = np.full(dataset_length, fill_value, dtype=np.float32)

    for idx, score in scores.items():
        if 0 <= idx < dataset_length:
            score_array[idx] = float(score)
        else:
            print(f"⚠️ 警告：idx {idx} 超出数据集范围 [0, {dataset_length-1}]，跳过")

    valid_count = np.sum(score_array != fill_value)
    print(f"✅ 成功创建 score 数组，形状: {score_array.shape}")
    print(f"   - 匹配到的有效样本数: {valid_count}")
    print(f"   - 未匹配样本数: {dataset_length - valid_count} (填充为 {fill_value})")

    return score_array


class CIFARGraphProbPrune(Dataset):
    """
    CIFAR100 专用的 GraphProbPrune Dataset

    与 ImageNet 版本的主要区别：
    1. 使用 idx（数据集索引）进行匹配，而非 image_name（文件路径）
    2. 直接包装 torchvision.datasets.CIFAR100，无需 is_valid_file 等
    """

    def __init__(
        self,
        dataset,
        input_graph_dir: str,
        ratio: float = 0.5,
        num_epoch: int = None,
        delta: float = None,
        mode: str = "prob"
    ):
        """
        参数:
            dataset: torchvision.datasets.CIFAR100 实例
            input_graph_dir: 图文件根目录（包含 class_*/cluster_*.graph.pkl）
            ratio: 剪枝比例
            num_epoch: 总训练 epoch 数
            delta: annealing 参数
            mode: 选择模式，'prob' 或 'topk'
        """
        self.dataset = dataset
        self.ratio = ratio
        self.num_epoch = num_epoch
        self.delta = delta
        self.mode = mode

        # 初始化分数数组
        self.node_scores = np.ones([len(self.dataset)])
        self.final_scores = np.ones([len(self.dataset)])
        self.weights = np.ones(len(self.dataset))
        self.save_num = 0

        # 加载图数据并计算 edge scores
        graph_data_list = load_all_graph_files(input_graph_dir)
        scores = collect_all_scores(graph_data_list)
        self.edge_scores = create_score_array_by_index(
            scores=scores,
            dataset_length=len(self.dataset),
            fill_value=0.0
        )

        print(f"prune selection mode: {self.mode}")
        print(f"成功计算 edge scores，数据集长度: {len(self.dataset)}")

    def __setscore__(self, indices, values):
        """更新节点的动态分数（训练过程中调用）"""
        self.node_scores[indices] = values
        # final_scores = node_scores - edge_scores (edge scores 高表示与邻居相似，更容易被剪枝)
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
        # 选择 final_scores 低于 99 分位数的样本作为 well_learned
        b = self.final_scores < np.percentile(self.final_scores, 99)
        well_learned_samples = np.where(b)[0]
        pruned_samples = np.where(np.invert(b))[0]

        if well_learned_samples.shape == (0,) or self.final_scores.min() == self.final_scores.max():
            print('Cut {} samples for next iteration'.format(len(self.dataset) - len(pruned_samples)))
            self.save_num += len(self.dataset) - len(pruned_samples)
            np.random.shuffle(pruned_samples)
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
        # 设置随机种子以确保每个 epoch 不同的 shuffle
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