import os
import torch
import numpy as np
import torchvision
import math
import time
from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms, utils
from collections import defaultdict
from torch.utils.data import Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler
from operator import itemgetter
from typing import Iterator, List, Optional, Union
from pathlib import Path
from tqdm import tqdm
import pickle

# import os
# import pickle
# from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
# import numpy as np
# from tqdm import tqdm
# import torch
# from torch.utils.data import Dataset
# from torchvision.datasets import ImageFolder


def load_all_graph_files(graph_dir: str) -> List[Dict[str, Any]]:
    """
    读取指定目录下所有的 *.graph.pkl 文件，并加载为图数据列表。

    返回: 列表，每个元素是 (file_path, graph_data) 的元组
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


def compute_average_edge_weight_scores(nodes: List[Dict], adj_matrix: np.ndarray) -> Dict[str, float]:
    """
    计算图中每个节点的 score：该节点所有连接边的权重的平均值（即平均邻接边权）

    参数:
        nodes: 节点列表，每个节点包含 'image_name'
        adj_matrix: 邻接矩阵 (N, N)

    返回:
        Dict[image_name] -> score (float)
    """
    N = len(nodes)
    avg_scores = {}

    W = adj_matrix if not hasattr(adj_matrix, 'toarray') else adj_matrix.toarray()

    for i in range(N):
        # 获取该节点的所有边权重（非零）
        weights = []
        for j in range(N):
            if i != j and W[i, j] > 0:
                weights.append(W[i, j])

        if len(weights) == 0:
            avg_score = 0.0  # 孤立点
        else:
            avg_score = float(np.mean(weights))

        image_name = nodes[i]['image_name']
        avg_scores[image_name] = avg_score

    return avg_scores


def collect_all_scores(graph_data_list: List[Dict]) -> Dict[str, float]:
    """
    遍历所有图文件，提取每个样本的平均边权 score

    返回:
        Dict[image_name: str -> score: float]
    """
    all_scores = {}

    for item in tqdm(graph_data_list, desc="计算每个图的平均边权分数"):
        graph_data = item['graph_data']
        nodes = graph_data['nodes']
        adj_matrix = graph_data['adj_matrix']

        scores_dict = compute_average_edge_weight_scores(nodes, adj_matrix)

        # 合并到全局字典（理论上 image_name 不应重复）
        for img_name, score in scores_dict.items():
            if img_name in all_scores:
                print(f"⚠️ 重复 image_name: {img_name}，将覆盖前值")
            all_scores[img_name] = score

    print(f"✅ 共计算出 {len(all_scores)} 个样本的分数")
    return all_scores


def create_image_to_index_mapping(dataset: Dataset) -> Dict[str, int]:
    """
    从 ImageFolder 类型 dataset 构建 image_path (或 basename) 到 index 的映射

    注意: ImageFolder.dataset.imgs 是 (image_path, class_idx) 的列表
    我们取 image_path 的 basename 作为 key

    返回:
        Dict[basename -> index]
    """
    mapping = {}
    for idx, (img_path, _) in tqdm(enumerate(dataset.imgs)):
        img_name = Path(img_path).name
        mapping[img_name] = idx

    print(f"📌 dataset 包含 {len(mapping)} 个样本")
    return mapping


def match_scores_to_dataset(
    scores: Dict[str, float],
    dataset: Dataset
) -> List[Tuple[int, float]]:
    """
    将 graph 中计算出的 score 通过 image_name 匹配到 dataset 的索引上

    返回:
        List[(dataset_index, score)]，可用于后续排序、采样等
    """
    img_to_idx = create_image_to_index_mapping(dataset)
    matched = []
    # matched = {}

    not_found = []

    for img_name, score in scores.items():
        base_name = Path(img_name).name  # 确保只取文件名
        if base_name in img_to_idx:
            dataset_idx = img_to_idx[base_name]
            matched.append((dataset_idx, score))
            # matched[dataset_idx] = score
        else:
            not_found.append(img_name)

    if not_found:
        print(f"🔍 警告：共 {len(not_found)} 个样本未在 dataset 中找到: {not_found[:5]}...")

    print(f"✅ 成功匹配 {len(matched)} 个样本到 dataset")
    return matched

def create_score_array_by_dataset_index(
    matched_results: List[Tuple[int, float]],
    dataset: Dataset,
    fill_value: float = np.nan
) -> np.ndarray:
    """
    根据 matched_results 创建一个与 dataset 索引对齐的 score 数组

    参数:
        matched_results: List[(dataset_idx, score)]，已匹配好的结果
        dataset: Dataset，如 ImageFolder，用于确定总长度
        fill_value: 对于未匹配到的样本，填充值（推荐 np.nan 或 0.0）

    返回:
        scores: np.ndarray, shape=(len(dataset),), scores[i] 是 dataset[i] 的 score
    """
    n = len(dataset)
    scores = np.full(n, fill_value, dtype=np.float32)

    for idx, score in matched_results:
        if idx < 0 or idx >= n:
            print(f"⚠️ 警告：index {idx} 超出 dataset 范围 [0, {n-1}]，跳过")
            continue
        scores[idx] = float(score)

    # 统计有效值数量
    if np.isnan(fill_value):
        valid_count = np.sum(~np.isnan(scores))
    else:
        valid_count = np.sum(scores != fill_value)

    print(f"✅ 成功创建 score 数组，形状: {scores.shape}")
    print(f"   - 匹配到的有效样本数: {valid_count}")
    print(f"   - 未匹配样本数: {n - valid_count} (填充为 {fill_value})")

    return scores

class GraphProbPrune(Dataset):
    def __init__(self, dataset, input_graph_dir, ratio = 0.5, num_epoch=None, delta = None):
        self.dataset = dataset
        self.ratio = ratio
        self.num_epoch = num_epoch
        self.delta = delta
        self.node_scores = np.ones([len(self.dataset)])
        self.final_scores = np.ones([len(self.dataset)])
        self.transform = dataset.transform
        self.weights = np.ones(len(self.dataset))
        self.save_num = 0

        # Step 1: 加载所有图
        graph_data_list = load_all_graph_files(input_graph_dir)

        # Step 2: 计算每个样本的平均边权 score
        scores = collect_all_scores(graph_data_list)  # image_name -> score

        # Step 3: 匹配到 dataset
        matched_results = match_scores_to_dataset(scores, dataset)
        self.edge_scores = create_score_array_by_dataset_index(
                          matched_results=matched_results,
                          dataset=dataset,
                          fill_value=0.0  # 或 fill_value=0.0
                      )
        print(f"成功计算 edge scores")


    def __setscore__(self, indices, values):
        self.node_scores[indices] = values
        self.final_scores[indices] = self.node_scores[indices] + self.edge_scores[indices]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data, target = self.dataset[index]
        weight = self.weights[index]
        return data, target, index, weight


    def prune(self, ratio):
        
        b = self.final_scores<=np.max(self.final_scores)
        well_learned_samples = np.where(b)[0]
        pruned_samples=np.where(np.invert(b))[0]
        print('{} now ratio is '.format(ratio))

        if well_learned_samples.shape==(0,) or self.final_scores.min()==self.final_scores.max():
            print('Cut {} samples for next iteration'.format(len(self.dataset)-len(well_learned_samples)))
            self.save_num += len(self.dataset)-len(well_learned_samples)
            np.random.shuffle(well_learned_samples)
            return well_learned_samples
        else:
            prob = torch.softmax(torch.from_numpy(self.final_scores[well_learned_samples]), dim=0).numpy()
            selected = np.random.choice(well_learned_samples, int(ratio * len(well_learned_samples)), p=prob)
            print('{} All samples '.format(len(self.dataset)))
            print('{} final_scores '.format(len(self.final_scores)))
            print('{} well_learned_samples '.format(len(well_learned_samples)))
            print('{} selecetd samples '.format(len(selected)))


            self.reset_weights()
            if len(selected)>0:
                self.weights[selected]=1/ratio
                # pruned_samples=np.append(pruned_samples,selected)
                pruned_samples=selected
            print('Cut {} samples for next iteration'.format(len(self.dataset)-len(pruned_samples)))
            self.save_num += len(self.dataset)-len(pruned_samples)
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

    def get_weights(self,indexes):
        return self.weights[indexes]

    def total_save(self):
        return self.save_num

    def reset_weights(self):
        self.weights = np.ones(len(self.dataset))



class InfoBatchSampler():
    def __init__(self, infobatch_dataset, num_epoch = math.inf, delta = 1):
        self.infobatch_dataset = infobatch_dataset
        self.seq = None
        self.stop_prune = num_epoch * delta
        self.start_prune = 2
        self.seed = 0
        self.reset()

    def reset(self):
        np.random.seed(self.seed)
        self.seed+=1
        if self.seed>self.stop_prune:
            if self.seed <= self.stop_prune+1:
                self.infobatch_dataset.reset_weights()
            self.seq = self.infobatch_dataset.no_prune()
        # elif self.seed<self.start_prune:
        #     self.seq = self.infobatch_dataset.no_prune()
        else:
            xiuzheng_ratio = min(1.0, self.infobatch_dataset.ratio + (1-self.infobatch_dataset.ratio) * (1-(self.seed-self.start_prune)/(self.stop_prune-self.start_prune)))
            self.seq = self.infobatch_dataset.prune(xiuzheng_ratio)
        self.ite = iter(self.seq)
        self.new_length = len(self.seq)

    def __next__(self):
        try:
            nxt = next(self.ite)
            return nxt
        except StopIteration:
            self.reset()
            raise StopIteration

    def __len__(self):
        return len(self.seq)

    def __iter__(self):
        self.ite = iter(self.seq)
        return self

class DatasetFromSampler(Dataset):
    """Dataset to create indexes from `Sampler`.
    Args:
        sampler: PyTorch sampler
    """

    def __init__(self, sampler: Sampler):
        """Initialisation for DatasetFromSampler."""
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        """Gets element of the dataset.
        Args:
            index: index of the element in the dataset
        Returns:
            Single element by index
        """
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        return len(self.sampler)

class DistributedSamplerWrapper(DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.
    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.
    .. note::
        Sampler can change size during training.
    """

    def __init__(
        self,
        sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
    ):
        """
        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
                distributed training
            rank (int, optional): Rank of the current process
                within ``num_replicas``
            shuffle (bool, optional): If true (default),
                sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        self.sampler = sampler

    def __iter__(self) -> Iterator[int]:
        """Iterate over sampler.
        Returns:
            python iterator
        """
#         self.sampler.reset()
        self.dataset = DatasetFromSampler(self.sampler)
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))


@torch.no_grad()
def concat_all_gather(tensor, dim=0):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
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

    if torch.distributed.get_rank()==0:
        return True

    return False

def split_index(t):
    low_mask = 0b111111111111111
    low = torch.tensor([x&low_mask for x in t])
    high = torch.tensor([(x>>15)&low_mask for x in t])
    return low,high

def recombine_index(low,high):
    original_tensor = torch.tensor([(high[i]<<15)+low[i] for i in range(len(low))])
    return original_tensor