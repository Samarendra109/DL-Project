import torch
from typing import Tuple, Any
from torch.utils.data import Dataset, SubsetRandomSampler, Subset


class IndexDataset(Dataset):
    """
    Custom dataset that returns the index of the datapoints 
        along with the datapoints.
    """

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        return index, self.dataset[index]


def get_subset_random_sampler(
    indices: torch.LongTensor, metrics: torch.FloatTensor, alpha: float
) -> SubsetRandomSampler:
    """
    Used for pruning the dataset based on the metric passed.
    alpha denotes the percentage of the data to keep
    """
    indices, metrics = indices.cpu(), metrics.cpu()
    k = int(len(indices) * alpha)
    pruned_indices = indices[torch.topk(metrics, k).indices]
    return SubsetRandomSampler(pruned_indices)


def get_subset(
    indices: torch.LongTensor,
    metrics: torch.FloatTensor,
    trainset: Dataset,
    alpha: float,
) -> Dataset:
    """
    Used for pruning the dataset based on the metric passed.
    alpha denotes the percentage of the data to keep
    """
    indices, metrics = indices.cpu(), metrics.cpu()
    k = int(len(indices) * alpha)
    pruned_indices = indices[torch.topk(metrics, k).indices]
    return Subset(trainset, pruned_indices)
