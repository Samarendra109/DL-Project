import torch
from torch import nn, optim
import copy
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from data_utils import IndexDataset
from torch.optim.lr_scheduler import MultiStepLR
from .base_metric import BaseMetric


class RandomPruningMetric(BaseMetric):
    """
    Implements the random pruning metric.
    Gives a random score for each datapoint in the metric.
    """

    def __init__(self, device: torch.device):
        self.device = device

    def train_step(self, train_loader: DataLoader, epoch: int):
        pass

    def get_metric(self, dataset: Dataset):
        """
        Assigns a random number as metric to each datapoint.
        """

        index_list = []
        metric_list = []

        dataset = IndexDataset(dataset)

        data_loader = DataLoader(
            dataset, batch_size=256, shuffle=False, num_workers=2  # Hardcoding for now
        )

        with torch.no_grad():

            for index, data in data_loader:
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # calculate the metric
                metric = torch.rand(*index.size()).to(self.device)

                index_list.append(index)
                metric_list.append(metric)

        return torch.hstack(index_list), torch.hstack(metric_list)
