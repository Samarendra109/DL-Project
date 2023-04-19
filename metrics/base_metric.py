from abc import ABC, abstractmethod
import torch
from torch import optim
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import MultiStepLR


class BaseMetric(ABC):
    """
    Base class for all the metrics (Abstract Class)
    Implement train_step and get_metric to add your implementation of metric

    (Must define a member called self.model 
        whose parameters will be added to optimizer.
    If not defined then override the get_optimizer method too.)
    """

    @abstractmethod
    def train_step(self, train_loader: DataLoader, epoch: int):
        """
        Represents one epoch in the training loop 
        """
        pass

    @abstractmethod
    def get_metric(self, dataset):
        """
        Returns the calculated metric for the dataset

        returns:
            tensor of indices in the dataset
            tensor of respective metrics for the datapoints in dataset
        """
        pass

    def get_optimizer(self):
        """
        Returns the optimizer for the model
        Hardcoding the optimizer (Can extend the class to overwrite it)

        (Must define a member called self.model in the __init__ of subclass
            If not defined then override the get_optimizer method too.)
        """
        return optim.SGD(
            self.model.parameters(),
            lr=0.1,
            momentum=0.9,
            weight_decay=0.0005,
            nesterov=True,
        )

    def train(self, train_loader: DataLoader, epochs: int):
        scheduler = MultiStepLR(self.optimizer, milestones=[60, 120, 160], gamma=0.2)
        self.model.train()
        for i in range(epochs):
            print(f"Epoch {i}\n")
            self.train_step(train_loader, i)
            scheduler.step()
