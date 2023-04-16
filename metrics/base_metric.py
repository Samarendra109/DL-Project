from abc import ABC, abstractmethod
import torch
from torch import optim
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import MultiStepLR


class BaseMetric(ABC):
    @abstractmethod
    def train_step(self, train_loader: DataLoader, epoch: int):
        pass

    @abstractmethod
    def get_metric(self, dataset):
        pass

    def get_optimizer(self):
        # Hardcoding the optimizer (Can extend the class to overwrite it)
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
