import torch
from torch import nn, optim
import copy
from torch.utils.data import Dataset, DataLoader
from data_utils import IndexDataset
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
import numpy as np
from .base_metric import BaseMetric


class VoGMetric(BaseMetric):
    def __init__(
        self, model: nn.Module, device: torch.device, checkpoint_interval: int
    ):
        self.model = copy.deepcopy(model)
        self.checkpoints = []
        self.device = device
        self.model.to(self.device)
        self.optimizer = self.get_optimizer()
        self.criterion = nn.CrossEntropyLoss()
        self.checkpoint_interval = checkpoint_interval

    def train_step(self, train_loader: DataLoader, epoch: int):

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(inputs, train=True)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")
        if (epoch + 1) % self.checkpoint_interval == 0:
            checkpoint = copy.deepcopy(self.model)
            self.checkpoints.append(checkpoint)

    def get_metric(self, dataset: Dataset):
        index_list = []
        metric_list = []

        dataset = IndexDataset(dataset)

        data_loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=2)

        for model in self.checkpoints:
            model.eval()

        # with torch.no_grad():
        for index, data in tqdm(data_loader):
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            inputs.requires_grad_(True)

            gradients = []
            for model in self.checkpoints:
                outputs = model(inputs)
                pre_softmax_grads = (
                    torch.autograd.grad(outputs.sum(), inputs, create_graph=True)[0]
                    .detach()
                    .cpu()
                )
                gradients.append(pre_softmax_grads)
            inputs.requires_grad_(False)
            mean_gradient = sum(gradients) / len(gradients)
            mean_gradient = mean_gradient.mean(
                dim=1, keepdim=True
            )  # Averaging over color channels
            pixelwise_var = np.sqrt(
                sum(
                    [
                        (g.mean(dim=1, keepdim=True) - mean_gradient) ** 2
                        for g in gradients
                    ]
                )
                / len(gradients)
            )
            vog = pixelwise_var.mean(dim=(-1, -2)).squeeze()

            index_list.append(index)
            metric_list.append(vog)

        return torch.hstack(index_list), torch.hstack(metric_list)
