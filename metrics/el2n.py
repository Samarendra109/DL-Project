import torch
from torch import nn, optim
import copy
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from data_utils import IndexDataset
from torch.optim.lr_scheduler import MultiStepLR
from .base_metric import BaseMetric


class EL2NMetric(BaseMetric):
    def __init__(self, model: nn.Module, num_models: int, device: torch.device):

        self.model = nn.ModuleList([copy.deepcopy(model) for _ in range(num_models)])
        self.device = device
        self.model = self.model.to(device)
        self.optimizer = self.get_optimizer()
        # For EL2N loss is L2Norm or MSE Loss
        self.criterion = nn.MSELoss(reduction="none")
        # Assuming during training different loss is used
        self.training_criterion = nn.CrossEntropyLoss()

    def train_step(self, train_loader: DataLoader, epoch: int):

        for data in tqdm(train_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # forward + backward + optimize
            total_loss = 0.0
            for model in self.model:
                outputs = model(inputs)
                loss = torch.mean(self.training_criterion(outputs, labels))
                total_loss += loss
            total_loss.backward()
            self.optimizer.step()

    def get_metric(self, dataset: Dataset):

        index_list = []
        metric_list = []

        dataset = IndexDataset(dataset)

        data_loader = DataLoader(
            dataset, batch_size=256, shuffle=False, num_workers=2  # Hardcoding for now
        )

        self.model.eval()

        with torch.no_grad():

            for index, data in data_loader:
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # calculate the metric
                total_loss = None
                for i, model in enumerate(self.model):
                    outputs = torch.nn.functional.softmax(model(inputs), dim=-1)

                    if i == 0:
                        labels = torch.nn.functional.one_hot(
                            labels, num_classes=outputs.size(1)
                        )

                    loss = self.criterion(outputs, labels)
                    if total_loss == None:
                        total_loss = loss / len(self.model)
                    else:
                        total_loss += loss / len(self.model)

                total_loss = total_loss.mean(dim=-1)

                index_list.append(index)
                metric_list.append(total_loss)

        return torch.hstack(index_list), torch.hstack(metric_list)
