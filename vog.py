import torch
from torch import nn, optim
import copy
from torch.utils.data import Dataset, DataLoader
from data_utils import IndexDataset

class VoGMetric:
    def __init__(self, model: nn.Module, num_checkpoints: int, device: torch.device):
        self.base_model = copy.deepcopy(model)
        self.checkpoints = []
        self.num_checkpoints = num_checkpoints
        self.device = device
        self.base_model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, train_loader: DataLoader, epochs: int, checkpoint_interval: int):
        optimizer = optim.SGD(self.base_model.parameters(), lr=0.001, momentum=0.9)
        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()

                outputs = self.base_model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")

            if (epoch + 1) % checkpoint_interval == 0:
                checkpoint = copy.deepcopy(self.base_model)
                self.checkpoints.append(checkpoint)

        print("Finished training")

    def get_metric(self, dataset: Dataset):
        index_list = []
        metric_list = []

        dataset = IndexDataset(dataset)

        data_loader = DataLoader(
            dataset, batch_size=256, shuffle=False, num_workers=2
        )

        for model in self.checkpoints:
            model.eval()

        # with torch.no_grad():
        for index, data in data_loader:
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            inputs.requires_grad_(True)

            gradients = []
            for model in self.checkpoints:
                outputs = model(inputs)
                pre_softmax_grads = torch.autograd.grad(outputs.sum(), inputs, create_graph=True)[0]
                gradients.append(pre_softmax_grads)
            inputs.requires_grad_(False)
            mean_gradient = sum(gradients) / len(gradients)
            mean_gradient = mean_gradient.mean(dim=1, keepdim=True)  # Averaging over color channels
            pixelwise_var = torch.sqrt(sum([(g.mean(dim=1, keepdim=True) - mean_gradient)**2 for g in gradients]) / len(gradients))
            vog = pixelwise_var.mean(dim=(-1, -2)).squeeze()

            index_list.append(index)
            metric_list.append(vog)

        return torch.hstack(index_list), torch.hstack(metric_list)
