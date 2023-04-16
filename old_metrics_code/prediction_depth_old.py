import torch
from torch import nn, optim
import copy
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from data_utils import IndexDataset
import torch.nn.functional as F
from models import ResNet
from clustering import KNN


class ResNetForPredictionDepth(ResNet):
    def forward(self, x, k=-1, train=True):
        """
        :param x:
        :param k:
        :param train: switch model to test and extract the FMs of the kth layer
        :return:

        If k=-1 return all the layers
        """
        i = 0
        out = self.bn1(self.conv1(x))
        layer_outputs = []
        if k == i and not (train):
            return None, out.view(out.shape[0], -1)
        elif k == -1 and not (train):
            layer_outputs.append(torch.clone(out.view(out.shape[0], -1)))
        out = F.relu(out)
        i += 1
        for module in self.layer1:
            if not (train):
                _, out = module(
                    out, train=False
                )  # take the output of ResBlock before relu
                if k == i:
                    return None, out.view(out.shape[0], -1)
                elif k == -1:
                    layer_outputs.append(torch.clone(out.view(out.shape[0], -1)))
            else:
                out = module(out)
            out = F.relu(out)
            i += 1

        for module in self.layer2:
            if not (train):
                _, out = module(
                    out, train=False
                )  # take the output of ResBlock before relu
                if k == i:
                    return None, out.view(out.shape[0], -1)
                elif k == -1:
                    layer_outputs.append(torch.clone(out.view(out.shape[0], -1)))
            else:
                out = module(out)
            out = F.relu(out)
            i += 1
        for module in self.layer3:
            if not (train):
                _, out = module(
                    out, train=False
                )  # take the output of ResBlock before relu
                if k == i:
                    return None, out.view(out.shape[0], -1)
                elif k == -1:
                    layer_outputs.append(torch.clone(out.view(out.shape[0], -1)))
            else:
                out = module(out)
            out = F.relu(out)
            i += 1
        for module in self.layer4:
            if not (train):
                _, out = module(
                    out, train=False
                )  # take the output of ResBlock before relu
                if k == i:
                    return None, out.view(out.shape[0], -1)
                elif k == -1:
                    layer_outputs.append(torch.clone(out.view(out.shape[0], -1)))
            else:
                out = module(out)
            out = F.relu(out)
            i += 1
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out) / self.temp
        if not (train):
            _f = F.softmax(out, 1)  # take the output of softmax
            if k == i:
                return None, _f
            elif k == -1:
                layer_outputs.append(torch.clone(_f))

        if k == -1 and not train:
            return layer_outputs
        else:
            return out


class PredictionDepth:
    def __init__(
        self, model: ResNetForPredictionDepth, device: torch.device, k=30, layers=5
    ):

        self.model = model.to(device)
        self.device = device
        self.optimizer = self.get_optimizer()
        # Assuming during training different loss is used
        self.training_criterion = nn.CrossEntropyLoss()
        self.k = k
        self.layers = layers

    def get_optimizer(self):
        # Hardcoding the optimizer (Can extend the class to overwrite it)
        return optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

    def train_step(self, train_loader: DataLoader):

        for data in tqdm(train_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # forward + backward + optimize
            outputs = self.model(inputs)
            loss = self.training_criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

    def train(self, train_loader: DataLoader, epochs: int):

        for i in range(epochs):
            print(f"Epoch {i}\n")
            self.train_step(train_loader)

    def get_metric(self, dataset: Dataset):

        index_list = []
        metric_list = []

        dataset = IndexDataset(dataset)

        data_loader = DataLoader(
            dataset, batch_size=256, shuffle=False, num_workers=2  # Hardcoding for now
        )

        layer_outputs = []
        labels_list = []
        knn_models = []

        with torch.no_grad():

            for index, data in data_loader:
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # store the layer outputs
                layer_outputs_for_batch = self.model(inputs, k=-1, train=False)

                if len(layer_outputs) == 0:
                    for layer_output in layer_outputs_for_batch[-self.layers :]:
                        layer_outputs.append([layer_output])
                        labels_list.append([labels])
                else:
                    for layer_index, layer_output in enumerate(
                        layer_outputs_for_batch[-self.layers :]
                    ):
                        layer_outputs[layer_index].append(layer_output)
                        labels_list[layer_index].append(labels)

            for layer_output, labels in zip(layer_outputs, labels_list):
                x = torch.vstack(layer_output)
                for ele in layer_output:
                    del ele
                print(x.size())
                y = torch.hstack(labels)
                knn_models.append(KNN(x, y, k=self.k))

            for data_loader_i, (index, data) in tqdm(enumerate(data_loader)):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                final_metric = torch.ones(inputs.size(0)) * len(knn_models)
                final_metric = final_metric.to(self.device)
                prev_output = labels

                for i, knn_model in enumerate(knn_models):
                    knn_output = knn_model.predict(layer_outputs[i][data_loader_i])
                    if i == 0:
                        mask = knn_output == labels
                    else:
                        mask = (knn_output == labels) & (knn_output != prev_output)
                    prev_output = knn_output
                    final_metric.masked_fill_(mask, i)

                index_list.append(index)
                metric_list.append(final_metric)

        return torch.hstack(index_list), torch.hstack(metric_list)
