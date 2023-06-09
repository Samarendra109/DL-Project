import torch
from torch import nn, optim
import copy
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from data_utils import IndexDataset
import torch.nn.functional as F
from models import ResNet
from clustering import KNN
from torch.optim.lr_scheduler import MultiStepLR
from .base_metric import BaseMetric


class PredictionDepth(BaseMetric):
    """
    Implements the prediction depth metric
    """

    def __init__(
        self, model: ResNet, device: torch.device, k=30, layers=5, alpha_to_save=1.0
    ):

        """
        Args:
            model: probing model to be trained
            device: device
            k: the nearest neighbors size for KNN model
            layers: If number is l then last l layers is taken.
            alpha_to_save: the percentage of data to save in the KNN model.
                (As it is inefficient to store the complete dataset in the KNN models)
        """

        self.model = model.to(device)
        self.device = device
        self.optimizer = self.get_optimizer()
        self.training_criterion = nn.CrossEntropyLoss()
        self.k = k
        self.layers = layers
        self.alpha_to_save = alpha_to_save

    def train_step(self, train_loader: DataLoader, epoch: int):

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

    def get_metric(self, dataset: Dataset):
        """
        returns the prediction depth metric for the dataset.
        """

        index_list = []
        metric_list = []

        dataset = IndexDataset(dataset)

        data_loader = DataLoader(
            dataset, batch_size=256, shuffle=False, num_workers=2  # Hardcoding for now
        )

        self.model.eval()

        with torch.no_grad():

            final_metric_list = []
            starting_layer_index = self.model.get_num_layers() - self.layers

            for layer_index in range(starting_layer_index, self.model.get_num_layers()):

                x, y = None, None

                for index, data in tqdm(data_loader):
                    # get the inputs; data is a list of [inputs, labels]
                    inputs, labels = data
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    _, layer_output = self.model(inputs, layer_index, train=False)

                    if layer_index == starting_layer_index:
                        final_metric = (
                            torch.ones(inputs.size(0)) * self.model.get_num_layers()
                        )
                        final_metric = final_metric.to(self.device)
                        # print(len(final_metric_list))
                        final_metric_list.append(final_metric)

                    indices = torch.randperm(layer_output.size(0))[
                        : int(self.alpha_to_save * layer_output.size(0))
                    ]
                    layer_output = layer_output[indices, :]
                    labels = labels[indices]

                    if x == None:
                        x = layer_output
                        y = labels
                    else:
                        x = torch.vstack((x, layer_output))
                        y = torch.hstack((y, labels))

                knn_model = KNN(x, y, k=self.k)

                curr_output = []

                for t_i, (index, data) in tqdm(enumerate(data_loader)):
                    # get the inputs; data is a list of [inputs, labels]
                    inputs, labels = data
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    _, layer_output = self.model(inputs, layer_index, train=False)
                    knn_output = knn_model.predict(layer_output)

                    # If current output is not labels then reset it to being hardest
                    final_metric_list[t_i].masked_fill_(
                        knn_output != labels, self.model.get_num_layers()
                    )

                    # If current output is equals to labels
                    #  then only update if the previous output was not equal to labels
                    if layer_index == starting_layer_index:
                        mask = knn_output == labels
                    else:
                        mask = (knn_output == labels) & (prev_output[t_i] != labels)

                    curr_output.append(knn_output)

                    final_metric_list[t_i].masked_fill_(mask, layer_index)

                    if layer_index == self.model.get_num_layers() - 1:
                        index_list.append(index)

                prev_output = curr_output

            return torch.hstack(index_list), torch.hstack(final_metric_list)
