import torch
from torch import nn, optim
import copy
from tqdm import tqdm


class EL2NMetric:
    def __init__(self, model, num_models, device):

        self.model_list = nn.ModuleList(
            [copy.deepcopy(model) for _ in range(num_models)]
        )
        self.device = device
        self.model_list = self.model_list.to(device)
        self.optimizer = self.get_optimizer()
        # For EL2N loss is L2Norm or MSE Loss
        self.criterion = nn.MSELoss(reduction="none")
        # Assuming during training different loss is used
        self.training_criterion = nn.CrossEntropyLoss()

    def get_optimizer(self):
        # Hardcoding the optimizer (Can extend the class to overwrite it)
        return optim.SGD(self.model_list.parameters(), lr=0.001, momentum=0.9)

    def train_step(self, train_loader):

        for data in tqdm(train_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # forward + backward + optimize
            total_loss = 0.0
            for model in self.model_list:
                outputs = model(inputs)
                loss = torch.mean(self.training_criterion(outputs, labels))
                total_loss += loss
            total_loss.backward()
            self.optimizer.step()

    def train(self, train_loader, epochs):

        for i in range(epochs):
            print(f"Epoch {i}\n")
            self.train_step(train_loader)

    def get_metric(self, data_loader):

        with torch.no_grad():

            for data in data_loader:
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # calculate the metric
                total_loss = None
                for i, model in enumerate(self.model_list):
                    outputs = model(inputs)

                    if i == 0:
                        labels = torch.nn.functional.one_hot(
                            labels, num_classes=outputs.size(1)
                        )

                    loss = self.criterion(outputs, labels)
                    if total_loss == None:
                        total_loss = loss / len(self.model_list)
                    else:
                        total_loss += loss / len(self.model_list)

                # For now simply priting, will later figure out what to do.
                print(total_loss.size())
