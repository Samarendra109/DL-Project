import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from vog import VoGMetric
from el2n import EL2NMetric
from models import BasicBlock, ResNet

# from torchvision.models import resnet18, ResNet18_Weights
from data_utils import get_subset_random_sampler, get_subset
import argparse
from torch.optim.lr_scheduler import MultiStepLR
import pickle
import os
import copy


def test(model, testloader, criterion, device):
    correct = 0
    total = 0
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()
    return 100 * (1 - correct / total), running_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metric",
        default="vog",
        choices=["vog", "el2n"],
        help="metric to use for data pruning",
    )
    parser.add_argument(
        "--dataset",
        default="cifar10",
        choices=["cifar10", "svhn"],
        help="dataset to use for data pruning",
    )
    parser.add_argument("--batch_size", type=int, help="batch size to use", default=128)
    parser.add_argument(
        "--probe_epochs",
        type=int,
        help="epochs to train the probe model for",
        default=200,
    )
    parser.add_argument("--epochs", type=int, help="epochs to train for", default=200)
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        help="checkpoint interval for VoG metric",
        default=5,
    )
    parser.add_argument(
        "--num_models", type=int, help="number of models for EL2N metric", default=10
    )
    args = parser.parse_args()

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    # transform = ResNet18_Weights.DEFAULT.transforms()
    batch_size = args.batch_size

    if args.dataset == "cifar10":
        trainset = torchvision.datasets.CIFAR10(
            root="./data/cifar10", train=True, download=True, transform=transform
        )
        testset = torchvision.datasets.CIFAR10(
            root="./data/cifar10", train=False, download=True, transform=transform
        )
        classes = (
            "plane",
            "car",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        )
        frac_list = list(range(100, 10, -10))
    elif args.dataset == "svhn":
        trainset = torchvision.datasets.SVHN(
            root="./data/svhn", split="train", download=True, transform=transform
        )
        testset = torchvision.datasets.SVHN(
            root="./data/svhn", split="test", download=True, transform=transform
        )
        classes = list(range(10))
        frac_list = [100, 60, 36, 22, 13, 8, 5, 3, 2, 1]

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    metrics_filename = f"./results/metrics_{args.dataset}_{args.metric}.pkl"
    if not os.path.exists(metrics_filename):
        model = ResNet(BasicBlock, [2, 2, 2, 2], temp=1.0, num_classes=len(classes))
        # model = resnet18(progress=False)
        if args.metric == "vog":
            metric = VoGMetric(model, device=device)
            metric_train_args = {
                "epochs": args.probe_epochs,
                "checkpoint_interval": args.checkpoint_interval,
            }
        elif args.metric == "el2n":
            metric = EL2NMetric(model, args.num_models, device)
            metric_train_args = {"epochs": args.probe_epochs}

        metric.train(trainloader, **metric_train_args)
        indices, metrics = metric.get_metric(trainset)
        with open(metrics_filename, "wb") as f:
            pickle.dump((indices, metrics), f)
    else:
        print("Loading metrics from file")
        with open(metrics_filename, "rb") as f:
            indices, metrics = pickle.load(f)

    for frac in frac_list:
        results_filename = f"./results/errors_{args.dataset}_{args.metric}_{frac}.pkl"
        if os.path.exists(results_filename):
            continue
        # sampler = get_subset_random_sampler(indices, metrics, frac/100)
        train_subset = get_subset(indices, metrics, trainset, frac / 100)
        frac_trainloader = torch.utils.data.DataLoader(
            train_subset, batch_size=batch_size, shuffle=True, num_workers=2
        )
        model = ResNet(BasicBlock, [2, 2, 2, 2], temp=1.0, num_classes=len(classes))
        # model = resnet18(progress=False)
        model.to(device)
        optimizer = optim.SGD(
            model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005, nesterov=True
        )
        criterion = nn.CrossEntropyLoss()
        scheduler = MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
        test_errors = {}
        training_losses = {}
        test_losses = {}
        for epoch in range(args.epochs):
            running_loss = 0.0
            for i, data in enumerate(frac_trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()

                running_loss += loss.item()

            test_error, test_loss = test(model, testloader, criterion, device)
            test_errors[epoch] = test_error
            test_losses[epoch] = test_loss / len(testloader)
            training_losses[epoch] = running_loss / len(frac_trainloader)
            print(
                f"Epoch: {epoch + 1}, Training Loss: {training_losses[epoch]}, Test Loss: {test_losses[epoch]}, Test Error: {test_error:.2f}%"
            )
            scheduler.step()
        with open(results_filename, "wb") as f:
            pickle.dump((test_errors, test_losses, training_losses), f)
        print(
            f"Finished training with frac {frac / 100:.2f}, Test Errors: {test_errors}"
        )
        del model


if __name__ == "__main__":
    main()
