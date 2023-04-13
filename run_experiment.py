import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from vog import VoGMetric
from el2n import EL2NMetric
from models import BasicBlock, ResNet
from data_utils import get_subset_random_sampler
import argparse
from torch.optim.lr_scheduler import MultiStepLR

def test(model, testloader, device):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * (1 - correct / total)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric', default='vog', choices = ['vog', 'el2n'], help='metric to use for data pruning')
    parser.add_argument('--dataset', default='cifar10', choices = ['cifar10', 'svhn'], help='dataset to use for data pruning')
    parser.add_argument('--batch_size', type=int, help='batch size to use', default = 128)
    parser.add_argument('--epochs', type=int, help='epochs to train for', default = 200)
    args = parser.parse_args()
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    batch_size = args.batch_size

    if args.dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False, download=True, transform=transform)
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        frac_list = list(range(100, 10, -10))
    elif args.dataset == 'svhn':
        trainset = torchvision.datasets.SVHN(root='./data/svhn', split='train', download=True, transform=transform)
        testset = torchvision.datasets.SVHN(root='./data/svhn', split='test', download=True, transform=transform)
        classes = list(range(10))
        frac_list = [100, 60, 36, 22, 13, 8, 5, 3, 2, 1]
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = ResNet(BasicBlock, [2, 2, 2, 2], temp=1.0, num_classes=len(classes))
    if args.metric == 'vog':
        metric = VoGMetric(model, device=device)
        metric_train_args = {'epochs':args.epochs, 'checkpoint_interval':10}
    elif args.metric == 'el2n':
        metric = EL2NMetric(model, 10, device)
        metric_train_args = {'epochs':args.epochs}
        
    metric.train(trainloader, **metric_train_args)
    indices, metrics = metric.get_metric(trainset)
    with open(f'./results_metrics_{args.dataset}_{args.metric}.pkl', 'wb') as f:
        pickle.dump((indices, metrics), f)
    
    for frac in frac_list:
        sampler = get_subset_random_sampler(indices, metrics, frac/100)
        frac_trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=2, sampler=sampler)
        model = ResNet(BasicBlock, [2, 2, 2, 2], temp=1.0, num_classes=len(classes))
        model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005, nesterov=True)
        criterion = nn.CrossEntropyLoss()
        scheduler = MultiStepLR(optimizer, milestones=[60,120,160], gamma=0.2)
        test_errors = {}
        for epoch in range(args.epochs):
            running_loss = 0.0
            for i, data in enumerate(frac_trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            test_error = test(model, testloader, device)
            test_errors[epoch] = test_error
            print(f"Epoch: {epoch + 1}, Test Error: {test_error:.2f}%")
            scheduler.step()
        with open(f'./results_errors_{args.dataset}_{args.metric}_{frac}.pkl', 'wb') as f:
            pickle.dump(test_errors, f)
        print(f"Finished training with frac {frac / 100:.2f}, Test Errors: {test_errors}")
        
if __name__ == '__main__':
    main()