import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def plot_data(trainloader):
    """ Task 3b) """
    idx = np.random.randint(low=0, high=trainloader.dataset.data.shape[0], size=16)
    fig = plt.figure(figsize=(10, 10))
    for i, i_idx in enumerate(idx):
        ax = fig.add_subplot(4, 4, i + 1)
        ax.axis('off')
        ax.imshow(trainloader.dataset.data[i_idx])
        ax.set_title(trainloader.dataset.classes[trainloader.dataset.targets[i_idx]])
    plt.tight_layout()
    plt.show()


class Net(nn.Module):
    """ Task 3c) """
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 3 * 32 * 32)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = F.log_softmax(self.fc3(x), dim=1)
        return out

    def train_net(self, criterion, optimizer, trainloader, epochs, _net="MLP"):
        log_interval = 10
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(trainloader):
                data, target = Variable(data), Variable(target)
                data, target = data.to(device), target.to(device)
                # resize data from (batch_size, 1, 28, 28) to (batch_size, 28*28)
                if _net == "MLP":
                    data = data.view(-1, 3 * 32 * 32)
                optimizer.zero_grad()
                net_out = self(data)
                loss = criterion(net_out, target)
                loss.backward()
                optimizer.step()
                if batch_idx % log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
                                                                                   len(trainloader.dataset),
                                                                                   100. * batch_idx / len(trainloader),
                                                                                   loss.data.item()))

    def test_net(self, criterion, testloader, _net="MLP"):
        # test the net
        test_loss = 0
        correct = 0
        for i_batch, (data, target) in enumerate(testloader):
            data, target = Variable(data), Variable(target)
            data, target = data.to(device), target.to(device)
            if _net == "MLP":
                data = data.view(-1, 3 * 32 * 32)
            net_out = self(data)
            # sum up batch loss
            test_loss += criterion(net_out, target).data.item()
            pred = net_out.data.max(1)[1]  # get the index of the max log-probability
            batch_labels = pred.eq(target.data)
            correct += batch_labels.sum()
        test_loss /= len(testloader.dataset)
        acc = 100. * float(correct) / len(testloader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(test_loss, correct,
                                                                                     len(testloader.dataset), acc))


class Net2(Net):
    """ Task 3c) """
    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=0, dilation=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride=1, padding=0, dilation=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0, dilation=1)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0, dilation=1)
        self.pool2= nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32*4*4, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        x = x.view(-1, 32*4*4)
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim=1)
        return x


class Net3(Net):
    """ Task 3c) """
    def __init__(self):
        super(Net3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2, dilation=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride=1, padding=2, dilation=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=1, stride=1, padding=0, dilation=1)
        self.bnc3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1)
        self.bnc4 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1, stride=1, padding=0, dilation=1)
        self.bnc5 = nn.BatchNorm2d(16)
        self.pool2= nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv6 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=1, stride=1, padding=0, dilation=1)
        self.bnc6 = nn.BatchNorm2d(64)
        self.conv7 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1)
        self.bnc7 = nn.BatchNorm2d(64)
        self.conv8 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=1, stride=1, padding=0, dilation=1)
        self.bnc8 = nn.BatchNorm2d(16)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(16*4*4, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        identity = x
        x = F.relu(self.conv3(x))
        x = self.bnc3(x)
        x = F.relu(self.conv4(x))
        x = self.bnc4(x)
        x = F.relu(self.conv5(x))
        x = self.bnc5(x)
        x += identity
        x = self.pool2(x)
        identity = x
        x = F.relu(self.conv6(x))
        x = self.bnc6(x)
        x = F.relu(self.conv7(x))
        x = self.bnc7(x)
        x = F.relu(self.conv8(x))
        x = self.bnc8(x)
        x += identity
        x = self.pool3(x)
        x = x.view(-1, 16*4*4)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim=1)
        return x


if __name__ == "__main__":

    flags = {"plot": False,
             "train": False,
             "test": True}

    transform = transforms.Compose([torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=4)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False, num_workers=4)
    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

    print("Es gibt {0} Trainingsbilder".format((trainloader.dataset.data.shape[0])))
    print("Es gibt {0} Testbilder".format((testloader.dataset.data.shape[0])))
    print("Es gibt {0} Klassen.".format(len(trainloader.dataset.classes)))
    print("Ein Bild hat {0}x{1} Pixel und {2} Features".format(trainloader.dataset.data.shape[1],
                                                               trainloader.dataset.data.shape[2],
                                                               trainloader.dataset.data.shape[1]*trainloader.dataset.data.shape[2]))
    for i in range(len(trainloader.dataset.classes)):
        print(f"Die Klasse {i} hat {(np.array(trainloader.dataset.targets) == i).sum()} Trainingsbilder"
              f" und {(np.array(testloader.dataset.targets) == i).sum()} Testbilder")

    if flags["plot"]:
        plot_data(trainloader)

    model_type = "MLP"

    if model_type == "MLP":
        device = "cpu"
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    if model_type == "MLP":
        model = Net()
    elif model_type == "CNN":
        model = Net2()
        model.to(device)
    elif model_type == "Res":
        model = Net3()
        model.to(device)
    print(model)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.NLLLoss()

    # train
    if flags["train"]:
        model.train(True)
        model.train_net(criterion, optimizer, trainloader, 10, _net=model_type)
        torch.save(model.state_dict(), f"net_{model_type}.pt")

    # test
    if flags["test"]:
        model.load_state_dict(torch.load(f"net_{model_type}.pt"))
        model.eval()
        model.test_net(criterion, testloader, _net=model_type)