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
    idx = np.random.randint(low=0, high=trainloader.dataset.data.shape[0], size=100)
    fig = plt.figure(figsize=(10, 10))
    for i, i_idx in enumerate(idx):
        ax = fig.add_subplot(10, 10, i + 1)
        ax.axis('off')
        ax.imshow(trainloader.dataset.data[i_idx].numpy().reshape(28, 28), cmap="gray")
        ax.set_title('Class: ' + str(trainloader.dataset.targets[i_idx].item()))
    plt.tight_layout()
    plt.show()


class Net(nn.Module):
    """ Task 3c) """
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 50)
        self.fc2 = nn.Linear(50, 25)
        self.fc3 = nn.Linear(25, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = F.log_softmax(self.fc3(x), dim=1)
        return out

    """ Task 3d) """
    def train_net(self, criterion, optimizer, trainloader, epochs):
        log_interval = 10
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(trainloader):
                data, target = Variable(data), Variable(target)
                # resize data from (batch_size, 1, 28, 28) to (batch_size, 28*28)
                data = data.view(-1, 28 * 28)
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

    def test_net(self, criterion, testloader):
        # test the net
        test_loss = 0
        correct = 0
        for i_batch, (data, target) in enumerate(testloader):
            data, target = Variable(data), Variable(target)
            data = data.view(-1, 28 * 28)
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


if __name__ == "__main__":

    """ Task 3a) """
    transform = transforms.Compose([torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize((0.5,), (0.5,))])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=4)
    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

    print("Es gibt {0} Trainingsbilder".format((trainloader.dataset.data.size()[0])))
    print("Es gibt {0} Testbilder".format((testloader.dataset.data.size()[0])))
    print("Es gibt {0} Klassen.".format(len(np.unique(trainloader.dataset.targets.numpy()))))
    print("Ein Bild hat {0}x{1} Pixel und {2} Features".format(trainloader.dataset.data.size()[1],
                                                               trainloader.dataset.data.size()[2],
                                                               trainloader.dataset.data.size()[1]*trainloader.dataset.data.size()[2]))
    for i in np.unique(trainloader.dataset.targets.numpy()):
        print("Die Klasse {0} hat {1} Trainingsbilder und {2} Testbilder".format(i,
                                                                                 (trainloader.dataset.targets == i).sum(),
                                                                                 (testloader.dataset.targets == i).sum()))

    """ Task 3b) """
    plot_data(trainloader)

    """ Task 3c) """
    model = Net()
    print(model)

    """ Task 3d) """
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    criterion = nn.NLLLoss()

    # train
    # model.train_net(criterion, optimizer, trainloader, 10)
    # torch.save(model.state_dict(), "net.pt")

    # test
    model.load_state_dict(torch.load('net.pt'))
    model.eval()
    model.test_net(criterion, testloader)