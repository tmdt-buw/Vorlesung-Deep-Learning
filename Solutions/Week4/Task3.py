import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


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
    def train_net(self, criterion, optimizer, _trainloader, epochs):
        log_interval = 10
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(_trainloader):
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
                                                                                   _trainloader.batch_sampler.sampler.indices.size,
                                                                                   100. * batch_idx / len(_trainloader),
                                                                                   loss.data.item()))

    def test_net(self, criterion, _testloader):
        # test the net
        test_loss = 0
        correct = 0
        correct_class = np.zeros(10)
        for i_batch, (data, target) in enumerate(_testloader):
            data, target = Variable(data), Variable(target)
            data = data.view(-1, 28 * 28)
            net_out = self(data)
            # sum up batch loss
            test_loss += criterion(net_out, target).data.item()
            pred = net_out.data.max(1)[1]  # get the index of the max log-probability
            batch_labels = pred.eq(target.data)
            correct += batch_labels.sum()
            for i_label in range(len(target)):
                label = target[i_label].item()
                correct_class[label] += batch_labels[i_label].item()
        test_loss /= len(_testloader.dataset)
        acc = 100. * float(correct) / len(_testloader.dataset)
        acc_class = np.zeros(10)
        for i_label in range(10):
            num = (_testloader.dataset.targets.numpy() == i_label).sum()
            acc_class[i_label] = correct_class[
                                     i_label] / num if num > 0 else 1.0  # if there is no data on the class, we replace the accuracy by 1
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(test_loss, correct,
                                                                                     len(_testloader.dataset), acc))
        return acc, acc_class


def plot_accs(fractions, accs, accs_class):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    ax.plot(fractions, accs, marker='s')
    for i in range(10):
        ax2.plot(fractions, accs_class[:, i], marker='s', label=f'{i}')
    ax2.axhline(y=0.9, lw=2, ls='--', c='r')

    for ax in [ax, ax2]:
        ax.invert_xaxis()
        ax.set_xticks(fractions)
        ax.set_xlim(left=1.1)

    ax2.legend(loc=2)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    """ Task 3a) """
    transform = transforms.Compose([torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize((0.5,), (0.5,))])
    trainset = torchvision.datasets.MNIST(root='../Week3/data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4)
    testset = torchvision.datasets.MNIST(root='../Week3/data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=4)

    """ Task 3a+b) """
    flags = {"train": True,
             "test": True,
             "plot": True}

    fractions = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    num_train_samples = trainloader.dataset.data.size()[0]
    if flags["train"]:
        for fraction in fractions:
            reduced_train_size = int(fraction*num_train_samples)
            sample_idx_list = np.random.choice(np.arange(num_train_samples), size=reduced_train_size, replace=False)
            batch_sampler = torch.utils.data.sampler.SubsetRandomSampler(sample_idx_list)
            trainloader_reduced = torch.utils.data.DataLoader(trainset, sampler=batch_sampler, batch_size=64, num_workers=4)

            model = Net()
            print(model)
            quit()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
            criterion = nn.NLLLoss()
            # train
            model.train_net(criterion, optimizer, trainloader_reduced, 10)
            torch.save(model.state_dict(), f"net_{fraction}.pt")

    if flags["test"]:
        accs = []
        accs_class = []
        for fraction in fractions:
            # test
            model = Net()
            model.load_state_dict(torch.load(f'net_{fraction}.pt'))
            model.eval()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
            criterion = nn.NLLLoss()
            acc, acc_class = model.test_net(criterion, testloader)
            accs.append(acc)
            accs_class.append(acc_class)

        np.savetxt('accs.txt', accs)
        np.savetxt('accs_class.txt', accs_class)

    if flags["plot"]:
        accs = np.loadtxt("accs.txt")
        accs_class = np.loadtxt("accs_class.txt")
        plot_accs(fractions, accs, accs_class)
    quit()

    """ Task 3c) """
    flags = {"train": True,
             "test": True}

    np.random.seed(1)
    torch.manual_seed(1)

    num_train_samples = trainloader.dataset.data.size()[0]
    class_train_sizes = [0.08, 0.08, 0.4, 0.4, 0.4, 0.55, 0.2, 0.3, 0.55, 0.4]
    sample_idx_list = []
    for i in range(10):
        class_idx_list = np.squeeze(np.argwhere(trainloader.dataset.targets.numpy() == i))
        sample_idx_list += np.random.choice(class_idx_list, size=int(class_train_sizes[i]*len(class_idx_list)), replace=False).tolist()
    batch_sampler = torch.utils.data.sampler.SubsetRandomSampler(np.array(sample_idx_list))
    trainloader_reduced = torch.utils.data.DataLoader(trainset, sampler=batch_sampler, batch_size=64, num_workers=4)

    if flags["train"]:
        model = Net()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
        criterion = nn.NLLLoss()
        # train
        model.train_net(criterion, optimizer, trainloader_reduced, 10)
        torch.save(model.state_dict(), "net_custom.pt")

    if flags["test"]:
        model = Net()
        model.load_state_dict(torch.load('net_custom.pt'))
        model.eval()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
        criterion = nn.NLLLoss()
        acc, acc_class = model.test_net(criterion, testloader)
        print(acc)
        print(acc_class)
