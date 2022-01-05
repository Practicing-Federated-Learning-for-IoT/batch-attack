import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img
import torch.nn.functional as F

np.random.seed(1)
args = args_parser()
args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

if args.dataset == 'mnist':
    trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
    dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
    # sample users
    if args.iid:
        dict_users = mnist_iid(dataset_train, args.num_users)
    else:
        dict_users = mnist_noniid(dataset_train, args.num_users)
elif args.dataset == 'cifar':
    trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
    dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
else:
    exit('Error: unrecognized dataset')
img_size = dataset_train[0][0].shape


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)



    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

net_glob = CNNCifar(args=args).to(args.device)


idxs = [i for i in range(len(dataset_train))]
ldr_train = DataLoader(DatasetSplit(dataset_train, idxs), batch_size=args.local_bs, shuffle=False)
loss_func = nn.CrossEntropyLoss()

def reorder_by_loss(net, attack_type, ATK):
    if attack_type == 'reorder':
        # net.eval()
        batch_losses = []
        for batch_idx, (images, labels) in enumerate(ldr_train):
            images, labels = images.to(args.device), labels.to(args.device)
            log_probs = net(images)
            loss = loss_func(log_probs, labels)
            batch_losses.append(loss.item())
        if ATK == 'oscillating out':
            sort_idxs = np.argsort(batch_losses)  # low->high
            sort_idxs = list(sort_idxs)
            left = sort_idxs[:len(sort_idxs) // 2][::-1]
            right = sort_idxs[len(sort_idxs) // 2:][::-1]
            new_idxs = []
            new_idxs.extend(left)
            new_idxs.extend(right)
            sort_idxs = new_idxs
        osc = False
        if ATK == 'lowhigh':
            sort_idxs = np.argsort(batch_losses)  # low->high
            return list(sort_idxs)
        elif ATK == 'highlow':
            sort_idxs = np.argsort(batch_losses)  # low->high
            return list(sort_idxs[::-1])
        elif ATK == 'oscillating in' or 'oscillating out':
            if ATK == 'oscillating in':
                sort_idxs = np.argsort(batch_losses)
                sort_idxs = list(sort_idxs)
            new_idxs = []
            while len(sort_idxs) > 0:
                osc = not osc
                if osc:
                    new_idxs.append(sort_idxs[-1])
                    sort_idxs = sort_idxs[:-1]
                else:
                    new_idxs.append(sort_idxs[0])
                    sort_idxs = sort_idxs[1:]
            return new_idxs
idx = reorder_by_loss(net_glob,'reorder','lowhigh')
reorder_idxs = []
idxs = list(idxs)
for i in idx:
    reorder_idxs.extend(idxs[i*args.local_bs:i*args.local_bs+args.local_bs])
ldr_train = DataLoader(DatasetSplit(dataset_train, reorder_idxs), batch_size=args.local_bs, shuffle=False)


optimizer = torch.optim.SGD(net_glob.parameters(), lr=args.lr, momentum=args.momentum)

def test_img(net_g, datatest, args):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs)
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.cuda(), target.cuda()
        log_probs = net_g(data)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss

for epoch in range(10):
    #train
    net_glob.train()
    if epoch == 4:
        print('begin to attack')
        idx = reorder_by_loss(net_glob, 'reorder', 'lowhigh')
        reorder_idxs = []
        idxs = list(idxs)
        for i in idx:
            reorder_idxs.extend(idxs[i * args.local_bs:i * args.local_bs + args.local_bs])
        ldr_train = DataLoader(DatasetSplit(dataset_train, reorder_idxs), batch_size=args.local_bs, shuffle=False)
    else:
        ldr_train = DataLoader(DatasetSplit(dataset_train, reorder_idxs), batch_size=args.local_bs, shuffle=False)
    for iter in range(args.local_ep):
        batch_loss = []
        for batch_idx, (images, labels) in enumerate(ldr_train):
            images, labels = images.to(args.device), labels.to(args.device)
            net_glob.zero_grad()
            log_probs = net_glob(images)
            loss = loss_func(log_probs, labels)
            loss.backward()
            optimizer.step()
    #test
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))



