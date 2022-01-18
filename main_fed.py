#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import random

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar, ResNet18, LeNet5
from models.Fed import FedAvg, FedSGD
from models.test import test_img


if __name__ == '__main__':
    # parse args
    np.random.seed(1)
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users
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
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
        net_tmp = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
        net_tmp = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
        net_tmp = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    elif args.model == 'resnet':
        net_glob = ResNet18().to(args.device)
        net_tmp = ResNet18().to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    print(net_tmp)

    net_glob.train()
    # surrogate model
    #net_surrogate = LeNet5(args).to(args.device)
    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []
    if args.all_clients:
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]
    if args.atk_num == 1:
        atk_client = [int(0)]
    elif args.atk_num == 0:
        atk_client = [int(1111)]
    else:
        atk_client = random.sample(range(0,10), args.atk_num)


    if args.a_method == 'fedsgd':
        for iter in range(args.epochs):
            loss_locals = []
            grad_info = []
            if not args.all_clients:
                w_locals = []
            m = max(int(args.frac * args.num_users), 1)
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)
            # choose some clients to order
            #n = max(int(args.frc_order * args.num_users), 1)
            #idxs_order_users = np.random.choice(range(args.num_users), len(idxs_users), replace=False)
            num = 0
            for idx in idxs_users:
                if num in atk_client and iter>0:
                    local = LocalUpdate(args=args, attack_state=False, net=copy.deepcopy(net_glob).to(args.device),
                                        dataset=dataset_train, idxs=dict_users[idx])
                else:
                    local = LocalUpdate(args=args, attack_state=False, dataset=dataset_train, idxs=dict_users[idx])
                grads, loss = local.train_grad(net=copy.deepcopy(net_glob).to(args.device))
                grad_info.append(grads)
                w = local.eval_local_model(net=copy.deepcopy(net_glob).to(args.device), grad=grads)
                net_tmp.load_state_dict(w)
                acc_test, loss_test = test_img(net_tmp, dataset_test, args)
                print('{}, local acc: {:.2f}, local loss: {:.2f}'.format(num, acc_test,loss_test))
                num = num + 1
                loss_locals.append(copy.deepcopy(loss))
            # update global gradients
            grads_global = FedSGD(grad_info)
            # copy weight to net_glob
            if args.optimizer == 'sgd':
                optimizer = torch.optim.SGD(net_glob.parameters(), lr=args.lr, momentum=args.momentum)
            elif args.optimizer == 'adam':
                optimizer = torch.optim.Adam(net_glob.parameters(), lr=args.lr, betas=(0.99,0.9))
            else:
                print('Error: There is no such optimizer ! ! !')
            net_glob.train()
            optimizer.zero_grad()
            for k, v in net_glob.named_parameters():
                v.grad = grads_global[k]
            optimizer.step()

            # print loss
            loss_avg = sum(loss_locals) / len(loss_locals)
            print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
            loss_train.append(loss_avg)
            # testing
            net_glob.eval()
            acc_train, _ = test_img(net_glob, dataset_train, args)
            acc_test, loss_test = test_img(net_glob, dataset_test, args)
            print("Training accuracy: {:.2f}".format(acc_train))
            print("Training accuracy: {:.2f}".format(_))
            print("Testing accuracy: {:.2f}".format(acc_test))
            print("Testing loss: {:.2f}".format(loss_test))
    elif args.a_method == 'fedavg':
        grads_local = []
        for iter in range(args.epochs):
            loss_locals = []
            if not args.all_clients:
                w_locals = []
            m = max(int(args.frac * args.num_users), 1)
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)
            # choose some clients to order
            # n = max(int(args.frc_order * args.num_users), 1)
            # idxs_order_users = np.random.choice(range(args.num_users), len(idxs_users), replace=False)
            num = 0
            for idx in idxs_users:
                if num in atk_client and iter>0:
                    print("begin to attack ! ! !")
                    local = LocalUpdate(args=args, attack_state=True,
                                        net=copy.deepcopy(net_glob).to(args.device),
                                        dataset=dataset_train, idxs=dict_users[idx],grads_global=grads_global)
                else:
                    local = LocalUpdate(args=args, attack_state=False, dataset=dataset_train, idxs=dict_users[idx])
                w, grad,loss = local.train(net=copy.deepcopy(net_glob).to(args.device), surrogate=False)
                net_tmp.load_state_dict(w)
                net_tmp.eval()
                acc_test, loss_test = test_img(net_tmp, dataset_test, args)
                print('{}, local acc: {:.2f}, local loss: {:.2f}'.format(num, acc_test, loss_test))
                num = num + 1
                if args.all_clients:
                    w_locals[idx] = copy.deepcopy(w)
                    grads_local.append(grad)
                else:
                    w_locals.append(copy.deepcopy(w))
                    grads_local.append(grad)
                loss_locals.append(copy.deepcopy(loss))
            # update global weights
            w_glob = FedAvg(w_locals)
            grads_global = FedSGD(grads_local)
            #print(grads_global)
            # copy weight to net_glob
            net_glob.load_state_dict(w_glob)
            # print loss
            loss_avg = sum(loss_locals) / len(loss_locals)
            print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
            loss_train.append(loss_avg)
            # testing
            net_glob.eval()
            acc_train, _ = test_img(net_glob, dataset_train, args)
            acc_test, loss_test = test_img(net_glob, dataset_test, args)
            print("Training accuracy: {:.2f}".format(acc_train))
            print("Training accuracy: {:.2f}".format(_))
            print("Testing accuracy: {:.2f}".format(acc_test))
            print("Testing loss: {:.2f}".format(loss_test))
    else:
        print('Error: There is no such aggregation ! ! !')