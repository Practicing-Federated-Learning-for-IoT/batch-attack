#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics
from models.test import test_img
import json


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, attack_state, net=None, dataset=None, idxs=None):
        self.args = args
        self.dataset = dataset
        self.idxs = idxs
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=False)
        if attack_state and self.args.attack_type=='reorder':
            idx = self.reorder_by_loss(net=net, attack_type=self.args.attack_type, ATK=self.args.atk)
            reorder_idxs = []
            idxs = list(idxs)
            for i in idx:
                reorder_idxs.extend(idxs[i*self.args.local_bs:i*self.args.local_bs+self.args.local_bs])
            self.ldr_train = DataLoader(DatasetSplit(dataset, reorder_idxs), batch_size=self.args.local_bs, shuffle=False)
        elif attack_state and self.args.attack_type == 'reshuffle':
            idx = self.reorder_by_loss(net=net, attack_type=self.args.attack_type, ATK=self.args.atk)
            self.ldr_train = DataLoader(DatasetSplit(dataset, idx), batch_size=self.args.local_bs, shuffle=False)


    def train(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def reorder_by_loss(self, net, attack_type, ATK):
        if attack_type == 'reorder':
            #net.eval()
            batch_losses = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                batch_losses.append(loss.item())
            if ATK == 'oscillating out':
                sort_idxs = np.argsort(batch_losses)  # low->high
                sort_idxs = list(sort_idxs)
                left = sort_idxs[:len(sort_idxs)//2][::-1]
                right = sort_idxs[len(sort_idxs)//2:][::-1]
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
                while len(sort_idxs)>0:
                    osc = not osc
                    if osc:
                        new_idxs.append(sort_idxs[-1])
                        sort_idxs = sort_idxs[:-1]
                    else:
                        new_idxs.append(sort_idxs[0])
                        sort_idxs = sort_idxs[1:]
                return new_idxs
        elif attack_type == 'reshuffle':
            data_losses = []
            for i in self.idxs:
                image, label = self.dataset[i]
                image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
                label = torch.tensor([label])
                image, label = image.to(self.args.device), label.to(self.args.device)
                log_prob = net(image)
                loss = self.loss_func(log_prob, label)
                data_losses.append(loss.item())
            if ATK == 'oscillating out':
                sort_idxs = np.argsort(data_losses)  # low->high
                sort_idxs = list(sort_idxs)
                left = sort_idxs[:len(sort_idxs)//2][::-1]
                right = sort_idxs[len(sort_idxs)//2:][::-1]
                new_idxs = []
                new_idxs.extend(left)
                new_idxs.extend(right)
                sort_idxs = new_idxs
            osc = False
            if ATK == 'lowhigh':
                sort_idxs = np.argsort(data_losses)  # low->high
                return list(sort_idxs)
            elif ATK == 'highlow':
                sort_idxs = np.argsort(data_losses)  # low->high
                return list(sort_idxs[::-1])
            elif ATK == 'oscillating in' or 'oscillating out':
                if ATK == 'oscillating in':
                    sort_idxs = np.argsort(data_losses)
                    sort_idxs = list(sort_idxs)
                new_idxs = []
                while len(sort_idxs)>0:
                    osc = not osc
                    if osc:
                        new_idxs.append(sort_idxs[-1])
                        sort_idxs = sort_idxs[:-1]
                    else:
                        new_idxs.append(sort_idxs[0])
                        sort_idxs = sort_idxs[1:]
                return new_idxs



    def ordering_methods(self, idx):
        batch_size = self.args.local_bs
        tmp = idx[:batch_size]
        idx[:batch_size] = idx[-batch_size:]
        idx[-batch_size:] = tmp
        return idx