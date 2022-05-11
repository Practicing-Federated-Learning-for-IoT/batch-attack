#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
from models.Nets import LeNet5
import random
from sklearn import metrics
from models.test import test_img
import json
import copy


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
    def __init__(self, args, attack_state, net=None, dataset=None, idxs=None, grads_global=None):
        self.args = args
        self.dataset = dataset
        self.idxs = idxs
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=False)
        if attack_state and self.args.attack_type=='reorder':
            idx = self.reorder_by_loss(net=net, attack_type=self.args.attack_type, ATK=self.args.atk,grads_global=grads_global)
            reorder_idxs = []
            idxs = list(idxs)
            #print(idx)
            worse_idx = idx[:1]
            #for i in idx:
            for i in range(len(idx)):
                k = i%len(worse_idx)
                reorder_idxs.extend(idxs[worse_idx[k]*self.args.local_bs:worse_idx[k]*self.args.local_bs+self.args.local_bs])
                #print(reorder_idxs)
                # 1.14 change
                #idx = reorder_idxs
                #self.idxs = idx
            self.ldr_train = DataLoader(DatasetSplit(dataset, reorder_idxs), batch_size=self.args.local_bs, shuffle=False)
        elif attack_state and self.args.attack_type == 'reshuffle':
            idx = self.reorder_by_loss(net=net, attack_type=self.args.attack_type, ATK=self.args.atk,grads_global=grads_global)
            # self.idxs = idx
            tmp = idx[:1 * self.args.local_bs]
            idxs = []
            for i in range(len(idx)):
                idxs.append(tmp[i % len(tmp)])
            self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=False)


    def train(self, net, surrogate):
        net.train()
        # train and update
        if surrogate:
            if self.args.optimizer == 'sgd':
                optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr_s_model, momentum=self.args.momentum)
            elif self.args.optimizer == 'adam':
                optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr_s_model, betas=(0.99,0.9))
            else:
                print('Error: There is no such optimizer ! ! !')
        else:
            if self.args.optimizer == 'sgd':
                optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
            elif self.args.optimizer == 'adam':
                optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr, betas=(0.99,0.9))
            else:
                print('Error: There is no such optimizer ! ! !')
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
        grads = {'n_samples': len(self.idxs), 'named_grads': {}}
        for name, param in net.named_parameters():
            if param.grad!=None:
                grads['named_grads'][name] = param.grad
        return net.state_dict(), grads, sum(epoch_loss) / len(epoch_loss)

    def train_grad(self, net):
        net.train()
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            bs = self.args.local_bs
            for i in range(int(len(self.idxs)/bs)-1):
                train_x_batch = []
                train_y_batch = []
                for k in range(bs):
                    image, label = self.dataset[list(self.idxs)[i*bs+k]]
                    train_x_batch.append(image)
                    train_y_batch.append(label)
                images = torch.stack(train_x_batch, 0)
                print(images.shape)
                labels = torch.tensor(train_y_batch)
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        grads = {'n_samples': len(self.idxs), 'named_grads': {}}
        for name, param in net.named_parameters():
            grads['named_grads'][name] = param.grad
        return grads, sum(epoch_loss) / len(epoch_loss)

    def eval_local_model(self, net, grad):
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        net.train()
        optimizer.zero_grad()
        for k, v in net.named_parameters():
            v.grad = grad['named_grads'][k]
        optimizer.step()
        return net.state_dict()

    def get_similarity(self,name_list,grads_o,grads_c):
        s_list = []
        for name in name_list:
            sim = torch.cosine_similarity(grads_o[name], grads_c['named_grads'][name], dim=0)
            shape = 1
            for i in sim.shape:
                shape = shape * i
            similarity = torch.div(torch.sum(sim),shape)
            s_list.append(similarity)
        sum_all_similarity = sum(s_list)
        return torch.div(sum_all_similarity,len(name_list))



    def reorder_by_loss(self, net, attack_type, ATK, grads_global):
        if attack_type == 'reorder':
            #net.eval()
            batch_similar = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                net_temp = copy.deepcopy(net)
                #if self.args.optimizer == 'sgd':
                #    optimizer = torch.optim.SGD(net_temp.parameters(), lr=self.args.lr,
                #                               momentum=self.args.momentum)  # ,weight_decay=self.args.weight_decay
                #else:
                #    optimizer = torch.optim.Adam(net_temp.parameters(), lr=self.args.lr, betas=(0.99, 0.9))
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net_temp.zero_grad()
                log_probs = net_temp(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                #optimizer.step()
                grads_change = {'named_grads': {}}
                name_list = []
                for name, param in net_temp.named_parameters():
                    if param.grad != None:
                        name_list.append(name)
                        grads_change['named_grads'][name] = param.grad
                #print(grads_change)

                batch_similar.append(self.get_similarity(name_list, grads_global, grads_change).item())
            #print(batch_similar)
            if ATK == 'oscillating_out':
                sort_idxs = np.argsort(batch_similar)  # low->high
                sort_idxs = list(sort_idxs)
                left = sort_idxs[:len(sort_idxs)//2][::-1]
                right = sort_idxs[len(sort_idxs)//2:][::-1]
                new_idxs = []
                new_idxs.extend(left)
                new_idxs.extend(right)
                sort_idxs = new_idxs
            osc = False
            if ATK == 'lowhigh':
                sort_idxs = np.argsort(batch_similar)  # low->high
                return list(sort_idxs)
            elif ATK == 'highlow':
                sort_idxs = np.argsort(batch_similar)  # low->high
                return list(sort_idxs[::-1])
            elif ATK == 'oscillating_in' or 'oscillating_out':
                if ATK == 'oscillating_in':
                    sort_idxs = np.argsort(batch_similar)
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
            data_similar = []
            for i in self.idxs:
                net_temp = copy.deepcopy(net)
                image, label = self.dataset[i]
                image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
                label = torch.tensor([label])
                image, label = image.to(self.args.device), label.to(self.args.device)
                log_prob = net_temp(image)
                net_temp.zero_grad()
                loss = self.loss_func(log_prob, label)
                loss.backward()
                grads_change = {'named_grads': {}}
                name_list = []
                for name, param in net_temp.named_parameters():
                    if param.grad != None:
                        name_list.append(name)
                        grads_change['named_grads'][name] = param.grad
                # print(grads_change)
                data_similar.append(self.get_similarity(name_list, grads_global, grads_change).item())
            if ATK == 'oscillating_out':
                sort_idxs = np.argsort(data_similar)  # low->high
                sort_idxs = list(sort_idxs)
                left = sort_idxs[:len(sort_idxs)//2][::-1]
                right = sort_idxs[len(sort_idxs)//2:][::-1]
                new_idxs = []
                new_idxs.extend(left)
                new_idxs.extend(right)
                sort_idxs = new_idxs
            osc = False
            if ATK == 'lowhigh':
                sort_idxs = np.argsort(data_similar)  # low->high
                return list(sort_idxs)
            elif ATK == 'highlow':
                sort_idxs = np.argsort(data_similar)  # low->high
                return list(sort_idxs[::-1])
            elif ATK == 'oscillating_in' or 'oscillating_out':
                if ATK == 'oscillating_in':
                    sort_idxs = np.argsort(data_similar)
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
