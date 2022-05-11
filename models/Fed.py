#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
import numpy as np
from torch import nn
from collections import defaultdict


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def FedSGD(grads_info):
    total_grads = {}
    n_total_samples = 0
    for info in grads_info:
        n_samples = info['n_samples']
        for k, v in info['named_grads'].items():
            if k not in total_grads:
                total_grads[k] = v
            total_grads[k] += v * n_samples
        n_total_samples += n_samples
    gradients = {}
    for k, v in total_grads.items():
        gradients[k] = torch.div(v, n_total_samples)
    return gradients

def trimmed_mean(w, args):
    number_to_consider = int((args.num_users - args.atk_num) * args.frac) - 1
    print(number_to_consider)
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        tmp = []
        for i in range(len(w)):
            tmp.append(w[i][k].cpu().numpy()) # get the weight of k-layer which in each client
        tmp = np.array(tmp)
        med = np.median(tmp,axis=0)
        new_tmp = []
        for i in range(len(tmp)):# cal each client weights - median
            new_tmp.append(tmp[i]-med)
        new_tmp = np.array(new_tmp)
        good_vals = np.argsort(abs(new_tmp),axis=0)[:number_to_consider]
        good_vals = np.take_along_axis(new_tmp, good_vals, axis=0)
        k_weight = np.array(np.mean(good_vals) + med)
        w_avg[k] = torch.from_numpy(k_weight).to(args.device)
    return w_avg

def krum(w, args):
    distances = defaultdict(dict)
    non_malicious_count = int((args.num_users - args.atk_num) * args.frac)
    num = 0
    for k in w[0].keys():
        if num == 0:
            for i in range(len(w)):
                for j in range(i):
                    distances[i][j] = distances[j][i] = np.linalg.norm(w[i][k].cpu().numpy() - w[j][k].cpu().numpy())
            num = 1
        else:
            for i in range(len(w)):
                for j in range(i):
                    distances[j][i] += np.linalg.norm(w[i][k].cpu().numpy() - w[j][k].cpu().numpy())
                    distances[i][j] += distances[j][i]
    minimal_error = 1e20
    for user in distances.keys():
        errors = sorted(distances[user].values())
        current_error = sum(errors[:non_malicious_count])
        if current_error < minimal_error:
            print(user)
            minimal_error = current_error
            minimal_error_index = user
    return w[minimal_error_index]

def fltrust(w, w_global, args):
    n = len(w)
    baseline = w_global
    cos_sim = []
    for param in w:
        cos_tmp = 0
        for k in param.keys():
            cos_tmp += torch.dot(torch.flatten(baseline[k], end_dim=-1).float(),
                                 torch.flatten(param[k], end_dim=-1).float()) / (
                                   torch.norm(torch.flatten(baseline[k], end_dim=-1).float()) + 1e-9) / (
                                   torch.norm(torch.flatten(param[k], end_dim=-1).float()) + 1e-9)
        cos_sim.append(cos_tmp)
    good_result = []
    for i in cos_sim:
        if i > 0:  # relu
            good_result.append(i.cpu().numpy())
        else:
            good_result.append(0.)
    good_result = np.array(good_result)
    print(good_result)
    if np.sum(good_result) == 0:
        normalized_weights = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    else:
        normalized_weights = good_result / (np.sum(good_result) + 1e-9)
    print(normalized_weights)
    new_param_list = []
    for k in w[0].keys():
        tmp = []
        for i in range(n):  # each client
            # print(torch.mul(w[i][k],normalized_weights[i]))
            '''
            tmp.append(torch.mul(torch.div(torch.mul(w[i][k], normalized_weights[i])
                                           , (torch.norm(torch.flatten(w[i][k], end_dim=-1).float()) + 1e-9))
                                 , torch.norm(torch.flatten(baseline[k], end_dim=-1).float())))
            '''
            tmp.append(torch.mul(w[i][k], normalized_weights[i]))
        new_param_list.append(sum(tmp))
    w_avg = copy.deepcopy(w[0])
    num = 0
    for k in w_avg.keys():
        w_avg[k] = new_param_list[num].to(args.device)
        num += 1
    return w_avg
