#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn


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