#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 15:27:33 2017

@author: antonperhunov
"""

import pandas as pd
import numpy as np
import math
from sklearn.metrics import roc_auc_score

data = pd.read_csv('data-logistic.csv', header = None)
X = data.iloc[:,1:].values
y = data.iloc[:,0].values

def grad_step(X, y, w1, w2, k, C):
    S1, S2 = 0, 0
    l = len(y)
    for i in range(l):
        S1 += y[i] * X[i, 0] * (1 - (1 + math.exp(-y[i] * (w1 * X[i, 0] + w2 * X[i, 1])))**(-1))
        S2 += y[i] * X[i, 1] * (1 - (1 + math.exp(-y[i] * (w1 * X[i, 0] + w2 * X[i, 1])))**(-1))
    w1 = w1 + k * S1 / l - k * C * w1
    w2 = w2 + k * S2 / l - k * C * w2
    return w1, w2

def grad(X, y, w1 = 0, w2 = 0, k = 0.1, C = 0, eps = 1e-5):
    i = 1
    error = 1
    w1_new, w2_new = w1, w2
    while i <= 10000 or error >= eps:
        w1_new, w2_new = grad_step(X, y, w1, w2, k, C)
        error = math.sqrt((w1_new - w1)**2 + (w2_new - w2)**2)
        w1, w2 = w1_new, w2_new
        i += 1
    print(i, error)
    return w1, w2

w1, w2 = grad(X, y)
rw1, rw2 = grad(X, y, C = 10)

y_pred = []
y_rpred = []
for i in range(len(y)):
    pred = (1 + math.exp(-w1 * X[i, 0] - w2 * X[i, 1]))**(-1)
    rpred = (1 + math.exp(-rw1 * X[i, 0] - rw2 * X[i, 1]))**(-1)
    y_pred.append(pred)
    y_rpred.append(rpred)

auc = roc_auc_score(y, y_pred)
aucr = roc_auc_score(y, y_rpred)


