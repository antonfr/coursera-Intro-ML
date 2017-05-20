#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 15:04:53 2017

@author: antonperhunov
"""

import pandas as pd
import numpy as np
import math
import scipy
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import log_loss

df = pd.read_csv('gbm-data.csv', header = 0)
X = df.iloc[:,1:].values
y = df.iloc[:,0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.8, random_state = 241)

def sigmoid(y_pred):
    return (1 + math.exp(-y_pred))**(-1)

def loss(X, y, classifier):
    results = []
    for value in classifier.staged_decision_function(X):
        results.append(log_loss(y, [sigmoid(y_pred) for y_pred in value]))
    return results

def loss_plot(rate, loss_train, loss_test):
    plt.figure()
    plt.plot(loss_train, 'r', linewidth = 2)
    plt.plot(loss_test, 'g', linewidth = 2)
    plt.legend(['train', 'test'])
    plt.show()
    
    min_value = min(loss_test)
    min_index = loss_test.index(min_value)
    return min_value, min_index

def gb_class(lr):
    classifier = GradientBoostingClassifier(n_estimators = 250, learning_rate = lr, verbose = True, random_state = 241)
    classifier.fit(X_train, y_train)
    
    loss_train = loss(X_train, y_train, classifier)
    loss_test = loss(X_test, y_test, classifier)
    return loss_plot(lr, loss_train, loss_test)
    
min_rate = {}
l_rate = [1, 0.5, 0.3, 0.2, 0.1]

for rate in l_rate : 
    min_rate[rate] = gb_class(rate)
    
rf_classifier = RandomForestClassifier(n_estimators = 37, random_state = 241)
rf_classifier.fit(X_train, y_train)
y_pred = rf_classifier.predict_proba(X_test)
print(log_loss(y_test, y_pred))
