#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 19:33:29 2017
This is the final project of Coursera's "Introduction to machine learning"
course offered by HSE. In this project we have to predict a winner in Dota 2
game after 5 minutes of match, using gradient boosting classification and 
logistic regression. Training dataset consists of 97230 observations and 108 
features, we will use 102 features to build a classifier because 6 of them 
contain the information, when the match was ended. Test dataset contains 17177
observations and 102 features without information about a winner, so we can't 
calculate ROC AUC score as quality metrics on test dataset, therefore we will 
use cross validation on training set, dividing it randomly into five subsets.
"""

#importation of necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

#importation of necessary csv files
train = pd.read_csv('features.csv', header = 0, index_col = 'match_id')
test = pd.read_csv('features_test.csv', header = 0, index_col = 'match_id')
all_heroes = pd.read_csv('heroes.csv', header = 0, index_col = 'id')

#choice of features and dealing with missing values
train = train.iloc[:,0:104]
train = train.drop('duration', axis = 1)
na_var = round((len(train) - train.describe().T['count'])/len(train), 5)
na_var = na_var[na_var > 0]
print(na_var)
train = train.fillna(0)
test = test.fillna(0)

#creation of matrix of features and vector of classes in np.array form
X_train = train.iloc[:,:-1].values
y_train = train.iloc[:,-1].values

#cross validation splits
kf = KFold(n_splits = 5, shuffle = True, random_state = 147)

#Gradient boosting classification and choice of optimal number of trees
trees = [10, 20, 30, 50, 100, 147]
auc_roc_gb = []
for n_trees in trees:
    start = datetime.datetime.now()
    gb_classifier = GradientBoostingClassifier(n_estimators = n_trees, random_state = 147)
    auc_roc_gb_score = cross_val_score(gb_classifier, X_train, y_train, cv = kf, scoring = 'roc_auc', n_jobs = -1)
    mean_auc_gb = np.mean(auc_roc_gb_score)
    auc_roc_gb.append(mean_auc_gb)
    finish = datetime.datetime.now() 
    print('Gradient boosting AUC-ROC score is ' + str(round(mean_auc_gb, 5)) + ' for ' + str(n_trees) + ' trees, time elapsed: ' + str(finish - start))

#Gradient boosting number of trees visualization
plt.plot(trees, auc_roc_gb)
plt.xlabel('Number of trees')
plt.ylabel('AUC-ROC score')
plt.title('Gradient Boosting')
plt.show()    

#-------------------------Logistic regressions---------------------------------

#------functions for logistic regression
#remove categorical features from pandas.DataFrame X
def remove_cf(X):
    X = X.drop('lobby_type', axis = 1)
    for i in range(1, 6):
        X = X.drop('r{}_hero'.format(i), axis = 1)
        X = X.drop('d{}_hero'.format(i), axis = 1)
    return X

#building np.array heroes according to pandas.DataFrame X
def create_heroes(X):
    n_all_heroes = len(all_heroes)
    heroes = np.zeros((X.shape[0], n_all_heroes))
    for i, match_id in enumerate(X.index):
        for p in range(5):
            heroes[i, X.ix[match_id, 'r%d_hero' % (p+1)]-1] = 1
            heroes[i, X.ix[match_id, 'd%d_hero' % (p+1)]-1] = -1
    return heroes

#logistic regression cross validation to find the best regularization rate
def lr_cross_val(X, y, lr_rate, lr_type = 'Logistic regression AUC-ROC score is '):
    auc_roc = []
    for C in lr_rate:
        start = datetime.datetime.now()
        classifier = LogisticRegression(C = C, random_state = 241)
        auc_roc_scores = cross_val_score(classifier, X, y, cv = kf, scoring = 'roc_auc', n_jobs = -1)
        mean_auc = np.mean(auc_roc_scores)
        auc_roc.append(mean_auc)
        finish = datetime.datetime.now()
        print(lr_type + str(round(mean_auc, 5)) + ' for regularization rate ' + str(C) + ', time elapsed: ' + str(finish - start))
    max_C = lr_rate[auc_roc.index(max(auc_roc))]
    max_rate = max(auc_roc)
    return max_C, max_rate, auc_roc

#visualization of regularization rate and AUC ROC score
def lr_plot(lr_rate, lr_roc, title = 'Logistic regression'):
    plt.plot(np.log10(lr_rate), lr_roc)
    plt.xlabel('Log of regularization')
    plt.ylabel('AUC-ROC score')
    plt.title(title)
    plt.show()

#-----------Logistic regression calculations-----------------------------------  

#number of heros used
heroes = train[['r1_hero', 'r2_hero', 'r3_hero', 'r4_hero', 'r5_hero', 'd1_hero', 'd2_hero', 'd3_hero', 'd4_hero', 'd5_hero']]
heroes = pd.DataFrame(np.reshape(heroes.values, (10 * len(heroes))), columns = ['heroes'])
n_heroes = len(heroes['heroes'].value_counts())

# matrix of features creation 
scaler = StandardScaler()
X_train_lr = scaler.fit_transform(X_train)
X_train_wc = scaler.fit_transform(remove_cf(train).iloc[:,:-1].values)
X_train_h = np.concatenate((X_train_wc, create_heroes(train)), axis = 1)
X_test = np.concatenate((scaler.transform(remove_cf(test).values), create_heroes(test)), axis = 1)

#list of regularization rates
reg_rate = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]

#logistic regression
max_C_lr, max_rate_lr, auc_roc_lr = lr_cross_val(X_train_lr, y_train, lr_rate = reg_rate)
lr_plot(lr_rate = reg_rate, lr_roc = auc_roc_lr)
print('Maximal value of AUC-ROC score is ' + str(round(max_rate_lr, 5)) + ' for regularization rate ' + str(max_C_lr))

#logistic regression without categorical features
max_C_wc, max_rate_wc, auc_roc_wc = lr_cross_val(X_train_wc, y_train, lr_rate = reg_rate, lr_type = 'Logistic regression without categorical features AUC-ROC score is ')
lr_plot(lr_rate = reg_rate, lr_roc = auc_roc_wc, title = 'Logistic regression without categorical features')
print('Maximal value of AUC-ROC score is ' + str(round(max_rate_wc, 5)) + ' for regularization rate ' + str(max_C_wc))

#Logistic regression with heroes 
max_C_h, max_rate_h, auc_roc_h = lr_cross_val(X_train_h, y_train, lr_rate = reg_rate, lr_type = 'Logistic regression with heroes information AUC-ROC score is ')
lr_plot(lr_rate = reg_rate, lr_roc = auc_roc_h, title = 'Logistic regression with heroes information')
print('Maximal value of AUC-ROC score is ' + str(round(max_rate_h, 5)) + ' for regularization rate ' + str(max_C_h))

#testing the best model, choosing the highest and the lowest probabilities
best_classifier = LogisticRegression(C = max_C_h, random_state = 147)
best_classifier.fit(X_train_h, y_train)
y_pred = best_classifier.predict_proba(X_test)[:,1]
print(round(max(y_pred), 5), round(min(y_pred), 5))