#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 13:32:23 2017

@author: antonperhunov
"""

import pandas as pd
import numpy as np

from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.svm import SVC
from sklearn.model_selection import KFold, GridSearchCV

newsgroups = datasets.fetch_20newsgroups(subset = 'all', categories=['alt.atheism', 'sci.space'])
news_data = newsgroups.data
news_class = newsgroups.target

vectorizer = TfidfVectorizer()
vectorizer.fit_transform(news_data)

grid = {'C': np.power(10.0, np.arange(-5, 6))}
kf = KFold(n_splits = 5, shuffle = True, random_state = 241)
svm = SVC(kernel = 'linear', random_state = 241)
gs = GridSearchCV(svm, grid, cv = kf, scoring = 'accuracy')
gs.fit(vectorizer.transform(news_data), news_class)

score = 0
C = 0
for result in gs.grid_scores_:
    if result.mean_validation_score > score:
        score = result.mean_validation_score
        C = result.parameters['C']

best_svm = SVC(kernel = 'linear', random_state = 241, C = C)
best_svm.fit(vectorizer.transform(news_data), news_class)

words = vectorizer.get_feature_names()
coefs = pd.DataFrame(best_svm.coef_.data, best_svm.coef_.indices)
top_words = coefs[0].map(lambda w: abs(w)).sort_values(ascending=False).head(10).index.map(lambda i: words[i])
top_words.sort()
print(top_words)