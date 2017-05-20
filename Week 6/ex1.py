#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 16:36:53 2017

@author: antonperhunov
"""

import pandas as pd
import numpy as np
import math
import scipy
import pylab

from skimage.io import imread, imsave
from skimage import img_as_float
from sklearn.cluster import KMeans 

image = imread('parrots.jpg')
img_float = img_as_float(image)

w, h, rgb = img_float.shape

img_df = pd.DataFrame(np.reshape(img_float, (w*h, rgb)), columns = ['R', 'G', 'B'])

def clusters(img_df, n_clusters = 8):
    img_df = img_df.copy()
    cluster = KMeans(n_clusters = n_clusters, init = 'k-means++', random_state = 241)
    img_df['Cluster'] = cluster.fit_predict(img_df)
    
    means = img_df.groupby('Cluster').mean().values
    mean_df = [means[c] for c in img_df['Cluster'].values]
    mean_img = np.reshape(mean_df, (w, h, rgb))
    
    medians = img_df.groupby('Cluster').median().values
    median_df = [medians[c] for c in img_df['Cluster'].values]
    median_img = np.reshape(median_df, (w, h, rgb))
    
    return median_img, mean_img
    
def psnr(img1, img2):
    return(-10*np.log10(np.mean((img1-img2)**2)))
    
for i in range(1, 21):
    img_mean, img_median = clusters(img_df, n_clusters = i)
    if psnr(img_float, img_mean) >= 20:
        pylab.imshow(img_mean)
        print(psnr(img_float, img_mean), i)
        break
    elif psnr(img_float, img_median) >= 20:
        pylab.imshow(img_median)
        print(psnr(img_float, img_median), i)
        break