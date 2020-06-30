
# This file uses Sam's data as training data and 'cosmoDC2_v1.1.4_small_photoz_magerr_10y' catalog as testing data. 


import sys
import numpy as np
from numpy.random import shuffle
from matplotlib import pyplot as plt

from sklearn.ensemble import RandomForestClassifier
import itertools
import copy
from astropy.coordinates import SkyCoord
import pandas as pd

import change_path
import GCRCatalogs
from astropy.table import Table

#Variable Definition

llinewidth = 3
label_fontsize = 18
title_fontsize = 22
legend_fontsize = 14
tick_fontsize = 16
ppointsize = 8
ccapsize = 12

n_bins = 101
z_min = 0
z_max = 3

galid_ind = 0
rz_ind = 1
tr_start = 4
tr_end = 16
mag_r_ind = tr_start + 2

# Loading the training set

train_dat_dir = '/global/projecta/projectdirs/lsst/groups/PZ/PhotoZDC2/COSMODC2v1.1.4/10_year_error_estimates/TESTRUN/sample_cosmodc2_w10year_errors.dat'
train_data = np.loadtxt(train_dat_dir, skiprows=1)
format_train = np.delete(train_data,[16,17],1)

# shuffle training data
data_len = len(format_train)
data_rand_ind = np.random.choice(data_len,data_len,replace = False)
random_train = format_train[data_rand_ind]

# Limit to the range of reshift value we are inspecting
train = random_train[(random_train[:,rz_ind] >= z_min) & (random_train[:,rz_ind] < z_max)] #appy to rows individually
training_size = len(train)


# Define Machine Learning method

class ClassCond(object):
    def __init__(self, classifier, bins):
        self.bins = bins
        self.clf = classifier

    def fit(self, X, Y):
        self.Y = np.array(Y, dtype=np.double)
        grid_Y = np.linspace(z_min, z_max, num=self.bins)
        self.delta_z = grid_Y[1:] - grid_Y[:-1]
        self.midpoints = grid_Y[:-1] + self.delta_z/2
        self.response_classes = np.array(np.digitize(Y, bins=grid_Y), dtype=np.int) 
        #put edge cases in bins in range

        self.clf.fit(X=X, y=self.response_classes)

    def predict(self, X):
        nbatch = 10
        prob_vec = self.clf.predict_proba(X)
        #normalize the histogram to unit area (p * delta z)
        #prob vec is a matrix with each row being a galaxy and each column being a histogram bin height.
        result = prob_vec/(self.delta_z[0]) #on the premise that the grid is evenly spaced
        return result
    
    def ind_result(self,X):
        prob_vec = self.predict(X)
        each = sum(self.midpoints * prob_vec *(z_max/(n_bins-1)))
        
def multiple_train(times):
    model_table = []
    for i in range(times):
        training_ind = np.random.choice(training_size, training_size, replace = True)
        train_per = train[training_ind]
        model = ClassCond(RandomForestClassifier(n_jobs=10), bins=n_bins) 
        model.fit(train_per[:, tr_start:tr_end], train_per[:, rz_ind])
        model_table += [model]     
    return model_table