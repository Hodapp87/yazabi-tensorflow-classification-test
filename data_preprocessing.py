#!/usr/bin/env python

###########################################################################
# data_preprocessing.py: Preprocessing for TensorFlow skills test
# Author: Chris Hodapp (hodapp87@gmail.com)
# Date: 2017-11-15
###########################################################################

import os

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sklearn.decomposition
import sklearn.externals
import sklearn.model_selection
import sklearn.preprocessing

# Data source(s):
# https://github.com/pdelboca/human-activity-recognition-using-smartphones
# http://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones

def get_feature_list():
    """Returns a list of (Pandas-friendly) feature names for the features
    in this dataset besides the inertial sensors.  These are in the
    same order as in the files X_train.txt and X_test.txt.
    """
    # I really don't feel like copying these into the document, so
    # instead load them from the included file:
    feature_list = pd.read_csv("UCI HAR Dataset/features.txt",
                               index_col=0, header=None, sep=" ")
    # ...and sanitize them so Pandas can use them for column names:
    san_dict = {ord("("): "", ord(")"): "", ord("-"): "_", ord(","): "_"}
    features = list(f.translate(san_dict) for f in feature_list.iloc[:,0])
    return(features)

def read_accel(axis, train_or_test):
    """Reads accelerometer data from the given axis ("x", "y", or
    "z"), and for the given stage ("train" or "test").  Returns a
    dataframe with 128 columns, named like accX_0 to accX_127 (and
    likewise for other axes).
    """
    fname = "UCI HAR Dataset/{0}/Inertial Signals/total_acc_{1}_{0}.txt".\
        format(train_or_test, axis)
    # It's a bit confusing to have column names that are also numbers,
    # so use accX_0...accX_127 (for instance):
    names = ["acc" + axis.upper() + "_" + str(i) for i in range(128)]
    df = pd.read_csv(fname, delim_whitespace=True, header=None,
                     index_col=None, names=names)
    assert df.shape[1] == 128
    return df

def read_raw_data():
    """Returns training & testing data as (train_X, train_y, test_X,
    test_y), where 'train_X' is a DataFrame containing features for
    the training data (this includes the named features, as well as
    the 128-dimension vectors from accelerometer readings for X, Y,
    and Z, in order), 'train_y' is a Series with the corresponding
    labels, and likewise for 'test_X' and 'test_y' for testing data.
    Data is not standardized or preprocessed.
    """
    # Read labels first, and extract just the series:
    train_y = pd.read_csv("UCI HAR Dataset/train/y_train.txt", header=None).iloc[:,0]
    test_y = pd.read_csv("UCI HAR Dataset/test/y_test.txt", header=None).iloc[:,0]
    # Read summary statistics (the labeled features):
    features = get_feature_list()
    train_X_summary = pd.read_csv(
        "UCI HAR Dataset/train/X_train.txt", delim_whitespace=True,
        header=None, names=features, index_col=None)
    test_X_summary = pd.read_csv(
        "UCI HAR Dataset/test/X_test.txt", delim_whitespace=True,
        header=None, names=features, index_col=None)
    # Read accelerometer data:
    train_accels = [read_accel(v, "train") for v in ("x","y","z")]
    test_accels  = [read_accel(v, "test")  for v in ("x","y","z")]
    # Merge everything together:
    train_X = pd.concat([train_X_summary] + train_accels, axis=1)
    test_X  = pd.concat([test_X_summary]  + test_accels,  axis=1)
    return (train_X, train_y, test_X, test_y)

def standardize(train_X, test_X):
    """Modifies 'train_X' and 'test_X' to standardize the data in-place to
    zero mean and unit variance."""
    ss = sklearn.preprocessing.StandardScaler()
    train_X.iloc[:, :] = ss.fit_transform(train_X)
    test_X.iloc[:, :] = ss.transform(test_X)

def split(train_X_orig, train_y_orig, ratio=0.75):
    """Splits original training data (i.e. what is supplied in the "train"
    directory) into training and validation data, using 'ratio' as the
    amount that should be split for training (default 0.7).  Returns
    (train_X, validation_X, train_y, validation_y).
    """
    # Class labels are a bit lopsided, so stratified sampling is done:
    splits = sklearn.model_selection.train_test_split(
        train_X_orig,
        train_y_orig,
        test_size=1 - ratio,
        random_state=123456,
        stratify=train_y_orig)
    return splits

if __name__ == '__main__':
    train_X_orig, train_y_orig, test_X, test_y = read_raw_data()
    standardize(train_X_orig, test_X)
    train_X, valid_X, train_y, valid_y = split(train_X_orig, train_y_orig)
    train_X.to_csv("split_train_X.csv")
    valid_X.to_csv("split_valid_X.csv")
    train_y.to_csv("split_train_y.csv")
    valid_y.to_csv("split_valid_y.csv")
