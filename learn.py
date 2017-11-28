#!/usr/bin/env python

###########################################################################
# learn.py: Machine learning algorithms for TensorFlow skills test
# Author: Chris Hodapp (hodapp87@gmail.com)
# Date: 2017-11-15
###########################################################################

import data_preprocessing
import graph_constructor

import tensorflow as tf
import numpy as np
import sklearn
import keras
from keras.optimizers import SGD

def train_and_validate(algorithm):
    """This function is mandated by the requirements.  Argument
    'algorithm' takes values of 'knn', 'logistic', and '2-layer',
    and the function will train the respective classifier
    on training data, predict on the validation data, and print that
    classifier's testing accuracy.
    """
    train_X_orig, train_y_orig, test_X, test_y = data_preprocessing.read_raw_data()
    data_preprocessing.standardize(train_X_orig, test_X)
    train_X, valid_X, train_y, valid_y = data_preprocessing.split(
        train_X_orig, train_y_orig)
    # Simplify things by making labels start at 0:
    train_y[:] = train_y[:] - 1
    valid_y[:] = valid_y[:] - 1

    if algorithm == "knn":
        val_acc = run_knn(train_X, train_y, valid_X, valid_y)
    else:
        # Logistic & ANN use Keras, so just get model & SGD params:
        if algorithm == "logistic":
            model = graph_constructor.lr_model(train_X.shape[1], 6)
            sgd = SGD(lr=0.02, decay=1e-5, momentum=0.9, nesterov=True)
        elif algorithm == "2-layer":
            model = graph_constructor.ann_model(train_X.shape[1], 6, 50)
            sgd = SGD(lr=0.02, decay=2e-5, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])
        # One-hot encode for Keras:
        train_y_cat = keras.utils.to_categorical(train_y - 1, num_classes=6)
        valid_y_cat = keras.utils.to_categorical(valid_y - 1, num_classes=6)
        # Train, and get validation accuracy at the same time:
        history = model.fit(
            train_X.values,
            train_y_cat,
            epochs=120,
            batch_size=128,
            validation_data=(valid_X.values, valid_y_cat))
        val_acc = history.history["val_acc"][-1]
        
    return val_acc

def run_knn(train_X, train_y, valid_X, valid_y):
    """Runs kNN for this dataset, returning validation accuracy."""
    knn = graph_constructor.KNearestNeighborGraph(
        1, train_X.values, train_y.values)
    with tf.Session() as session:
        init = tf.global_variables_initializer()
        session.run(init)
        # Evaluate validation accuracy:
        batch_size = 50
        valid_y_predict = np.zeros((valid_X.shape[0]))
        batches = valid_X.shape[0] // batch_size + 1
        for b in range(batches):
            print("Batch {0}/{1}...".format(b + 1, batches))
            i0 = b * batch_size
            i1 = (b + 1) * batch_size
            valid_y_predict[i0:i1] = knn.predict(session, valid_X.iloc[i0:i1, :])
        valid_acc = sklearn.metrics.accuracy_score(valid_y, valid_y_predict)
        return valid_acc

if __name__ == '__main__':
    for algo in ("logistic", "2-layer", "knn"):
        print("-"*60)
        print(algo + ": ")
        print("-"*60)
        acc = train_and_validate(algo)
        #print("Loss: %s" % ev["loss"])
        print("{0} validation accuracy: {1}".format(algo, acc))
