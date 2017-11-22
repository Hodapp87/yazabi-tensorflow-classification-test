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
    
    # key = algorithm name, value = (model function, # of epochs, params)
    models = {
        "knn": (graph_constructor.knn_model_fn,
                 0,
                 { "k": 1,
                   "X": train_X.values,
                   "y": train_y.values,
                   "n_classes": 6,
                 }),
        "logistic": (graph_constructor.lr_model_fn,
                     80,
                     { "learning_rate": 0.02,
                       "n_classes": 6,
                       "n_features": train_X.shape[1],
                       "lambda": 0.02,
                     }),
        "2-layer": (graph_constructor.ann_model_fn,
                    100,
                    { "learning_rate": 0.01,
                      "n_classes": 6,
                      "hidden_units": 200,
                    }),
    }

    # Create estimator with given function & params:
    model_fn, epochs, params = models[algorithm]
    est = tf.estimator.Estimator(model_fn=model_fn, params=params)

    # Train:
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_X.values.astype(np.float32)},
        y=train_y.values,
        num_epochs=epochs,
        shuffle=False)
    est.train(input_fn=train_input_fn)

    # Evaluate:
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": valid_X.values.astype(np.float32)},
        y=valid_y.values,
        num_epochs=1,
        shuffle=False)
    ev = est.evaluate(input_fn=test_input_fn)
    return ev

if __name__ == '__main__':
    for algo in ("logistic", "2-layer", "knn"):
        print("-"*60)
        print(algo + ": ")
        print("-"*60)
        ev = train_and_validate(algo)
        #print("Loss: %s" % ev["loss"])
        print("{0} validation accuracy: {1}".format(algo, ev["accuracy"]))
