#!/usr/bin/env python

###########################################################################
# graph_constructor.py: Computational graphs for TensorFlow skills test
# Author: Chris Hodapp (hodapp87@gmail.com)
# Date: 2017-11-15
###########################################################################

import tensorflow as tf
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense

def lr_model(n_features, n_labels, scope="lr"):
    """Returns a Keras model for logistic regression, given a number of
    feature dimensions (n_features) and labels (n_labels)
    """
    with tf.variable_scope(scope):
        # Note that multinomial logistic regression for N classes is
        # just input layer connected to N-unit output with softmax:
        model = Sequential([
            Dense(n_labels, activation='softmax', input_dim=n_features)
        ])
        return model

def ann_model(n_features, n_labels, hidden_units, scope="ann"):
    """Returns a Keras model for a 2-layer neural network, given a number
    of feature dimensions (n_features), number of labels (n_labels),
    and number of units for the hidden layer (hidden_units).
    """
    with tf.variable_scope(scope):
        model = Sequential([
            Dense(hidden_units, input_dim=n_features, activation='relu'),
            Dense(n_labels, activation='softmax'),
        ])
        return model

class KNearestNeighborGraph(object):
    def __init__(self, k, X, y, scope="knn"):
        """Create a TensorFlow graph for a k nearest neighbor classifier.
        This requires no training, but it does require that all of the data
        be supplied here in the constructor. Prediction does not do any
        weighting; the nearest k neighbors simply vote on a label.
        
        Pass inputs in smaller batches (e.g. 25 or 50 samples) in
        'predict'.  Memory usage is very inefficient in this
        implementation.
        
        Arguments:
        k -- Number of nearest neighbors to use
        X -- Input data (should be a NumPy array)
        y -- Input labels (must have same length as X)

        """
        with tf.variable_scope(scope):
            self.example = tf.placeholder(tf.float32, [None, X.shape[1]])
            # All data & labels must be stored:
            self.y = tf.convert_to_tensor(y)
            X = tf.convert_to_tensor(X, dtype=np.float32)
            # However, add a degenerate dimension to X and the example
            # so that it does a sort of 'outer' broadcast (and gives us
            # every pairwise difference in dist2):
            self.X_ = tf.expand_dims(X, 0)
            example_ = tf.expand_dims(self.example, 1)
            # Everything is based on squared Euclidean distance
            # between the input point, and every data point in X:
            dist2 = tf.reduce_sum((self.X_ - example_)**2, axis=2)
            if k == 1:
                # k=1 requires no voting, just label of min-distance point:
                self.predict_node = tf.gather(self.y, tf.argmin(dist2, axis=1))
            else:
                # The below relies on some functions that work only on certain
                # dimensions, so map_fn is used to run it across each row:
                def single_knn(dist2_row):
                    # For each input row, get indices of the k
                    # nearest neighbors:
                    _, predict_idxs = tf.nn.top_k(-dist2_row, k=k)
                    # (Note negation of dist2 - top_k does *descending* sort.)
                    # And respective k labels of those k indices:
                    self.predict_y = tf.gather(self.y, predict_idxs)
                    # Tally them up:
                    ys, _, num_y = tf.unique_with_counts(self.predict_y)
                    # Find most common one:
                    return tf.gather(ys, tf.argmax(num_y))
                self.predict_node = tf.map_fn(single_knn, dist2, dtype=tf.int64)
        # TODO: Make tie-breaking behavior more helpful. In the event of a tie
        # of distance, guess argmin and top_k likely pick the sample with the
        # lower index, which would bias towards a specific ordering of the data
        # (which should be irrelevant).  In the event of a tie in the voting for
        # labels, argmax will likewise pick the label that appears first, which
        # may bias either towards the lower-numbered label or the ordering of
        # the data (likewise, both irrelevant).

    def predict(self, session, X_in):
        return session.run(self.predict_node, {self.example: X_in})
