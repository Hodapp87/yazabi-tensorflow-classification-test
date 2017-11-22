#!/usr/bin/env python

###########################################################################
# graph_constructor.py: Computational graphs for TensorFlow skills test
# Author: Chris Hodapp (hodapp87@gmail.com)
# Date: 2017-11-15
###########################################################################

import tensorflow as tf
import numpy as np

def lr_model_fn(features, labels, mode, params):
    """Model function for an Estimator for a logistic regression
    classifier (using weight decay for regularization).
    
    Parameters needed in 'params':
    n_features -- Number of feature dimensions
    n_classes -- Number of distinct classes
    learning_rate -- Learning rate for gradient descent
    lambda -- Weight decay factor for regularization

    """
    # Model:
    weights = tf.get_variable(
        "weights", [params["n_features"], params["n_classes"]],
        dtype=tf.float32)
    bias = tf.get_variable("bias", [params["n_classes"]], dtype=tf.float32)
    # Unnormalized output (logits?):
    logits = tf.matmul(features["x"], weights) + bias
    sm = tf.nn.softmax(logits)
    # TODO: Actually, are either of those necessary? Could I just use
    # argmax on logits directly?
    predictions = tf.argmax(sm, axis=1)
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode, predictions={"activity": predictions})
    else:
        # Cost function consists of cross-entropy, plus weight decay for
        # regularization:
        onehot_labels = tf.one_hot(labels, params["n_classes"])
        decay = tf.reduce_sum(weights**2)
        xe = tf.losses.softmax_cross_entropy(
            onehot_labels=onehot_labels, logits=logits)
        loss = xe + params["lambda"] * decay / 2

        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(labels=labels,
                                            predictions=predictions)
        }
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=params["learning_rate"])
        train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=eval_metric_ops)

def ann_model_fn(features, labels, mode, params):
    """Model function for an Estimator for a 2-layer neural network.
    
    Parameters needed in 'params':
    hidden_units -- Number of units in hidden layer
    n_classes -- Number of distinct classes
    learning_rate -- Learning rate for gradient descent
    """
    
    # Hidden layer:
    hidden = tf.layers.dense(
        features["x"], params["hidden_units"], activation=tf.nn.relu)
    # Output layer:
    logits = tf.layers.dense(hidden, params["n_classes"])
    # If we need class probabilities:
    #prob = tf.nn.softmax(self.logits)
    predictions = tf.argmax(logits, axis=1)
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode, predictions={"activity": predictions})
    else:
        #loss = tf.nn.softmax_cross_entropy_with_logits(
        #    onehot_labels=onehot_labels, logits=logits)
        onehot_labels = tf.one_hot(labels, params["n_classes"])
        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=onehot_labels, logits=logits)
        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(labels=labels,
                                            predictions=predictions)
        }
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=params["learning_rate"])
        train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=eval_metric_ops)

def knn_model_fn(features, labels, mode, params):
    """Model function for an Estimator for a k-nearest neighbors
    classifier.  The training phase on this Estimator is a no-op.
    
    Parameters needed in 'params':
    X -- NumPy array for all training data
    y -- NumPy array with corresponding labels to 'X'
    k -- Number of neighbors to use
    n_classes -- Number of distinct classes
    """
    k = params["k"]
    # All data & labels must be stored:
    y = tf.convert_to_tensor(params["y"])
    X = tf.convert_to_tensor(params["X"], dtype=np.float32)
    # However, add a degenerate dimension to X and the input so that
    # it does a sort of 'outer' broadcast (and gives us every pairwise
    # difference in dist2):
    X_ = tf.expand_dims(X, 0)
    X_in = tf.expand_dims(features["x"], 1)
    # Everything is based on squared Euclidean distance between the
    # input point, and every data point in X:
    dist2 = tf.reduce_sum((X_ - X_in)**2, axis=2)
    if k == 1:
        # k=1 requires no voting, just label of min-distance point:
        predictions = tf.gather(y, tf.argmin(dist2, axis=1))
    else:
        # The below relies on some functions that work only on certain
        # dimensions, so map_fn is used to run it across each row:
        def single_knn(dist2_row):
            # For each input row, get indices of the k nearest
            # neighbors:
            _, predict_idxs = tf.nn.top_k(-dist2_row, k=k)
            # (Note negation of dist2 - top_k does *descending* sort.)
            # And respective k labels of those k indices:
            predict_y = tf.gather(y, predict_idxs)
            # Tally them up:
            ys, _, num_y = tf.unique_with_counts(predict_y)
            # Find most common one:
            return tf.gather(ys, tf.argmax(num_y))
        predictions = tf.map_fn(single_knn, dist2, dtype=tf.int64)
    # TODO: Make tie-breaking behavior more helpful. In the event of a
    # tie of distance, guess argmin and top_k likely pick the sample
    # with the lower index, which would bias towards a specific
    # ordering of the data (which should be irrelevant).  In the event
    # of a tie in the voting for labels, argmax will likewise pick the
    # label that appears first, which may bias either towards the
    # lower-numbered label or the ordering of the data (likewise, both
    # irrelevant).
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode, predictions={"activity": predictions})
    else:
        onehot_labels = tf.one_hot(labels, params["n_classes"])
        # loss is irrelevant here (no training, thus loss doesn't
        # matter), but it appears I must have it for EstimatorSpec:
        loss = tf.losses.absolute_difference(labels, predictions)
        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(labels=labels,
                                            predictions=predictions)
        }
        return tf.estimator.EstimatorSpec(
            mode=mode,
            train_op=tf.no_op(),
            loss=loss,
            eval_metric_ops=eval_metric_ops)
