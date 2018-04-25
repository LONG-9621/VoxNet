#!/usr/bin/python3.5
# -*- coding: utf-8 -*-

"""
  Voxnet implementation on tensorflow
"""

import tensorflow as tf
import numpy as np
import os
from glob import glob
import random

class VoxNet(object):
    def __init__(self, learning_rate=0.001, num_classes=14, batch_size=32, epochs=64):
        """
        Init paramters
        """
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        # to enable tf logging info
        tf.logging.set_verbosity(tf.logging.INFO)

    def core(self, features, labels, mode):
        """
        Voxnet tensorflow graph.
        It follows description from this TensorFlow tutorial:
        `https://www.tensorflow.org/versions/master/tutorials/mnist/pros/index.html#deep-mnist-for-experts`

        Args:
        `features`: default paramter for tf.model_fn
        `labels`: default paramter for tf.model_fn
        `mode`: default paramter for tf.model_fn

        Ret:
        `EstimatorSpec`:    predictions/loss/train_op/eval_metric_ops in EstimatorSpec object
        """
        input_layer = tf.reshape(features['OccuGrid_input'], [-1, 32, 32, 32, 1])

        # Layer 1: 3D conv(filters_num=32, filter_kernel_size=5, strides=2)
        # Input(32*32*32), Output:(14*14*14)*32
        conv1 = tf.layers.conv3d(inputs=input_layer, filters=32, kernel_size=[5,5,5], strides=[2,2,2],name='conv1')

        # Layer 2: 3D conv(filters_num=32, filter_kernel_size=3, strides=1)
        # Max-pooling (2*2*2)
        # Input(32*32*32)*32, Output:(6*6*6)*32
        conv2 = tf.layers.conv3d(inputs=conv1, filters=32, kernel_size=[3,3,3], strides=[1,1,1],name='conv2')
        # TODO: (vincent.cheung.mcer@gmail.com) not sure about the pool_size
        max_pool1 = tf.layers.max_pooling3d(inputs=conv2, pool_size=2,strides=2)
        # TODO: (vincent.cheung.mcer@gmail.com), later can try 3D conv instead of Fully Connect dense layer
        max_pool1_flat = tf.reshape(max_pool1, [-1,6*6*6*32])

        # Layer 3: Fully Connected 128
        # Input (6*6*6)*32, Output:(128)
        dense4 = tf.layers.dense(inputs=max_pool1_flat, units=128)

        # Layer 4: Fully Connected Output
        # Input: (128), Output:K class
        dense5 = tf.layers.dense(inputs=dense4, units=self.num_classes)
        logits = dense5

        predictions = {
            # Generate predictions (for PREDICT and EVAL mode)
            'pred_cls': tf.argmax(input=logits, axis=1),
            # Add `softmax_tensor` to the graph. It is used for PREDICT and by the `logging_hook`.
            'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        # Define loss function (for both TRAIN and EVAL modes)
        onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=self.num_classes)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)
        tf.summary.scalar("loss", loss)
        # Configure the Training Op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        # Add evaluation metrics (for EVAL mode)
        eval_metric_ops = {'accuracy': tf.metrics.accuracy(labels=labels, predictions=predictions['pred_cls'])}

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)