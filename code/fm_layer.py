#!/usr/bin/env python
# -*- encoding:utf-8 -*-

import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.feature_column as feature_column

from .utils import logger

def fm_layer(embedding_input, embedding_fea_num, embedding_dim):
    embeddings = tf.reshape(embedding_input, [-1, embedding_fea_num, embedding_dim])  # None, n, k
    sum_square = tf.square(tf.reduce_sum(embeddings, 1))
    square_sum = tf.reduce_sum(tf.square(embeddings), 1)
    y_v = 0.5 * tf.subtract(sum_square, square_sum)  # None, k
    return y_v
