#!/usr/bin/env python
# -*- encoding:utf-8 -*-

import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.feature_column as feature_column

from .utils import logger
from module.network import *

def pn_layer(scene_input):
    """
    Args:
        scene_input: need a batch data in same scene
    Returns:
    """
    pass
    return

def weighted_pn_layer(scene_input):
    """
    Args:
        scene_input: need a batch data in same scene
    Returns:
    """
    pass
    return


def monotonicity_dnn(x, output_shape, name, activation=None):
    with tf.name_scope(name):
        n_inputs = x.get_shape().as_list()[1]
        W = tf.Variable(tf.random.uniform([n_inputs, output_shape], -1.0, 1.0), name='weights')
        b = tf.Variable(tf.zeros([output_shape]), name='bias')
        z = tf.matmul(x, -W * W) + b
        if activation == 'relu':
            return tf.nn.relu(z)
        elif activation == 'sigmoid':
            return tf.sigmoid(z)
        else:
            return z

def inverse_frequency_gate(scene_input, freq_input):
    n = scene_input.get_shape().as_list()[1]
    # [B, 1]
    inverse_weight = monotonicity_dnn(freq_input, 1, 'monotonic', activation='sigmoid')
    inverse_weight = 2 * inverse_weight  # range(0,2)
    inverse_weight = tf.tile(inverse_weight, [1, n])
    return inverse_weight * scene_input


def scene_info_enhance_layer(scene_input, output_dim, frequency_input, mode='dnn'):
    if mode == 'dnn':
        scene_output = layers.fully_connected(scene_input, 64, activation_fn=get_act_fn('lrelu'), scope='info_enhance1')
        scene_output = layers.fully_connected(scene_output, output_dim, activation_fn=get_act_fn('lrelu'), scope='info_enhance2')
    elif mode == 'partition_normalization':
        scene_output = pn_layer(scene_input)
    elif mode == 'weight_partition_normalization':
        scene_output = weighted_pn_layer(scene_input)
    elif mode == 'inverse_frequency_gate':
        scene_input = layers.fully_connected(scene_input, output_dim, activation_fn=get_act_fn('lrelu'), scope='info_enhance')
        scene_output = inverse_frequency_gate(scene_input, frequency_input)
    else:
        scene_output = layers.fully_connected(scene_input, 64, activation_fn=get_act_fn('lrelu'), scope='info_enhance1')
        scene_output = layers.fully_connected(scene_output, output_dim, activation_fn=get_act_fn('lrelu'),
                                              scope='info_enhance2')
    return scene_output
