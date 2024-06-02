#!/usr/bin/env python
# -*- encoding:utf-8 -*-

import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.feature_column as feature_column

from .utils import logger
from module.network import *

def interest_fusion_layer(mix_input, interest_inputs, interests_num):
    """
    Args:
        scene_input: [B, d1]
        interest_inputs: [B, m * d], m denotes interests num
        interests_num: interests num
    Returns:
    """
    # intra_gate
    intra_gate_num = interest_inputs.get_shape().as_list()[1]
    intra_gate = layers.fully_connected(mix_input, 256, activation_fn=get_act_fn('lrelu'), scope='intra_gate1')
    # [B, m*d]
    intra_gate = layers.fully_connected(intra_gate, intra_gate_num, activation_fn=get_act_fn('sigmoid'), scope='intra_gate2')
    intra_gate = intra_gate * 2

    # [B, m]
    out1 = tf.reshape(intra_gate, [-1, interests_num, intra_gate_num // interests_num])
    out1 = tf.reduce_mean(out1, axis=-1)

    logger('intra_gate', intra_gate)
    # inter_gate

    inter_gate_num = interests_num
    inter_gate = layers.fully_connected(mix_input, 256, activation_fn=get_act_fn('lrelu'), scope='inter_gate1')
    # [B, 1, m]
    inter_gate = layers.fully_connected(inter_gate, inter_gate_num, activation_fn=get_act_fn('softmax'), scope='inter_gate2')

    out2 = inter_gate

    inter_gate = tf.expand_dims(inter_gate, axis=1)  # [B, 1, m]
    logger('inter_gate', inter_gate)
    # intra fusion
    intra_fusion = intra_gate * interest_inputs
    logger('intra_fusion', intra_fusion)
    # inter fusion
    # [B, m, d]
    inter_fusion = tf.reshape(intra_fusion, [-1, interests_num, intra_gate_num // interests_num])
    logger('inter_fusion', inter_fusion)
    # Weighted sum (B, 1, m) * (B, m, d)
    outputs = tf.matmul(inter_gate, inter_fusion)  # [B, 1, d]
    logger('outputs', outputs)
    dim = outputs.get_shape().as_list()
    logger('dim', dim)
    # [B, d]
    outputs = tf.reshape(outputs, [-1, dim[1] * dim[-1]])
    logger('out1', out1)
    logger('out2', out2)
    return outputs, out1, out2
