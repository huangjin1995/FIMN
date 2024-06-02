#!/usr/bin/env python
# -*- encoding:utf-8 -*-

import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.feature_column as feature_column

from .utils import logger
from module.network import *

def scene_mutual_layer(expert_outputs):
    """
    Args:
        expert_outputs: list with length of task_num, each element of [B, d]
    Returns:
    """
    mutual_input = tf.stack(expert_outputs, axis=1)  # [B, n, d]

    hist_len = tf.shape(mutual_input)[1]
    hist_len = tf.reshape(hist_len, [1, 1])
    valid_length = tf.tile(hist_len, [tf.shape(mutual_input)[0], 1])
    max_len = mutual_input.get_shape().as_list()[1]
    key_masks = tf.sequence_mask(valid_length, max_len)
    seq_last_dim = mutual_input.shape.as_list()[-1]

    # [B, n, d]
    mutual_output, mutual_W = multihead_attention_maskself(queries=mutual_input,
                                     keys=mutual_input,
                                     num_heads=1,
                                     query_masks=key_masks,
                                     key_masks=key_masks,
                                     linear_projection=True,
                                     num_units=seq_last_dim,
                                     num_output_units=seq_last_dim,
                                     activation_fn='lrelu',
                                     att_mode='ln',
                                     is_target_attention=False,)


    gate_input = tf.concat([mutual_input, mutual_output], axis=-1)  # [B, n, 2d]
    # [B, n, 8]
    d1 = layers.fully_connected(gate_input, 8, activation_fn=get_act_fn('lrelu'), scope='mutual1')
    logger('scene_mutual_layer: ', d1)
    # [B, n, 1]
    d2 = layers.fully_connected(d1, 1, activation_fn=get_act_fn('sigmoid'), scope='mutual2')
    logger('scene_mutual_layer: ', d2)

    w = tf.tile(d2, [1, 1, tf.shape(mutual_input)[-1]])
    # [B, n, d]
    mutual_end = w * mutual_input + (1-w) * mutual_output

    # list with length of task_num, each element of [B, d]
    mutual_end = tf.unstack(mutual_end, len(expert_outputs), axis=1)
    return mutual_end, mutual_W







