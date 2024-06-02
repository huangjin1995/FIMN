#!/usr/bin/env python
# -*- encoding:utf-8 -*-

import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.feature_column as feature_column

from .utils import logger
from module.network import *

def ple_layer(input, task_num, share_expert_num, specific_expert_num, num_levels, ple_hidden_dims, ple_output_dim, gate_hidden_dims, training=False):
    """
    Args:
        input:
        task_num:
        share_expert_num:
        specific_expert_num:
        num_levels:
        ple_hidden_dims:
        ple_output_dim:
        gate_hidden_dims:
        training:
    Returns:
    """

    def cgc_net(inputs, level_name, is_last=False):
        # inputs: [task1, task2, ... taskn, shared task]
        specific_expert_outputs = []
        for i in range(task_num):
            for j in range(specific_expert_num):
                expert_network = dnn(level_name + 'pexpert_' + str(i) + str(j), inputs[i], ple_hidden_dims, ple_output_dim, None, training=training)  # [B, d]
                specific_expert_outputs.append(expert_network)

        shared_expert_outputs = []
        for k in range(share_expert_num):
            expert_network = dnn(level_name + 'sexpert_' + str(k), inputs[-1], ple_hidden_dims, ple_output_dim, None, training=training)  # [B, d]
            shared_expert_outputs.append(expert_network)

        cgc_outs = []
        for i in range(task_num):
            # concat task-specific expert and task-shared expert
            cur_expert_num = specific_expert_num + share_expert_num
            # task_specific + task_shared
            cur_experts = specific_expert_outputs[
                          i * specific_expert_num:(i + 1) * specific_expert_num] + shared_expert_outputs

            expert_concat = tf.stack(cur_experts, axis=1)  # [B, experts_num, d]

            # build gate layers
            gate_output = dnn(level_name + 'gate_' + str(i), inputs[i], gate_hidden_dims, cur_expert_num, None,
                              training=training)  # [B, experts_num]
            gate_output = tf.nn.softmax(gate_output, axis=1)
            gate_output = tf.expand_dims(gate_output, axis=1)  # [B, 1, experts_num]
            outputs = tf.matmul(gate_output, expert_concat)  # [B, 1, d]
            shape = outputs.shape.as_list()
            outputs = tf.reshape(outputs, [-1, shape[1] * shape[2]])  # [B, d]
            cgc_outs.append(outputs)

        # task_shared gate, if the level not in last, add one shared gate
        if not is_last:
            cur_expert_num = task_num * specific_expert_num + share_expert_num
            cur_experts = specific_expert_outputs + shared_expert_outputs  # all the expert include task-specific expert and task-shared expert

            expert_concat = tf.stack(cur_experts, axis=1)  # [B, experts_num, d]

            # build gate layers
            gate_output = dnn(level_name + 'gate_share', inputs[-1], gate_hidden_dims, cur_expert_num, None,
                              training=training)  # [B, experts_num]
            gate_output = tf.expand_dims(gate_output, axis=1)  # [B, 1, experts_num]
            outputs = tf.matmul(gate_output, expert_concat)  # [B, 1, d]
            shape = outputs.shape.as_list()
            outputs = tf.reshape(outputs, [-1, shape[1] * shape[2]])  # [B, d]
            cgc_outs.append(outputs)

        return cgc_outs


    ple_inputs = [input] * (task_num + 1)  # [task1, task2, ... taskn, shared task]
    ple_outputs = []
    for i in range(num_levels):
        if i == num_levels - 1:  # the last level
            ple_outputs = cgc_net(inputs=ple_inputs, level_name='level_' + str(i) + '_', is_last=True)
        else:
            ple_outputs = cgc_net(inputs=ple_inputs, level_name='level_' + str(i) + '_', is_last=False)
            ple_inputs = ple_outputs
    return ple_outputs


def gple_layer(input, task_num, share_expert_num, specific_expert_num, num_levels, ple_hidden_dims, ple_output_dim, gate_hidden_dims, training=False):
    """
    Args:
        input:
        task_num:
        share_expert_num:
        specific_expert_num:
        num_levels:
        ple_hidden_dims:
        ple_output_dim:
        gate_hidden_dims:
        training:
    Returns:
    """

    def cgc_net(inputs, level_name, is_last=False):
        # inputs: [task1, task2, ... taskn, shared task]
        specific_expert_outputs = []
        for i in range(task_num):
            for j in range(specific_expert_num):
                expert_network = dnn(level_name + 'pexpert_' + str(i) + str(j), inputs[i], ple_hidden_dims, ple_output_dim, None, training=training)  # [B, d]
                specific_expert_outputs.append(expert_network)

        shared_expert_outputs = []
        for k in range(share_expert_num):
            expert_network = dnn(level_name + 'sexpert_' + str(k), inputs[-1], ple_hidden_dims, ple_output_dim, None, training=training)  # [B, d]
            shared_expert_outputs.append(expert_network)

        cgc_outs = []
        for i in range(task_num):
            # concat task-specific expert and task-shared expert
            cur_expert_num = specific_expert_num + share_expert_num
            # task_specific + task_shared
            cur_experts = specific_expert_outputs[
                          i * specific_expert_num:(i + 1) * specific_expert_num] + shared_expert_outputs

            expert_concat = tf.stack(cur_experts, axis=1)  # [B, experts_num, d]

            # build gate layers
            gate_output = dnn(level_name + 'gate_' + str(i), inputs[i], gate_hidden_dims, cur_expert_num, None,
                              training=training)  # [B, experts_num]
            gate_output = tf.nn.softmax(gate_output, axis=1)
            gate_output = tf.expand_dims(gate_output, axis=1)  # [B, 1, experts_num]
            outputs = tf.matmul(gate_output, expert_concat)  # [B, 1, d]
            shape = outputs.shape.as_list()
            outputs = tf.reshape(outputs, [-1, shape[1] * shape[2]])  # [B, d]
            cgc_outs.append(outputs)

        # task_shared gate, if the level not in last, add one shared gate
        if not is_last:
            cur_expert_num = task_num * specific_expert_num + share_expert_num
            cur_experts = specific_expert_outputs + shared_expert_outputs  # all the expert include task-specific expert and task-shared expert

            expert_concat = tf.stack(cur_experts, axis=1)  # [B, experts_num, d]

            # build gate layers
            gate_output = dnn(level_name + 'gate_share', inputs[-1], gate_hidden_dims, cur_expert_num, None,
                              training=training)  # [B, experts_num]
            gate_output = tf.expand_dims(gate_output, axis=1)  # [B, 1, experts_num]
            outputs = tf.matmul(gate_output, expert_concat)  # [B, 1, d]
            shape = outputs.shape.as_list()
            outputs = tf.reshape(outputs, [-1, shape[1] * shape[2]])  # [B, d]
            cgc_outs.append(outputs)

        return cgc_outs


    ple_inputs = [input] * (task_num + 1)  # [task1, task2, ... taskn, shared task]
    ple_outputs = []
    for i in range(num_levels):
        if i == num_levels - 1:  # the last level
            ple_outputs = cgc_net(inputs=ple_inputs, level_name='glevel_' + str(i) + '_', is_last=True)
        else:
            ple_outputs = cgc_net(inputs=ple_inputs, level_name='glevel_' + str(i) + '_', is_last=False)
            ple_inputs = ple_outputs
    return ple_outputs



