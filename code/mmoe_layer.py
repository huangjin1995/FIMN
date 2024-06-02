import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.feature_column as feature_column

from .utils import logger
from module.network import *


def mmoe_layer(input, experts_num, gates_num, mmoe_hidden_dims, gate_hidden_dims, expert_output_dim, training=False):
    """
    Args:
        input:
        experts_num: task_num + 1
        gate_num: task_num
        mmoe_hidden_dims:
        mmoe_output_dim:
        training:
    Returns:
    """
    expert_outs = []
    for i in range(experts_num):
        expert_network = dnn('expert_' + str(i), input, mmoe_hidden_dims, expert_output_dim, None, training=training)  # [B, d]
        expert_outs.append(expert_network)
    expert_concat = tf.stack(expert_outs, axis=1)  # [B, experts_num, d]

    mmoe_output = []  # list with length of task_num, each element of [B, d]
    for i in range(gates_num):
        gate_output = dnn('gate_' + str(i), input, gate_hidden_dims, experts_num, None, training=training)  # [B, experts_num]
        gate_output = tf.nn.softmax(gate_output, axis=1)
        gate_output = tf.expand_dims(gate_output, axis=1)  # [B, 1, experts_num]

        outputs = tf.matmul(gate_output, expert_concat)  # [B, 1, d]
        shape = outputs.shape.as_list()
        outputs = tf.reshape(outputs, [-1, shape[1] * shape[2]])  # [B, d]
        mmoe_output.append(outputs)
    return mmoe_output
