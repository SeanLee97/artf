# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf 
from . import dense, mask

def multihead_attention(queries, keys, values, num_heads=8, num_units=None, bias=False,
                        dense_kernel=None, residual=False, Q_len=None, V_len=None, dropout=0.0,
                        name='multi_head_attnetion', reuse=None, is_training=False):
    """implement multi-head attention
    https://arxiv.org/abs/1706.03762
    """
    with tf.variable_scope(name, reuse=reuse):

        if num_units is None:
            # embedding as num_units
            num_units = queries.get_shape().as_list()[-1]

        if dense_kernel is None:
            dense_kernel = dense  # you can try conv

        # Dense
        Q = dense_kernel(queries, num_heads * num_units, bias=bias)
        K = dense_kernel(keys, num_heads * num_units, bias=bias)
        V = dense_kernel(values, num_heads * num_units, bias=bias)

        # reshape and transpose
        Q = tf.reshape(Q, (-1, tf.shape(Q)[1], num_heads, num_units))
        Q_T = tf.transpose(Q, [0, 2, 1, 3])

        K = tf.reshape(K, (-1, tf.shape(K)[1], num_heads, num_units))
        K_T = tf.transpose(K, [0, 2, 1, 3])

        V = tf.reshape(V, (-1, tf.shape(V)[1], num_heads, num_units))
        V_T = tf.transpose(V, [0, 2, 1, 3])

        # dot product
        Q_K = tf.matmul(Q_T, K_T, transpose_b=True) / tf.sqrt(float(num_units))
        Q_K = tf.transpose(Q_K, [0, 3, 2, 1])
        Q_K = mask(Q_K, V_len, mode='add')
        Q_K = tf.transpose(Q_K, [0, 3, 2, 1])
        Q_K = tf.nn.softmax(Q_K)

        Q_K_V = tf.matmul(Q_K, V_T)
        Q_K_V = tf.transpose(Q_K_V, [0, 2, 1, 3])
        Q_K_V = tf.reshape(Q_K_V, (-1, tf.shape(Q_K_V)[1], num_heads * num_units))
        Q_K_V = mask(Q_K_V, Q_len, 'mul')

        outputs = Q_K_V
        outputs = tf.nn.dropout(outputs, 1.0 - dropout)

        return outputs