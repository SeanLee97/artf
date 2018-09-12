# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf 

from . import dense, mask
from .nn import mask_logits, trilinear

def multihead_attention(queries, keys, values, num_heads=8, num_units=None, bias=False,
                        dense_kernel=None, residual=False, Q_len=None, V_len=None, keep_prob=1.0,
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
        outputs = tf.nn.dropout(outputs, keep_prob)

def dot_attention(inputs, memory, mask, hidden, keep_prob=1.0, is_train=None, scope="dot_attention"):
    with tf.variable_scope(scope):

        d_inputs = dropout(inputs, keep_prob=keep_prob, is_train=is_train)
        d_memory = dropout(memory, keep_prob=keep_prob, is_train=is_train)
        JX = tf.shape(inputs)[1]

        with tf.variable_scope("attention"):
            inputs_ = tf.nn.relu(
                dense(d_inputs, hidden, use_bias=False, scope="inputs"))
            memory_ = tf.nn.relu(
                dense(d_memory, hidden, use_bias=False, scope="memory"))
            outputs = tf.matmul(inputs_, tf.transpose(
                memory_, [0, 2, 1])) / (hidden ** 0.5)
            mask = tf.tile(tf.expand_dims(mask, axis=1), [1, JX, 1])
            logits = tf.nn.softmax(softmax_mask(outputs, mask))
            outputs = tf.matmul(logits, memory)
            res = tf.concat([inputs, outputs], axis=2)

        with tf.variable_scope("gate"):
            dim = res.get_shape().as_list()[-1]
            d_res = dropout(res, keep_prob=keep_prob, is_train=is_train)
            gate = tf.nn.sigmoid(dense(d_res, dim, use_bias=False))
            return res * gate

        return outputs

def bi_attention(p_enc, q_enc, p_len, q_len, p_mask, q_mask, keep_prob=1.0):

    p = tf.tile(tf.expand_dims(p_enc, 2), [1, 1, q_len, 1])
    q = tf.tile(tf.expand_dims(q_enc, 1), [1, p_len, 1, 1])
    
    S = trilinear([p, q, p*q], input_keep_prob = keep_prob)
    mask_q = tf.expand_dims(q_mask, 1)
    S_ = tf.nn.softmax(mask_logits(S, mask = mask_q))
    mask_p = tf.expand_dims(p_mask, 2)
    S_T = tf.transpose(tf.nn.softmax(mask_logits(S, mask = mask_p), dim = 1),(0,2,1))

    alpha_p = S_
    alpha_q = tf.matmul(S_, S_T)

    p2q = tf.matmul(alpha_p, q_enc)
    q2p = tf.matmul(alpha_q, p_enc)

    return alpha_p, p2q, alpha_q, q2p
