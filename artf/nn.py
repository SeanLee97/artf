# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import tensorflow as tf 
from functools import reduce
from operator import mul

from tensorflow.python.util import nest
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import clip_ops

from artf._libs import *
from artf import layer_norm
from artf.attention import multihead_attention

def glu(inputs, name='glu', reuse=None):
    """Gated Linear Units
    Split x into two parts along last dimension
    """
    with tf.variable_scope(name, reuse=reuse):
        x1, x2 = tf.split(inputs, 2, axis=-1)
        return tf.sigmoid(x1) * x2 

def leaky_relu(inputs, alpha=0.2, name='leaky_relu', reuse=None):
    with tf.variable_scope(name, reuse=reuse):
        return tf.maximum(alpha * inputs, inputs)

def position_embedding(inputs, position_dim):
    """position embedding
    inputs: (batch_size, seq_len, word_dim)
    outputs: (batch_size, seq_len, position_dim)
    """
    batch_size,seq_len = tf.shape(inputs)[0],tf.shape(inputs)[1]
    pos_j = 1. / tf.pow(10000.0, 
                        2 * tf.range(position_dim / 2, dtype=tf.float32 
                        ) / position_dim)
    pos_j = tf.expand_dims(position_j, 0)
    pos_i = tf.range(tf.cast(seq_len, tf.float32), dtype=tf.float32)
    pos_i = tf.expand_dims(position_i, 1)
    pos_ij = tf.matmul(position_i, position_j)
    pos_ij = tf.concat([tf.cos(position_ij), tf.sin(position_ij)], 1)
    outputs = tf.expand_dims(position_ij, 0) \
                         + tf.zeros((batch_size, seq_len, position_dim))
    return outputs

def conv(inputs, out_size, bias=None, activation=None, 
         kernel_size=1, name='conv', reuse=None):
    
    with tf.variable_scope(name, reuse=reuse):
        shapes = inputs.shape.as_list()
        if len(shapes) > 4:
            raise NotImplementedError
        elif len(shapes) == 4:
            filter_shape = [1, kernel_size, shapes[-1], out_size]
            bias_shape = [1, 1, 1, out_size]
            strides = [1, 1, 1, 1]
        else:
            filter_shape = [kernel_size, shapes[-1], out_size]
            bias_shape = [1, 1, out_size]
            strides = 1
        conv_func = tf.nn.conv1d if len(shapes) == 3 else tf.nn.conv2d
        kernel_ = tf.get_variable("kernel_",
                        filter_shape,
                        dtype=tf.float32,
                        regularizer=regularizer,
                        initializer=initializer_relu() if activation is not None else initializer())
        outputs = conv_func(inputs, kernel_, strides, "VALID")
        if bias:
            outputs += tf.get_variable("bias_",
                        bias_shape,
                        regularizer=regularizer,
                        initializer = tf.zeros_initializer())
        if activation is not None:
            return activation(outputs)
        else:
            return outputs

def residual_block(inputs, num_blocks, num_conv_layers, kernel_size, 
                   num_filters=128, input_projection=False, num_heads = 8,
                   seq_len=None, name="res_block", is_training=True,
                   reuse=None, bias=True, dropout=0.0):
    
    with tf.variable_scope(name, reuse=reuse):

        if input_projection:
            inputs = conv(inputs, num_filters, name = "input_projection", reuse = reuse)
        outputs = inputs
        sublayer = 1
        total_sublayers = (num_conv_layers + 2) * num_blocks
        for i in range(num_blocks):
            outputs = add_timing_signal_1d(outputs)
            outputs, sublayer = conv_block(outputs, kernel_size, num_conv_layers, num_filters,
                seq_len=seq_len, name="encoder_block_%d"%i,reuse = reuse, bias = bias,
                dropout = dropout, sublayers = (sublayer, total_sublayers))
            outputs, sublayer = self_attention_block(outputs, num_filters, seq_len, num_heads = num_heads,
                name="self_attention_layers_%d"%i, reuse=reuse, is_training=is_training,
                bias=bias, dropout=dropout, sublayers=(sublayer, total_sublayers))
        return outputs

def conv_block(inputs, kernel_size, num_layers, num_filters=None, 
               seq_len=None, name='conv_block', is_training=True,
               reuse=None, bias=True, dropout=0.0, sublayers=(1, 1)):

    with tf.variable_scope(name, reuse=reuse):

        if num_filters is None:
            num_filters = inputs.get_shape().as_list()[-1]

        outputs = tf.expand_dims(inputs, 2)
        l, L = sublayers
        for i in range(num_layers):
            residual = outputs
            outputs = layer_norm(outputs, name='layer_norm_%d' % i, reuse=reuse)
            if i % 2 == 0:
                outputs = tf.nn.dropout(outputs, 1.0-dropout)
            outputs = depthwise_separable_conv(outputs, kernel_size=(kernel_size, 1),
                                               num_filters=num_filters, name='depth_conv_%d' % i,
                                               is_training=is_training, reuse=reuse)
            outputs = layer_dropout(outputs, residual, dropout * float(l) / L)
            l += 1
        return tf.squeeze(outputs, 2), l

def self_attention_block(inputs, num_filters=None, seq_len=None, num_heads=8, 
                         name='self_attention_block', reuse=None, is_training=True,
                         bias=True, dropout=0.0, sublayers=(1, 1)):

    """residual self attention"""
    with tf.variable_scope(name, reuse=reuse):

        if num_filters is None:
            num_filters = inputs.get_shape().as_list()[-1]

        l, L = sublayers
        outputs = layer_norm(inputs, name='layer_norm_1', reuse=reuse)
        outputs = tf.nn.dropout(outputs, 1.0-dropout)
        outputs = multihead_attention(outputs, outputs, outputs, num_heads=num_heads, num_units=num_filters, 
                                      Q_len=seq_len, V_len=seq_len, reuse=reuse, bias=bias, 
                                      dropout=dropout, is_training=is_training)
        residual = layer_dropout(outputs, inputs, dropout * float(l) / L)
        l += 1
        outputs = layer_norm(residual, name='layer_norm_2', reuse=reuse)
        outputs = tf.nn.dropout(outputs, 1.0-dropout)
        outputs = conv(outputs, num_filters, bias=True, activation=tf.nn.relu, 
                       name='ffn_2', reuse=reuse)
        outputs = conv(outputs, num_filters, bias=True, name='ffn_1', reuse=reuse)
        outputs = layer_dropout(outputs, residual, dropout * float(l) / L)
        l += 1
        return outputs, l

def layer_dropout(inputs, residual, dropout):
    pred = tf.random_uniform([]) < dropout
    #return: either fn1() or fn2() based on the boolean predicate `pred`
    return tf.cond(pred, lambda: residual, lambda: tf.nn.dropout(inputs, 1.0 - dropout) + residual)
    
def depthwise_separable_conv(inputs, kernel_size, num_filters,
                             name="depthwise_separable_conv",
                             bias=True, is_training=True, reuse=None):
    with tf.variable_scope(name, reuse = reuse):
        shapes = inputs.shape.as_list()

        depthwise_filter = tf.get_variable("depthwise_filter",
                                        (kernel_size[0], kernel_size[1], shapes[-1], 1),
                                        dtype = tf.float32,
                                        regularizer=regularizer,
                                        initializer = initializer_relu())
        pointwise_filter = tf.get_variable("pointwise_filter",
                                        (1,1,shapes[-1],num_filters),
                                        dtype = tf.float32,
                                        regularizer=regularizer,
                                        initializer = initializer_relu())
        outputs = tf.nn.separable_conv2d(inputs,
                                        depthwise_filter,
                                        pointwise_filter,
                                        strides = (1,1,1,1),
                                        padding = "SAME")
        if bias:
            b = tf.get_variable("bias",
                    outputs.shape[-1],
                    regularizer=regularizer,
                    initializer = tf.zeros_initializer())
            outputs += b
        outputs = tf.nn.relu(outputs)
        return outputs

def add_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4):
    """Adds a bunch of sinusoids of different frequencies to a Tensor.
    Each channel of the input Tensor is incremented by a sinusoid of a different
    frequency and phase.
    This allows attention to learn to use absolute and relative positions.
    Timing signals should be added to some precursors of both the query and the
    memory inputs to attention.
    The use of relative position is possible because sin(x+y) and cos(x+y) can be
    experessed in terms of y, sin(x) and cos(x).
    In particular, we use a geometric sequence of timescales starting with
    min_timescale and ending with max_timescale.  The number of different
    timescales is equal to channels / 2. For each timescale, we
    generate the two sinusoidal signals sin(timestep/timescale) and
    cos(timestep/timescale).  All of these sinusoids are concatenated in
    the channels dimension.
    Args:
    x: a Tensor with shape [batch, length, channels]
    min_timescale: a float
    max_timescale: a float
    Returns:
    a Tensor the same shape as x.
    """
    length = tf.shape(x)[1]
    channels = tf.shape(x)[2]
    signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)
    return x + signal

def get_timing_signal_1d(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    """Gets a bunch of sinusoids of different frequencies.
    Each channel of the input Tensor is incremented by a sinusoid of a different
    frequency and phase.
    This allows attention to learn to use absolute and relative positions.
    Timing signals should be added to some precursors of both the query and the
    memory inputs to attention.
    The use of relative position is possible because sin(x+y) and cos(x+y) can be
    experessed in terms of y, sin(x) and cos(x).
    In particular, we use a geometric sequence of timescales starting with
    min_timescale and ending with max_timescale.  The number of different
    timescales is equal to channels / 2. For each timescale, we
    generate the two sinusoidal signals sin(timestep/timescale) and
    cos(timestep/timescale).  All of these sinusoids are concatenated in
    the channels dimension.
    Args:
    length: scalar, length of timing signal sequence.
    channels: scalar, size of timing embeddings to create. The number of
        different timescales is equal to channels / 2.
    min_timescale: a float
    max_timescale: a float
    Returns:
    a Tensor of timing signals [1, length, channels]
    """
    position = tf.to_float(tf.range(length))
    num_timescales = channels // 2
    log_timescale_increment = (
        math.log(float(max_timescale) / float(min_timescale)) /
            (tf.to_float(num_timescales) - 1))
    inv_timescales = min_timescale * tf.exp(
        tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
    signal = tf.reshape(signal, [1, length, channels])
    return signal

def mask_logits(inputs, mask, mask_value = -1e30):
    shapes = inputs.shape.as_list()
    mask = tf.cast(mask, tf.float32)
    return inputs*mask + mask_value * (1 - mask)


def trilinear(args, output_size = 1, bias = True, squeeze=False, wd=0.0, input_keep_prob= 1.0, scope = "trilinear"):
    def _reconstruct(tensor, ref, keep):
        ref_shape = ref.get_shape().as_list()
        tensor_shape = tensor.get_shape().as_list()
        ref_stop = len(ref_shape) - keep
        tensor_start = len(tensor_shape) - keep
        pre_shape = [ref_shape[i] or tf.shape(ref)[i] for i in range(ref_stop)]
        keep_shape = [tensor_shape[i] or tf.shape(tensor)[i] for i in range(tensor_start, len(tensor_shape))]
        target_shape = pre_shape + keep_shape
        out = tf.reshape(tensor, target_shape)
        return out

    def _flatten(tensor, keep):
        fixed_shape = tensor.get_shape().as_list()
        start = len(fixed_shape) - keep
        left = reduce(mul, [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start)])
        out_shape = [left] + [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start, len(fixed_shape))]
        flat = tf.reshape(tensor, out_shape)
        return flat

    def _linear(args, output_size, bias, bias_initializer=tf.zeros_initializer(), 
                scope = None, kernel_initializer=initializer(), reuse = None):

        if args is None or (nest.is_sequence(args) and not args):
            raise ValueError("`args` must be specified")
        if not nest.is_sequence(args):
            args = [args]

        # Calculate the total size of arguments on dimension 1.
        total_arg_size = 0
        shapes = [a.get_shape() for a in args]
        for shape in shapes:
            if shape.ndims != 2:
                raise ValueError("linear is expecting 2D arguments: %s" % shapes)
            if shape[1].value is None:
                raise ValueError("linear expects shape[1] to be provided for shape %s, "
                                                 "but saw %s" % (shape, shape[1]))
            else:
                total_arg_size += shape[1].value

        dtype = [a.dtype for a in args][0]

        # Now the computation.
        with tf.variable_scope(scope, reuse = reuse) as outer_scope:
            weights = tf.get_variable(
                    "linear_kernel", [total_arg_size, output_size],
                    dtype=dtype,
                    regularizer=regularizer,
                    initializer=kernel_initializer)
            if len(args) == 1:
                res = math_ops.matmul(args[0], weights)
            else:
                res = math_ops.matmul(array_ops.concat(args, 1), weights)
            if not bias:
                return res
            with tf.variable_scope(outer_scope) as inner_scope:
                inner_scope.set_partitioner(None)
                biases = tf.get_variable(
                        "linear_bias", [output_size],
                        dtype=dtype,
                        regularizer=regularizer,
                        initializer=bias_initializer)
            return nn_ops.bias_add(res, biases)

    with tf.variable_scope(scope):
        flat_args = [_flatten(arg, 1) for arg in args]
        flat_args = [tf.nn.dropout(arg, input_keep_prob) for arg in flat_args]
        flat_out = _linear(flat_args, output_size, bias, scope=scope)
        out = _reconstruct(flat_out, args[0], 1)
        return tf.squeeze(out, -1)
