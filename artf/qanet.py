# -*- coding: utf-8 -*-

"""
Refence to: https://github.com/NLPLearn/QANet
"""

import tensorflow as tf
import math
import artf
from artf.conv import Conv
from artf.attention.multihead_attention import MultiheadAttention


helper_doc="""\n[artf]> Implement QANet Residual Block

Args:
    ResidualBlock:
        - num_heads: int
            heads of multihead attention
        - num_blocks: int
        - num_conv_layers: int
            layers of conv block
        - activation:
        - dropout: float
            dropout rate
        -bias: bool
            wether to use bias

    __call__:
        - inputs: tensor
        - kernel_size: int
            size of conv kernel
        - num_filters: int
            the size of last dim of inputs, default None
        - input_mask:
            mask
        - scope: str
            scope name
        - reuse: bool
            wether to reuse

Output:
    the same shape as inputs

Usage:
    residual = ResidualBlock(num_heads=2,
                             num_blocks=4,
                             num_conv_layers=2,
                             activation=tf.nn.relu,
                             dropout=0.0,
                             bias=True)
    kernel_size = 5
    outputs = residual(inputs, kernel_size)
"""

class ResidualBlock(object):
    @staticmethod
    def helper():
        print(helper_doc)

    def __init__(self, 
                 num_heads=2,
                 num_blocks=4,
                 num_conv_layers=2,
                 activation=tf.nn.relu,
                 dropout=0.0,   
                 bias=True):

        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.num_conv_layers = num_conv_layers
        self.activation = activation
        self.dropout = dropout
        self.bias = bias

    def __call__(self, inputs, kernel_size, num_filters=None,
                 input_mask=None, scope="residual_block", 
                 reuse=None):

        with tf.variable_scope(scope, reuse = reuse):
            if num_filters is None:
                self.num_filters = inputs.get_shape().as_list()[-1]
            else:
                self.num_filters = num_filters

            outputs = inputs
            sublayer = 1
            total_sublayers = (self.num_conv_layers + 2) * self.num_blocks

            for i in range(self.num_blocks):
                outputs = self.add_timing_signal_1d(outputs)
                outputs, sublayer = self.conv_block(outputs, kernel_size,
                    scope="encoder_block_%d"%i, reuse=reuse,
                    sublayers=(sublayer, total_sublayers))

                outputs, sublayer = self.attention_block(outputs, 
                    input_mask=input_mask, scope="self_attention_layers%d"%i, reuse=reuse, 
                    bias=self.bias, sublayers=(sublayer, total_sublayers))

            return outputs


    def add_timing_signal_1d(self, x, min_timescale=1.0, max_timescale=1.0e4):
        length = tf.shape(x)[1]
        channels = tf.shape(x)[2]
        signal = self.get_timing_signal_1d(length, channels, min_timescale, max_timescale)
        return x + signal

    def get_timing_signal_1d(self, length, channels, min_timescale=1.0, max_timescale=1.0e4):
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

    def conv_block(self, inputs, kernel_size, 
                   scope="conv_block", reuse=None, bias=True, 
                   sublayers=(1, 1)):

        with tf.variable_scope(scope, reuse=reuse):
            outputs = tf.expand_dims(inputs,2)
            l, L = sublayers
            for i in range(self.num_conv_layers):
                residual = outputs
                if (i) % 2 == 0:
                    outputs = tf.nn.dropout(outputs, 1.0 - self.dropout)
                outputs = tf.contrib.layers.layer_norm(outputs, scope="layer_norm_%d"%i, reuse=reuse)
                outputs = self.depthwise_separable_convolution(outputs,
                    kernel_size=(kernel_size, 1), scope="depthwise_conv_layers_%d"%i, reuse=reuse)
                outputs = self.layer_dropout(outputs, residual, self.dropout * float(l) / L)
                l += 1
            return tf.squeeze(outputs, 2), l

    def attention_block(self, inputs, input_mask=None, 
                        scope="self_attention_ffn", reuse=None, 
                        bias=True, sublayers=(1, 1)):

        with tf.variable_scope(scope, reuse = reuse):
            l, L = sublayers
            # Self attention
            outputs = tf.nn.dropout(inputs, 1.0-self.dropout)
            outputs = tf.contrib.layers.layer_norm(outputs, scope = "layer_norm_1", reuse = reuse)

            attn_fn = MultiheadAttention(num_heads=self.num_heads, dropout=self.dropout)
            outputs = attn_fn(outputs, outputs, outputs, 
                              num_units=self.num_filters,
                              query_mask=input_mask,
                              value_mask=input_mask,
                              reuse=reuse)
            residual = self.layer_dropout(outputs, inputs, self.dropout * float(l) / L)
            l += 1
            # Feed-forward
            outputs = tf.nn.dropout(residual, 1.0-self.dropout)
            outputs = tf.contrib.layers.layer_norm(outputs, scope = "layer_norm_2", reuse = reuse)
            outputs = Conv(activation=self.activation, bias=True)(outputs, self.num_filters, scope="FFN_1", reuse=reuse)
            outputs = Conv(bias=True)(outputs, self.num_filters, scope="FFN_2", reuse=reuse)
            outputs = self.layer_dropout(outputs, residual, self.dropout * float(l) / L)
            l += 1
            return outputs, l

    def depthwise_separable_convolution(self, inputs, kernel_size,
                                        scope="depthwise_separable_convolution",
                                        bias=True, reuse=None):

        with tf.variable_scope(scope, reuse = reuse):
            shapes = inputs.shape.as_list()
            depthwise_filter = tf.get_variable("depthwise_filter",
                                            (kernel_size[0], kernel_size[1], shapes[-1], 1),
                                            dtype = tf.float32,
                                            regularizer=tf.contrib.layers.l2_regularizer(scale = 3e-7),
                                            initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,
                                                             mode='FAN_AVG',
                                                             uniform=True,
                                                             dtype=tf.float32))
            pointwise_filter = tf.get_variable("pointwise_filter",
                                            (1, 1, shapes[-1], self.num_filters),
                                            dtype = tf.float32,
                                            regularizer=tf.contrib.layers.l2_regularizer(scale = 3e-7),
                                            initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,
                                                             mode='FAN_AVG',
                                                             uniform=True,
                                                             dtype=tf.float32))
            outputs = tf.nn.separable_conv2d(inputs,
                                            depthwise_filter,
                                            pointwise_filter,
                                            strides=(1,1,1,1),
                                            padding="SAME")
            if bias:
                b = tf.get_variable("bias",
                        outputs.shape[-1],
                        regularizer=tf.contrib.layers.l2_regularizer(scale = 3e-7),
                        initializer=tf.zeros_initializer())
                outputs += b
            outputs = tf.nn.relu(outputs)

            return outputs

    def layer_dropout(self, inputs, residual, dropout):
        pred = tf.random_uniform([]) < dropout
        return tf.cond(pred, lambda: residual, lambda: tf.nn.dropout(inputs, 1.0 - dropout) + residual)
