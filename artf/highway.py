# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""implement highway by fully-connection and conv
"""

import tensorflow as tf 
from .nn import conv

class Highway(object):
    """highway
    Usage:
        highway = Highway(kernel=None, size=None, activation=None, num_layers=2,
                  name='scope_name', dropout=0.0, reuse=None)
        highway(inputs)
    """

    def __init__(self, kernel=None, **kwargs):
        self.kernel = kernel
        if kernel is None:
            self.kernel = 'conv'
        self.kwargs = kwargs

    def __call__(self, inputs):
        if 'conv' == self.kernel:
            return self._conv(inputs, **self.kwargs)
        elif 'fcn' == self.kernel:
            return self._fcn(inputs, **self.kwargs)
        elif 'fcn3d' == self.kernel:
            return self._fcn3d(inputs, **self.kwargs)
        else:
            raise ValueError('%s is a invalid kernel, kernel only support conv | fcn | fcn3d')

    def _fcn(self, inputs, size=None, activation=None, num_layers=2,
             name='highway-fcn', dropout=0.0, reuse=None):
        """
        implement highway by fully-connection
        """
        with tf.variable_scope(name, reuse=reuse):
            if size is None:
                size = inputs.shape.as_list()[-1]
            if activation is None:
                activation = tf.nn.relu

            curr_x = inputs
            curr_x = tf.reshape(curr_x, (-1, size))
            
            for i in range(num_layers):
                # init
                W = tf.Variable(
                    tf.truncated_normal(shape=[size, size], stddev=0.1),
                    name='weight_%d' % i
                )
                b = tf.Variable(
                    tf.constant(0.1, shape=[size]),
                    name='bias_%d' % i
                )
                W_T = tf.Variable(
                    tf.truncated_normal(shape=[size, size], stddev=0.1),
                    name='weight_transform_%d' % i
                )
                b_T = tf.Variable(
                    tf.constant(-0.1, shape=[size]),
                    name='bias_transform_%d' % i
                )
                H = activation(tf.matmul(curr_x, W)+b, name='activation_%d' % i)
                T = tf.sigmoid(tf.matmul(curr_x, W_T)+b_T, name='transorm_%d' % i)
                C = tf.subtract(tf.constant(1.0), T, name='gate_%d' % i)

                H = tf.nn.dropout(H, 1.0 - dropout)
                # curr_x = (H * T) + (x * C)
                curr_x = tf.add(tf.multiply(H, T), tf.multiply(curr_x, C))

            curr_x = tf.reshape(curr_x, tf.shape(inputs))
            return curr_x

    def _fcn3d(self, inputs, size=None, activation=None, num_layers=2,
               name='highway-fcn3d', dropout=0.0, reuse=None):
        """
        If the dimension of x is 3, you can use the function instead of fcn.
        Of course fcn is also okay.
        """
        # check shape
        shapes = inputs.shape.as_list()
        if len(shapes) != 3:
            raise ValueError("""Error: the dimension of input shouble be 3, but got %s 
                                [artf.highway.fcn3d]""" % len(shapes))

        if size is None:
            size = inputs.shape.as_list()[-1]
        if activation is None:
            activation = tf.nn.relu
        
        with tf.variable_scope(name, reuse=reuse):
            for i in range(num_layers):
                W = tf.Variable(
                    tf.truncated_normal(shape=[size, size], stddev=0.1), 
                    name='weight_%d' % i
                )
                b = tf.Variable(
                    tf.constant(0.1, shape=[size]), 
                    name='bias_%d' % i
                )
                W_T = tf.Variable(
                    tf.truncated_normal(shape=[size, size], stddev=0.1), 
                    name='weight_T_%d' % i
                )
                b_T = tf.Variable(
                    tf.constant(-0.1, shape=[size]), 
                    name='bias_T_%d' % i
                )

                shape = [tf.shape(inputs)[0], tf.shape(W)[0],tf.shape(W)[1]]
                W_ = tf.tile(W, [tf.shape(inputs)[0], 1])  
                W = tf.reshape(W_, shape) 
                W_T_ = tf.tile(W_T, [tf.shape(inputs)[0], 1])  
                W_T = tf.reshape(W_T_, shape)   

                H = activation(tf.matmul(inputs, W) + b, name='activation_%d' % i)
                T = tf.sigmoid(tf.matmul(inputs, W_T) + b_T, name='transform_%d' % i)
                C = tf.subtract(tf.constant(1.0), T, name='gate_%d' % i)
                H = tf.nn.dropout(H, 1.0 - dropout)

                inputs = tf.add(tf.multiply(H, T), tf.multiply(inputs, C)) # y = (H * T) + (inputs * C)
        return inputs
    
    def _conv(self, inputs, size=None, activation=None, num_layers=2, 
              name="highway-conv", dropout=0.0, reuse=None):

        with tf.variable_scope(name, reuse=reuse):
            if size is None:
                size = inputs.shape.as_list()[-1]
            else:
                inputs = conv(inputs, size, name = "input_projection", reuse = reuse)

            for i in range(num_layers):
                H = conv(inputs, size, bias = True, activation=activation,
                         name = "activation_%d" % i, reuse=reuse)
                T = conv(inputs, size, bias = True, activation=tf.sigmoid,
                         name = "gate_%d"%i, reuse=reuse)
                H = tf.nn.dropout(H, 1.0 - dropout)

                inputs = H * T + inputs * (1.0 - T)
            return inputs