# -*- coding: utf-8 -*-

"""封装了tensorflow常用的操作
"""

__version__ = '0.1.0'
__author__  = '[sean lee](seanlee97@gmail.com)'


import tensorflow as tf
from artf._libs import *

def mask(inputs, seq_len=None, mode='mul', mask_value=-1e12, name='mask', reuse=None):
    with tf.variable_scope(name, reuse=None):
        if seq_len is None:
            return inputs
        else:
            mask = tf.cast(tf.sequence_mask(seq_len), tf.float32)
            for _ in range(len(inputs.shape)-2):
                mask = tf.expand_dims(mask, 2)
            if mode == 'mul':
                return inputs * mask
            if mode == 'add':
                return inputs + mask_value * (1 - mask)

def dense(inputs, ouput_size, bias=True, seq_len=None, name='dense', reuse=None):
    with tf.variable_scope(name, reuse=None):
        input_size = int(inputs.shape[-1])
        W = tf.Variable(tf.random_uniform([input_size, ouput_size], -0.05, 0.05))
        if bias:
            b = tf.Variable(tf.random_uniform([ouput_size], -0.05, 0.05))
        else:
            b = 0
        outputs = tf.matmul(tf.reshape(inputs, (-1, input_size)), W) + b
        outputs = tf.reshape(outputs, \
                             tf.concat([tf.shape(inputs)[:-1], [ouput_size]], 0)
                            )
        if seq_len != None:
            outputs = self._mask(outputs, seq_len, 'mul')
        return outputs

def layer_norm(inputs, size=None, epsilon=1e-6, name='layer_norm', reuse=None):
    """Layer normalize the tensor inputs, averaging over the last dimension."""
    if size is None:
        size = inputs.get_shape()[-1]

    with tf.variable_scope(name, values=[inputs], reuse=reuse):
        W = tf.get_variable(
            "layer_norm_scale", [size], regularizer=regularizer, initializer=tf.ones_initializer())
        b = tf.get_variable(
            "layer_norm_bias", [size], regularizer=regularizer, initializer=tf.zeros_initializer())

        mean = tf.reduce_mean(inputs, axis=[-1], keep_dims=True)
        variance = tf.reduce_mean(tf.square(inputs - mean), axis=[-1], keep_dims=True)
        norm_x = (inputs - mean) * tf.rsqrt(variance + epsilon)
        result = norm_x * W + b
        return result

def group_norm(inputs, size=None, num_groups=8, epsilon=1e-5, name='group_norm', reuse=None):
    """Group normalization as in https://arxiv.org/abs/1803.08494."""
    inputs_shape = shape_list(inputs)
    if size is None:
        size = inputs_shape[-1]
    assert len(inputs_shape) == 4
    assert size % num_groups == 0
    with tf.variable_scope(name, reuse=None):
        W = tf.get_variable(
                "group_norm_W", [size], initializer=tf.ones_initializer())
        b = tf.get_variable(
                "group_norm_b", [size], initializer=tf.zeros_initializer())
        epsilon, W, b = [tf.cast(t, inputs.dtype) for t in [epsilon, W, b]]
        # Reshape and compute group norm.
        inputs = tf.reshape(inputs, inputs_shape[:-1] + [num_groups, size // num_groups])
        # Calculate mean and variance on heights, width, channels (not groups).
        mean, variance = tf.nn.moments(inputs, [1, 2, 4], keep_dims=True)
        norm_inputs = (inputs - mean) * tf.rsqrt(variance + epsilon)
        return tf.reshape(norm_inputs, inputs_shape) * W + b


def split_last_dimension(inputs, parts):
    """split the last dimension
    inputs: (...., m)
    outputs: (...., parts, m/parts)
    """
    prev_shape = inputs.get_shape().dims 
    last = prev_shape[-1]
    curr_shape = prev_shape[:-1] + [parts] + [last // parts if last else None]
    outputs = tf.reshape(inputs, tf.concat([tf.shape(inputs)[:-1], [parts, -1]], 0))
    outputs.set_shape(curr_shape)
    return tf.transpose(outputs, [0,2,1,3])

def ndims(inputs):
    dims = inputs.get_shape()._dims
    if dims is not None:
        return len(dims)
    return None

def shape_list(inputs):
    """return list of dims"""
    inputs = tf.convert_to_tensor(inputs)
    if inputs.get_shape().dims is None:
        return tf.shape(inputs)

    s = inputs.get_shape().as_list()
    shape = tf.shape(inputs)

    ret = []
    for i in range(len(s)):
        dim = s[i]
        if dim is None:
            dim = shape[i]
        ret.append(dim)
    return ret

def reshape_as(a, b):
    """reshapes a to match the shape of b"""
    ret = tf.reshape(a. tf.shape(b))
    if not tf.contrib.eager.in_eager_mode():
        ret.set_shape(b.get_shape())
    return ret

def dot(x, y):
    """Modified from keras==2.0.6
    Multiplies 2 tensors (and/or variables) and returns a *tensor*.

    When attempting to multiply a nD tensor
    with a nD tensor, it reproduces the Theano behavior.
    (e.g. `(2, 3) * (4, 3, 5) -> (2, 4, 5)`)

    # Arguments
        x: Tensor or variable.
        y: Tensor or variable.

    # Returns
        A tensor, dot product of `x` and `y`.
    """
    if ndim(x) is not None and (ndim(x) > 2 or ndim(y) > 2):
        x_shape = []
        for i, s in zip(x.get_shape().as_list(), tf.unstack(tf.shape(x))):
            if i is not None:
                x_shape.append(i)
            else:
                x_shape.append(s)
        x_shape = tuple(x_shape)
        y_shape = []
        for i, s in zip(y.get_shape().as_list(), tf.unstack(tf.shape(y))):
            if i is not None:
                y_shape.append(i)
            else:
                y_shape.append(s)
        y_shape = tuple(y_shape)
        y_permute_dim = list(range(ndim(y)))
        y_permute_dim = [y_permute_dim.pop(-2)] + y_permute_dim
        xt = tf.reshape(x, [-1, x_shape[-1]])
        yt = tf.reshape(tf.transpose(y, perm=y_permute_dim), [y_shape[-2], -1])
        return tf.reshape(tf.matmul(xt, yt),
                          x_shape[:-1] + y_shape[:-2] + y_shape[-1:])
    if isinstance(x, tf.SparseTensor):
        out = tf.sparse_tensor_dense_matmul(x, y)
    else:
        out = tf.matmul(x, y)
    return out

def batch_dot(x, y):
    """Copy from keras==2.0.6
    `batch_dot` is used to compute dot product of `x` and `y` when
    `x` and `y` are data in batch, i.e. in a shape of
    `(batch_size, :)`.
    `batch_dot` results in a tensor or variable with less dimensions
    than the input. If the number of dimensions is reduced to 1,
    we use `expand_dims` to make sure that ndim is at least 2.

    # Arguments
        x: Keras tensor or variable with `ndim >= 2`.
        y: Keras tensor or variable with `ndim >= 2`.
        axes: list of (or single) int with target dimensions.
            The lengths of `axes[0]` and `axes[1]` should be the same.

    # Returns
        A tensor with shape equal to the concatenation of `x`'s shape
        (less the dimension that was summed over) and `y`'s shape
        (less the batch dimension and the dimension that was summed over).
        If the final rank is 1, we reshape it to `(batch_size, 1)`.
    """
    if isinstance(axes, int):
        axes = (axes, axes)
    x_ndim = ndim(x)
    y_ndim = ndim(y)
    if x_ndim > y_ndim:
        diff = x_ndim - y_ndim
        y = tf.reshape(y, tf.concat([tf.shape(y), [1] * (diff)], axis=0))
    elif y_ndim > x_ndim:
        diff = y_ndim - x_ndim
        x = tf.reshape(x, tf.concat([tf.shape(x), [1] * (diff)], axis=0))
    else:
        diff = 0
    if ndim(x) == 2 and ndim(y) == 2:
        if axes[0] == axes[1]:
            out = tf.reduce_sum(tf.multiply(x, y), axes[0])
        else:
            out = tf.reduce_sum(tf.multiply(tf.transpose(x, [1, 0]), y), axes[1])
    else:
        if axes is not None:
            adj_x = None if axes[0] == ndim(x) - 1 else True
            adj_y = True if axes[1] == ndim(y) - 1 else None
        else:
            adj_x = None
            adj_y = None
        out = tf.matmul(x, y, adjoint_a=adj_x, adjoint_b=adj_y)
    if diff:
        if x_ndim > y_ndim:
            idx = x_ndim + y_ndim - 3
        else:
            idx = x_ndim - 1
        out = tf.squeeze(out, list(range(idx, idx + diff)))
    if ndim(out) == 1:
        out = tf.expand_dims(out, 1)
    return out

def total_params():
    total_params = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_params = 1
        for dim in shape:
            variable_params *= dim.value
        total_params += variable_params
    return total_params