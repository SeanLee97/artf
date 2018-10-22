# -*- coding: utf-8 -*-

import tensorflow as tf
import artf
from artf.attention.multihead_attention import MultiheadAttention

helper_doc="""\n[artf]> Implement Transformer Encoder

Args:
    Encoder:
        - num_heads: int
            heads of multihead attention, default 8
        - num_blocks: int
            default 4
        - activation: non-linear activation function
            default tf.nn.relu
        - bias: bool
            whether to use bias
        - dropout: float
            dropout rate

    __call__:
        - inputs: tensor
        - num_units: int
        - input_mask: 
        - scope: str
        - reuse: bool

Usage:
    transEnc = transformer.Encoder(num_heads=8,
                                   num_blocks=4,
                                   activation=tf.nn.relu,
                                   dropout=0.0,
                                   bias=False)
    output = transEnc(inputs, num_units, 
                      input_mask=None, 
                      scope='transformer_encoder', 
                      reuse=None)

"""

class Encoder(object):
    @staticmethod
    def helper():
        print(helper_doc)

    def __init__(self, 
                 num_heads=8,
                 num_blocks=4,
                 activation=tf.nn.relu,
                 dropout=0.0,   
                 bias=False):

        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.activation = activation
        self.bias = bias
        self.dropout = dropout

    def feedforward(self, inputs, num_units):
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                  "activation": self.activation, "use_bias": self.bias}

        hidden = tf.layers.conv1d(**params)
        
        params = {"inputs": hidden, "filters": num_units[1], "kernel_size": 1,
                  "activation": None, "use_bias": True}

        outputs = tf.layers.conv1d(**params)

        # residual connection
        outputs += inputs

        # normalization
        outputs = tf.contrib.layers.layer_norm(outputs)

        return outputs

    def __call__(self, inputs, num_units, input_mask=None, 
                 scope="transformer_encoder", reuse=None):

        with tf.variable_scope(scope, reuse = reuse):

            enc = inputs

            for i in range(self.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i)):                
                    # multihead_attention
                    enc = MultiheadAttention(num_heads=self.num_heads, 
                                             dropout=self.dropout)(enc, enc, enc, 
                                                                   num_units=num_units, 
                                                                   query_mask=input_mask,
                                                                   value_mask=input_mask)

                    # feed_forward
                    enc = self.feedforward(enc, [4*num_units, num_units])

            return enc
            
