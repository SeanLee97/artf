# -*- coding: utf-8 -*-

import tensorflow as tf

helper_doc = """\n[artf]>  Implement Multihead Attention.
    
Args:
    MultiheadAttention:
        - num_heads: int (default 8)
            number of heads
        - dropout: float
            dropout rate
        - causality: bool
            If true, units that reference the future are masked.
    __call__:
        - queries: tensor
            (batch_size, q_len, embed_size)
        - keys: tensor
            (batch_size, k_len, embed_size)
        - values: tensor
            (batch_size, v_len, embed_size)
        - num_units: int (default None)
            the last dim of queries
        - scope: str
        - reuse: bool
            whether to reuse the weights of a previous layer
        
Usage:
    attention = MultiheadAttention(num_heads=8, dropout=0.0, causality=False)
    output = attention(query, key, values, num_units=8, scope="multihead_attention", reuse=None)

Outputs:
    (batch_size, q_len , embed_size)  
"""

class MultiheadAttention(object):
    @staticmethod
    def helper():
        print(helper_doc)

    def __init__(self, 
                 num_heads=8, 
                 dropout=0.0, 
                 causality=False):

        self.num_heads = num_heads
        self.dropout = dropout
        self.causality = causality

    def __call__(self, 
                 queries, 
                 keys, 
                 values,
                 num_units=None, 
                 scope="multihead_attention", 
                 reuse=None):

        with tf.variable_scope(scope, reuse=reuse):

            if num_units is None:
                num_units = queries.get_shape().as_list()[-1]
        
            # projection
            Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu) # (N, T_q, C)
            K = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
            V = tf.layers.dense(values, num_units, activation=tf.nn.relu) # (N, T_k, C)
        
            # Split and concat
            Q_ = tf.concat(tf.split(Q, self.num_heads, axis=2), axis=0) # (h*N, T_q, C/h) 
            K_ = tf.concat(tf.split(K, self.num_heads, axis=2), axis=0) # (h*N, T_k, C/h) 
            V_ = tf.concat(tf.split(V, self.num_heads, axis=2), axis=0) # (h*N, T_k, C/h) 

            # Multiplication
            outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1])) # (h*N, T_q, T_k)
        
            # Scale
            outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
        
            # Key Masking
            key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1))) # (N, T_k)
            key_masks = tf.tile(key_masks, [self.num_heads, 1]) # (h*N, T_k)
            key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1]) # (h*N, T_q, T_k)
        
            paddings = tf.ones_like(outputs)*(-2**32+1)
            outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs) # (h*N, T_q, T_k)
  
            # Causality = Future blinding
            if self.causality:
                diag_vals = tf.ones_like(outputs[0, :, :]) # (T_q, T_k)
                tril = tf.contrib.linalg.LinearOperatorTriL(diag_vals).to_dense() # (T_q, T_k)
                masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1]) # (h*N, T_q, T_k)
   
                paddings = tf.ones_like(masks)*(-2**32+1)
                outputs = tf.where(tf.equal(masks, 0), paddings, outputs) # (h*N, T_q, T_k)
  
            # Activation
            outputs = tf.nn.softmax(outputs) # (h*N, T_q, T_k)
         
            # Query Masking
            query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1))) # (N, T_q)
            query_masks = tf.tile(query_masks, [self.num_heads, 1]) # (h*N, T_q)
            query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]]) # (h*N, T_q, T_k)
            outputs *= query_masks # broadcasting. (N, T_q, C)
          
            # Dropouts
            outputs = tf.nn.dropout(outputs, 1.0 - self.dropout)
               
            # Weighted sum
            outputs = tf.matmul(outputs, V_) # ( h*N, T_q, C/h)
        
            # Restore shape
            outputs = tf.concat(tf.split(outputs, self.num_heads, axis=0), axis=2 ) # (N, T_q, C)
              
            # Residual connection
            outputs += queries
              
            # Normalize
            outputs = tf.contrib.layers.layer_norm(outputs) # (N, T_q, C)
 
        return outputs
