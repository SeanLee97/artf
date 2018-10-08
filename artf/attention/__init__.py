# -*- coding: utf-8 -*-

import tensorflow as tf
import artf

def bi_attention(p_enc, q_enc, 
                 p_mask, q_mask,
                 kernel='bilinear', dropout=0.0):
    """
    Args: 
        - p_enc : (batch_size, p_len, embedding_size)
        - q_enc:  (batch_size, q_len, embedding_size)
        - p_mask: (batch_size, p_len)
        - q_mask: (batch_size, q_len)
        - kernel: str
            bilinear(default) | trilinear

    output:
        - p2q: (batch_size, p_len, embedding_size)
        - q2p: (batch_size, p_len, embedding_size)

    """

    p_len = tf.reduce_sum(tf.cast(p_enc, tf.int32), axis=1)
    q_len = tf.reduce_sum(tf.cast(q_enc, tf.int32), axis=1)
    
    if kernel == 'trilinear':
        p_ex = tf.tile(tf.expand_dims(p_enc, 2), [1, 1, q_len, 1])
        q_ex = tf.tile(tf.expand_dims(q_enc, 1), [1, p_len, 1, 1])
        score = artf.trilinear([p_ex, q_ex])

        q_mask_ex = tf.expand_dims(q_mask, 1)
        score_ = tf.nn.softmax(artf.mask_logits(score, mask=q_mask_ex))

        p_mask_ex = tf.expand_dims(p_mask, 2)
        score_t = tf.transpose(tf.nn.softmax(artf.mask_logits(score, mask=p_mask_ex), dim=1), (0, 2, 1))

        p2q = tf.matmul(score_, q_enc)
        q2p = tf.matmul(tf.matmul(score_, score_t), p_enc)
    else:
        score = artf.bilinear(p_enc, q_enc)
        q_mask_ex = tf.expand_dims(q_mask, 1) # batch x 1 x q_len
        p_mask_ex = tf.expand_dims(p_mask, 1) # batch x 1 x p_len

        score_ = tf.nn.softmax(tf.expand_dims(
                                tf.reduce_max(
                                    artf.mask_logits(score, mask=p_mask_ex), axis=1), 1), -1) # batch x 1 x p_len
        score_t = tf.nn.softmax(artf.mask_logits(tf.transpose(score, [0,2,1]), mask=q_mask_ex)) # batch x p_len x q_len

        p2q = tf.tile(tf.matmul(score_, p_enc), [1, tf.shape(p_enc)[1], 1]) # batch x p_len x embedding_size
        q2p = tf.matmul(score_t, q_enc) # batch x p_len x embedding_size

    return p2q, q2p
