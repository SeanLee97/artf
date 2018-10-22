# -*- coding: utf-8 -*-

import sys
sys.path.append("..")

import tensorflow as tf
import artf.transformer as transf

sess = tf.InteractiveSession()

inputs = tf.placeholder(tf.float32, shape=(32, 60, 256)) # 32 是 batch_size
outputs = transf.Encoder(activation=tf.nn.relu, num_blocks=4)(inputs, inputs.get_shape().as_list()[-1])

print(sess.run(tf.shape(inputs)))
print(sess.run(tf.shape(outputs)))
