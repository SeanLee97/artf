# -*- coding: utf-8 -*-

import sys
sys.path.append("..")

import tensorflow as tf
from artf.qanet import ResidualBlock

ResidualBlock.helper()

sess = tf.InteractiveSession()

inputs = tf.placeholder(tf.float32, shape=(32, 60, 256)) # 32 是 batch_size

outputs = ResidualBlock(num_heads=2,
                        num_blocks=4,
                        num_conv_layers=2,
                        activation=tf.nn.relu,
                        dropout=0.0,   
                        bias=True)(inputs, 5) # dense

print(sess.run(tf.shape(outputs)))
