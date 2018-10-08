# -*- coding: utf-8 -*-

import sys
sys.path.append("..")

import tensorflow as tf
from artf.conv import Conv

Conv.helper()

sess = tf.InteractiveSession()

input = tf.placeholder(tf.float32, shape=(32, 60, 256)) # 32 是 batch_size
output = Conv(activation=tf.nn.relu, kernel_size=1)(input, 64) # dense

print(sess.run(tf.shape(output)))
