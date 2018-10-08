# -*- coding: utf-8 -*-

import sys
sys.path.append("..")

import tensorflow as tf
from artf.highway import Highway

Highway.helper()

sess = tf.InteractiveSession()

inputs = tf.placeholder(tf.float32, shape=(32, 60, 256)) # 32 是 batch_size
output = Highway(activation=tf.nn.relu, kernel="fcn3d", num_layers=2)(inputs) # dense

print(sess.run(tf.shape(output)))
