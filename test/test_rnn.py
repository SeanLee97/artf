# -*- coding: utf-8 -*-

import sys
sys.path.append("..")

import tensorflow as tf
from artf.rnn import RNN, BiRNN, CudnnRNN, BiCudnnRNN

# helper
RNN.helper()

sess = tf.InteractiveSession()

X = tf.placeholder(tf.float32, shape=(32,40,5)) # 32 是 batch_size

# rnn
num_units = 128
batch_size = X.get_shape()[0]
input_size = X.get_shape()[-1]
rnn = BiRNN(num_units, batch_size, input_size, kernel='lstm', num_layers=2, dropout=0.5)

output, state, h  = rnn(X, batch_first=True)

print(sess.run(tf.shape(output)))
print(sess.run(tf.shape(state)))
