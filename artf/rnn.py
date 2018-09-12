# !/usr/bin/env python3
# -*- coding: utf-8 -*-


class StackGRU(object):
    """
    Args :
        batch_size:
        input_size:
        num_units:   num of units of hidden
        num_layers:  layers of num [default 1]
        keep_prob:
        is_train: trainalble
        scope: 
        kernel: cudnn | native [default cudnn]
    """
    def __init__(self,  batch_size, input_size, num_units, num_layers=1, keep_prob=1.0, is_train=None, scope=None, kernel='cudnn'):
        self.num_layers = num_layers
        self.kernel = kernel
        self.grus = []
        self.inits = []
        self.dropout_mask = []
        for layer in range(num_layers):
            input_size_ = input_size if layer == 0 else 2 * num_units
            if self.kernel == 'cudnn':
                gru_fw = tf.contrib.cudnn_rnn.CudnnGRU(1, num_units)
                gru_bw = tf.contrib.cudnn_rnn.CudnnGRU(1, num_units)
            else:
                gru_fw = tf.contrib.rnn.GRUCell(num_units)
                gru_bw = tf.contrib.rnn.GRUCell(num_units)

            init_fw = tf.tile(tf.Variable(
                tf.zeros([1, 1, num_units])), [1, batch_size, 1])
            init_bw = tf.tile(tf.Variable(
                tf.zeros([1, 1, num_units])), [1, batch_size, 1])
            mask_fw = self.dropout(tf.ones([1, batch_size, input_size_], dtype=tf.float32),
                              keep_prob=keep_prob, is_train=is_train)
            mask_bw = self.dropout(tf.ones([1, batch_size, input_size_], dtype=tf.float32),
                              keep_prob=keep_prob, is_train=is_train)
            self.grus.append((gru_fw, gru_bw, ))
            self.inits.append((init_fw, init_bw, ))
            self.dropout_mask.append((mask_fw, mask_bw, ))

    def __call__(self, inputs, seq_len, keep_prob=1.0, is_train=None, concat_layers=True):
        outputs = [tf.transpose(inputs, [1, 0, 2])]
        for layer in range(self.num_layers):
            gru_fw, gru_bw = self.grus[layer]
            init_fw, init_bw = self.inits[layer]
            mask_fw, mask_bw = self.dropout_mask[layer]

            if self.kernel == 'cudnn':
                with tf.variable_scope("fw_{}".format(layer)):
                    out_fw, _ = gru_fw(
                        outputs[-1] * mask_fw, initial_state=(init_fw, ))
                with tf.variable_scope("bw_{}".format(layer)):
                    inputs_bw = tf.reverse_sequence(
                        outputs[-1] * mask_bw, seq_lengths=seq_len, seq_dim=0, batch_dim=1)
                    out_bw, _ = gru_bw(inputs_bw, initial_state=(init_bw, ))
                    out_bw = tf.reverse_sequence(
                        out_bw, seq_lengths=seq_len, seq_dim=0, batch_dim=1)
            else:
                with tf.variable_scope("fw_{}".format(layer)):
                    out_fw, _ = tf.nn.dynamic_rnn(
                        gru_fw, outputs[-1] * mask_fw, seq_len, initial_state=init_fw, dtype=tf.float32)
                with tf.variable_scope("bw_{}".format(layer)):
                    inputs_bw = tf.reverse_sequence(
                        outputs[-1] * mask_bw, seq_lengths=seq_len, seq_dim=1, batch_dim=0)
                    out_bw, _ = tf.nn.dynamic_rnn(
                        gru_bw, inputs_bw, seq_len, initial_state=init_bw, dtype=tf.float32)
                    out_bw = tf.reverse_sequence(
                        out_bw, seq_lengths=seq_len, seq_dim=1, batch_dim=0)

            outputs.append(tf.concat([out_fw, out_bw], axis=2))
        if concat_layers:
            res = tf.concat(outputs[1:], axis=2)
        else:
            res = outputs[-1]
        res = tf.transpose(res, [1, 0, 2])
        return res
    
    def dropout(args, keep_prob, is_train):
        #if keep_prob < 1.0:
        noise_shape = None
        scale = 1.0
        shape = tf.shape(args)
        args = tf.cond(is_train, lambda: tf.nn.dropout(
            args, keep_prob, noise_shape=noise_shape) * scale, lambda: args)
        return args
