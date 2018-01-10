#!/usr/bin/env python
# encoding: utf-8

import tensorflow as tf
import numpy as np
import tflearn
import os

class WaveNet(object):

    def __init__(self, n_out, batch_size = 1, n_mfcc = 20, hidden_dim = 128, n_blocks = 3, learning_rate = 0.0003, is_training = True):
        self.n_out = n_out
        self.batch_size = batch_size
        self.n_mfcc = n_mfcc
        self.hidden_dim = hidden_dim
        self.n_blocks = n_blocks
        self.learning_rate = learning_rate
        self.net = None
        self.sess = None

    def init_sess(self):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def restore(self, chpt_path):
        self.sess = tf.Session()
        self.saver.restore(self.sess, chpt_path)

    def build_graph(self):

        self.conv1d_idx = 0
        self.atrous2d_idx = 0
        #self.input_ = tflearn.input_data(shape = [self.batch_size, None, self.n_mfcc])
        #self.targets = tflearn.input_data(shape = [self.batch_size, None])
        self.input_ = tf.placeholder(shape = [self.batch_size, None, self.n_mfcc], dtype = tf.float32)
        self.targets = tf.placeholder(shape = [self.batch_size, None], dtype = tf.int32)
        self.seq_len = tf.reduce_sum(tf.cast(tf.not_equal(tf.reduce_sum(self.input_, reduction_indices=2), 0.), tf.int32), reduction_indices=1)
        # conv 1d with batch nomalization, filter kernel is [1, n_mfcc].
        net = self.conv1d_bn(self.input_, size = 1, dim = self.hidden_dim)

        # stack hole cnn
        skip = 0
        for _ in range(self.n_blocks):
            for r in [1, 2, 4, 8, 16]:
                net, s = self.residual_block(net, size = 7, rate = r, dim = self.hidden_dim)
                skip += s

        self.net = net
        # get logit
        logit = self.conv1d_bn(skip, dim = skip.get_shape().as_list()[-1])
        self.logit = tflearn.conv_1d(logit, self.n_out,
                        [1, logit.get_shape().as_list()[-1]], bias = True, activation = 'relu')

        # CTC loss
        indices = tf.where(tf.not_equal(tf.cast(self.targets, tf.float32), 0.))
        target = tf.SparseTensor(indices = indices, values = tf.gather_nd(self.targets, indices)-1,
                        dense_shape = tf.cast(tf.shape(self.targets), tf.int64))
        loss = tf.nn.ctc_loss(target, self.logit, self.seq_len, time_major = False)
        self.cost = tf.reduce_mean(loss)

        # accuracy
        #self.accuracy = tf.reduce_mean(
        #        tf.cast(tf.equal(tf.argmax(self.logit, 2), self.targets), tf.float32), name = 'acc')
        self.accuracy = None

        # optimizer
        # optimizer = tf.train.AdamOptimizer()
        # var_list = [ var for var in tf.trainable_variables()]
        # gradient = optimizer.compute_gradients(self.cost, var_list = var_list)
        # self.optimizer_op = optimizer.apply_gradients(gradient)
        self.optimizer_op = tf.train.AdamOptimizer(self.learning_rate)

        self.saver = tf.train.Saver(tf.global_variables())

    def conv1d_bn(self, input_tensor, size = 1, dim = 128, activation = 'tanh'):
        with tf.variable_scope('conv1d_bn' + str(self.conv1d_idx)):
            shape = input_tensor.get_shape().as_list()
            channels = shape[-1]
            net = tflearn.conv_1d(input_tensor, dim, [size, channels], strides = 1,
                bias = False, padding = 'same', activation = activation)
            net = tflearn.batch_normalization(net)

            self.conv1d_idx += 1
            return net

    def aconv1d_bn(self, input_tensor, size = 7, dim = 1, rate = 2, activation = 'tanh'):
        with tf.variable_scope('aconv1d_bn' + str(self.atrous2d_idx)):
            shape = input_tensor.get_shape().as_list()
            net = tflearn.layers.conv.atrous_conv_2d(tf.expand_dims(input_tensor, dim = 1), shape[-1],
                [1, size], bias = False, activation = activation)
            net = tf.squeeze(net, [1])
            net = tflearn.batch_normalization(net)

            self.atrous2d_idx += 1
            return net

    def residual_block(self, input_tensor, size, rate, dim):
        conv_filter = self.aconv1d_bn(input_tensor, size = size, rate = rate, activation = 'tanh')
        conv_gate = self.aconv1d_bn(input_tensor, size = size, rate = rate, activation = 'sigmoid')
        net = conv_filter * conv_gate
        net = self.conv1d_bn(net, size = 1, dim = dim)

        return net + input_tensor, net

    def predict(self, wave_features):
        decoded = tf.transpose(self.logit, perm = [1, 0, 2])
        decoded, prob = tf.nn.ctc_beam_search_decoder(decoded, self.seq_len, top_paths = 1, merge_repeated = True)
        inference = tf.sparse_to_dense(decoded[0].indices, decoded[0].dense_shape, decoded[0].values) + 1
        output, prob =  self.sess.run([inference, prob], feed_dict = {self.input_: wave_features})

        return output


    def train_val(self, X, Y, ckpt_dir = None, n_epoch = 10, val_rate = 0.1, log_path = './log_dir'):
        ckpt_path = os.path.join(ckpt_dir, "wavenet.ckpt")
        best_ckpt_path = os.path.join(ckpt_dir, "best_wavenet.ckpt")
        trainop = tflearn.TrainOp(loss = self.cost, optimizer = self.optimizer_op,
                                  metric = self.accuracy, batch_size = self.batch_size)
        trainer = tflearn.Trainer(train_ops = trainop, best_checkpoint_path = best_ckpt_path,
                                  checkpoint_path = ckpt_path, keep_checkpoint_every_n_hours = 0.5,
                                  tensorboard_dir = log_path, tensorboard_verbose = 0, session = self.sess)
        self.sess.run(tf.global_variables_initializer())
        trainer.fit({self.input_: X, self.targets: Y}, val_feed_dicts = val_rate, n_epoch = n_epoch, show_metric = True)

    def train(self, X, Y, testX, testY, ckpt_dir = None, n_epoch = 10, log_path = './log_dir'):
        ckpt_path = os.path.join(ckpt_dir, "wavenet.ckpt")
        best_ckpt_path = os.path.join(ckpt_dir, "best_wavenet.ckpt")
        trainop = tflearn.TrainOp(loss = self.cost, optimizer = self.optimizer_op,
                                  metric = self.accuracy, batch_size = self.batch_size)
        trainer = tflearn.Trainer(train_ops = trainop, best_checkpoint_path = best_ckpt_path,
                                  checkpoint_path = ckpt_path, keep_checkpoint_every_n_hours = 0.5,
                                  tensorboard_dir = log_path, tensorboard_verbose = 0, session = self.sess)
        self.sess.run(tf.global_variables_initializer())
        trainer.fit({self.input_: X, self.targets: Y}, val_feed_dicts = {self.input_: testX, self.targets: testY},
                    n_epoch = n_epoch, show_metric = True)


