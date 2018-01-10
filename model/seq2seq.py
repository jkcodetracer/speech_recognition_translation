#!/usr/bin/env python
# encoding: utf-8

import tensorflow as tf
import numpy as np
import tflearn

class Seq2SeqNMT(object):

    def __init__(self, source_vocab_size, target_vocab_size, buckets, n_units, n_layers, batch_size,
                 learning_rate, n_samples = 512):
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.buckets = buckets
        self.batch_size = batch_size
        self.n_units = n_units
        self.n_layers = n_layers
        self.max_gradient_norm = 10
        self.learning_rate = tf.Variable(float(learning_rate), trainable = False, dtype = tf.float32)
        self.attention = True
        self.global_step = tf.Variable(0, trainable = False)

        # for sampled softmax
        self.n_samples = n_samples
        self.sess = tf.Session()

        self.train = True

    def init_sess(self):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def restore(self, chpt_path):
        self.sess = tf.Session()
        self.saver.restore(self.sess, chpt_path)

    def sampled_output_projection(self):
        return (self.w, self.b)

    def sampled_softmax_loss(self, labels, logits):
        labels = tf.reshape(labels, [-1, 1])
        # We need to compute the sampled_softmax_loss using 32bit floats to
        # avoid numerical instabilities.
        '''
        local_w_t = tf.cast(self.w_t, tf.float32)
        local_b = tf.cast(self.b, tf.float32)
        local_inputs = tf.cast(logits, tf.float32)
        sample_loss = tf.cast(
            tf.nn.sampled_softmax_loss(local_w_t, local_b, local_inputs, labels,
                                       self.n_samples, self.target_vocab_size), tf.float32)
        '''
        return tf.nn.sampled_softmax_loss(self.w_t, self.b, labels, logits,
                                   self.n_samples, self.target_vocab_size)

    def seq2seq_train(self, encoder_inputs, decoder_inputs):
        return tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
            encoder_inputs, decoder_inputs,
            self.cells,
            num_encoder_symbols = self.source_vocab_size,
            num_decoder_symbols = self.target_vocab_size,
            embedding_size = self.n_units,
            output_projection = self.output_projection,
            feed_previous = False,
            dtype = tf.float32
        )

    def seq2seq_predict(self, encoder_inputs, decoder_inputs):
        return tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
            encoder_inputs, decoder_inputs,
            self.cells,
            num_encoder_symbols = self.source_vocab_size,
            num_decoder_symbols = self.target_vocab_size,
            embedding_size = self.n_units,
            output_projection = self.output_projection,
            feed_previous = True,
            dtype = tf.float32
        )

    def build_graph(self, train = True):

        self.train = train
        # for softmax sampling
        self.output_projection = None
        self.softmax_loss_function = None

        # create multi-layer RNN cells
        single_cell = tf.contrib.rnn.GRUCell(self.n_units)
        single_cell = tf.contrib.rnn.DropoutWrapper(single_cell, output_keep_prob = 0.75)
        self.cells = single_cell
        if self.n_layers > 1:
            self.cells = tf.contrib.rnn.MultiRNNCell([single_cell] * self.n_layers)

        # setup the inputs
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.target_weights = []

        for i in xrange(self.buckets[-1][0]):
            self.encoder_inputs.append(tf.placeholder(tf.int32, shape = [None], name = "encoder{0}".format(i)))

        for i in xrange(self.buckets[-1][1] + 1):
            self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                name = "decoder{0}".format(i)))
            self.target_weights.append(tf.placeholder(tf.float32, shape=[None],
                name = "weight{0}".format(i)))

        # ground truth should remove <sos>
        targets = [self.decoder_inputs[i+1]
                   for i in xrange(len(self.decoder_inputs) - 1)]

        # Sampled softmax only makes sense if we sample less than vocabulary size.
        if self.n_samples > 0 and self.n_samples < self.target_vocab_size:
            self.w_t = tf.get_variable("proj_w", [self.target_vocab_size, self.n_units], dtype = tf.float32)
            self.w = tf.transpose(self.w_t)
            self.b = tf.get_variable("proj_b", [self.target_vocab_size], dtype = tf.float32)
            self.output_projection = self.sampled_output_projection()
            self.softmax_loss_function = self.sampled_softmax_loss

        # build seq2seq model
        if self.train:
            self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
                self.encoder_inputs, self.decoder_inputs, targets,
                self.target_weights, self.buckets,
                seq2seq = self.seq2seq_train,
                softmax_loss_function = self.softmax_loss_function)
        else:
            self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
                self.encoder_inputs, self.decoder_inputs, targets,
                self.target_weights, self.buckets,
                seq2seq = self.seq2seq_predict,
                softmax_loss_function = self.softmax_loss_function)
            # If we use output projection, we need to project outputs for decoding.
            if self.output_projection is not None:
                for b in xrange(len(self.buckets)):
                    self.outputs[b] = [
                        tf.matmul(output, self.output_projection[0]) + self.output_projection[1]
                        for output in self.outputs[b]
                    ]

        # optimizer
        params = tf.trainable_variables()
        self.gradient_norms = []
        self.updates = []
        # self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        opt = tf.train.RMSPropOptimizer(self.learning_rate)
        for b in xrange(len(self.buckets)):
            gradients = tf.gradients(self.losses[b], params)
            clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                             self.max_gradient_norm)
            self.gradient_norms.append(norm)
            self.updates.append(opt.apply_gradients(
                zip(clipped_gradients, params), global_step=self.global_step))
            # gradients = opt.compute_gradients(self.losses[b], var_list = params)
            # clipped_gradients, norm = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
            # self.gradient_norms.append(norm)
            # self.updates.append(opt.apply_gradients(gradients, global_step = self.global_step))

        # construct model saver
        self.saver = tf.train.Saver(tf.global_variables())

    def save(self, path):
        self.saver.save(self.sess, path, global_step = self.global_step)

    def predict(self, encoder_inputs, decoder_inputs, target_weights, bucket_id):
        encoder_size, decoder_size = self.buckets[bucket_id]
        # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
        input_feed = {}
        for l in xrange(encoder_size):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
        for l in xrange(decoder_size):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
            input_feed[self.target_weights[l].name] = target_weights[l]

        # Since our targets are decoder inputs shifted by one, we need one more.
        last_target = self.decoder_inputs[decoder_size].name
        input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

        # Output feed: depends on whether we do a backward step or not.
        output_feed = [self.losses[bucket_id]]  # Loss for this batch.
        for l in xrange(decoder_size):  # Output logits.
            output_feed.append(self.outputs[bucket_id][l])

        outputs = self.sess.run(output_feed, input_feed)
        return outputs[0], outputs[1:]  # loss, outputs.

    def train_batch(self, encoder_inputs, decoder_inputs, target_weights, bucket_id):
        encoder_size, decoder_size = self.buckets[bucket_id]
        # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
        input_feed = {}
        for l in xrange(encoder_size):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
        for l in xrange(decoder_size):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
            input_feed[self.target_weights[l].name] = target_weights[l]

        # Since our targets are decoder inputs shifted by one, we need one more.
        last_target = self.decoder_inputs[decoder_size].name
        input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

        output_feed = [self.updates[bucket_id],  # Update Op that does SGD.
                       self.gradient_norms[bucket_id],  # Gradient norm.
                       self.losses[bucket_id]]  # Loss for this batch.

        outputs = self.sess.run(output_feed, input_feed)
        return outputs[1], outputs[2] # Gradient norm, loss
        '''
        trainop = tflearn.TrainOp(loss = self.losses[bucket_id], optimizer = self.optimizer,
                                  metric = None, batch_size = self.batch_size)
        trainer = tflearn.Trainer(train_ops = trainop, tensorboard_dir = log_path, tensorboard_verbose = 0, session = self.sess)
        self.sess.run(tf.global_variables_initializer())
        loss = trainer.fit_batch(input_feed)
        #output = trainer.fit(input_feed, n_epoch = 2, show_metric = True)

        return loss
        '''




