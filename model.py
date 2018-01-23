# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division


import tensorflow as tf
import tensorflow.contrib.layers as layers
FLAGS = tf.flags.FLAGS


class Classifier:

    def __init__(self):
        self.train_data = tf.placeholder(tf.int32, [None, FLAGS.num_unroll_steps, FLAGS.max_word_length])
        self.targets = tf.placeholder(tf.int32, [None])
        self.learning_rate = tf.placeholder(tf.float32)
        self.dropout_keep_prob = tf.placeholder(tf.float32)
        self.weight_decay = FLAGS.weight_decay
        self.normalize_decay = FLAGS.normalize_decay
        self.kernel_list = [1, 2, 3, 4, 5, 6, 7]
        self.kernel_features = [50, 100, 150, 200, 200, 200, 200]
        # inference character-aware-neural language model
        logits = self.inference()
        # calculate loss and accuracy
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.targets, logits=logits, name="loss")
        self.loss = tf.reduce_mean(losses)
        correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), self.targets)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # applying optimizer
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        tvars = tf.trainable_variables()
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        grads, global_norm = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), FLAGS.max_grad_norm)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)

    def inference(self):
        with tf.variable_scope('embedding'):
            char_embedding = tf.get_variable('char_embedding', [FLAGS.char_vocab_size, FLAGS.char_embed_size])
            input_embedded = tf.nn.embedding_lookup(char_embedding, self.train_data)
        #    input_embedded = tf.reshape(input_embedded, [-1, FLAGS.max_word_length, FLAGS.char_embed_size])

        net = self.__tdnn(input_embedded)
        words = [tf.squeeze(x, [1]) for x in tf.split(net, FLAGS.num_unroll_steps, 1)]
        words = [tf.expand_dims(self.__highway(word, word.get_shape()[-1]), 1) for word in words]
        words = tf.expand_dims(tf.concat(words, 1), 2)
        with tf.variable_scope('CNN'):
            net = self.__conv2d(words, 550, [3, 1], scope='conv_0')
            net = layers.max_pool2d(net, [3, 1], scope="max_pool_0")
            net = self.__conv2d(net, FLAGS.num_classes, [1, 1], scope='conv_1')
            net = layers.avg_pool2d(net, [net.get_shape()[1], 1], stride=1, scope='avg_pool_0')
            logits = tf.squeeze(net, [1, 2])
        return logits

    def __create_rnn_cell(self, rnn_size):
        return tf.contrib.rnn.BasicLSTMCell(rnn_size, state_is_tuple=True, forget_bias=0.0)

    def __conv2d_backup(self, input_tensor, output_size, kernel_size, scope='conv'):
        with tf.variable_scope(scope):
            w = tf.get_variable('w', [kernel_size[0], kernel_size[1], input_tensor.get_shape()[-1], output_size])
            b = tf.get_variable('b', [output_size])
        return tf.nn.conv2d(input_tensor, w, strides=[1, 1, 1, 1], padding='VALID') + b

    def __conv2d(self, input_tensor, num_outputs, kernel_size, stride=1, scope=None, padding="VALID", is_training=True):
        # convolution layer with default parameter
        return layers.conv2d(input_tensor, num_outputs, kernel_size, stride=stride, scope=scope,
                             data_format="NHWC", padding=padding,
                             weights_regularizer=layers.l2_regularizer(self.weight_decay),
                             normalizer_fn=layers.batch_norm,
                             normalizer_params={'is_training': is_training, 'fused': True, 'decay': self.normalize_decay})

    def __tdnn(self, input_tensor, scope='TDNN'):
        layer_list = []
        with tf.variable_scope(scope):
            for kernel, size in zip(self.kernel_list, self.kernel_features):
                conv = self.__conv2d(input_tensor, size, [1, kernel], scope='kernel_%d' % kernel)
                pool = tf.nn.max_pool(tf.tanh(conv), [1, 1, FLAGS.max_word_length - kernel + 1, 1], [1, 1, 1, 1], padding='VALID')
                layer_list.append(tf.squeeze(pool, [2]))
        return tf.concat(layer_list, 2) if len(layer_list) > 1 else layer_list[0]

    def __linear(self, input_tensor, output_size, scope='linear'):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            w = tf.get_variable("w", [output_size, input_tensor.get_shape()[-1]], dtype=tf.float32)
            b = tf.get_variable('b', [output_size], dtype=tf.float32)
        return tf.matmul(input_tensor, tf.transpose(w)) + b

    def __highway(self, input_tensor, size, bias=-2.0, scope='highway'):  # TODO : test with conv2d highway
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            for idx in range(FLAGS.highway_layers):
                g = tf.nn.relu(self.__linear(input_tensor, size, scope='linear_%d' % idx))
                t = tf.sigmoid(self.__linear(input_tensor, size, scope='gate_%d' % idx) + bias)
                output = tf.add(tf.multiply(g, t), tf.multiply(input_tensor, tf.subtract(1.0, t)), name='y')
                input_tensor = output
        return output
