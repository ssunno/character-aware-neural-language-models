# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import numpy as np
import tensorflow as tf

from model import Classifier
from utils import BatchGenerator
# TODO : 필요없는 코드 지우기


flags = tf.flags

# data
flags.DEFINE_string('data_dir', 'data/JDT', 'data directory. Should contain train.txt/valid.txt/test.txt with input data')  # TODO: DataReader 수정에 맞춰 변경
flags.DEFINE_string('train_dir', 'cv', 'training directory (models and summaries are saved there periodically)')
flags.DEFINE_string('load_model', None, '(optional) filename of the model to load. Useful for re-starting training from a checkpoint')
flags.DEFINE_integer('num_valid', 100, '(optional)number of dataset.')

# model params  TODO: 하이퍼파라미터 최적화 실험
flags.DEFINE_integer('rnn_size', 650, 'size of LSTM internal state')
flags.DEFINE_integer('highway_layers', 2, 'number of highway layers')
flags.DEFINE_string('kernels', '[1,2,3,4,5,6,7]', 'CNN kernel widths')
flags.DEFINE_string('kernel_features', '[50,100,150,200,200,200,200]', 'number of features in the CNN kernel')
flags.DEFINE_integer('rnn_layers', 2, 'number of layers in the LSTM')
flags.DEFINE_float('dropout_keep_prob', 0.5, 'dropout keep probability. 1.0 = no dropout')

# optimization  TODO: 하이퍼파라미터 최적화 실험
flags.DEFINE_float('learning_rate_decay', 0.5, 'learning rate decay')
flags.DEFINE_float('learning_rate', 0.01, 'starting learning rate')
flags.DEFINE_float('decay_when', 1.0, 'decay if validation perplexity does not improve by more than this much')
flags.DEFINE_float('param_init', 0.05, 'initialize parameters at')
flags.DEFINE_integer('num_unroll_steps', 100, 'number of timesteps to unroll for')
flags.DEFINE_integer('batch_size', 20, 'number of sequences to train on in parallel')
flags.DEFINE_integer('max_epochs', 25, 'number of full passes through the training data')
flags.DEFINE_float('max_grad_norm', 5.0, 'normalize gradients at')
flags.DEFINE_integer('max_word_length', 30, 'maximum word length')  # TODO : 특수문자가 많이 포함되기 때문에, word 단위가 적절한지 고민해야 함.

# bookkeeping
flags.DEFINE_integer('seed', 3435, 'random number generator seed')
flags.DEFINE_integer('print_every', 5, 'how often to print current loss')

FLAGS = flags.FLAGS


def main(_):
    """ Trains model from data """

    if not os.path.exists(FLAGS.train_dir):
        os.mkdir(FLAGS.train_dir)
        print('Created training directory', FLAGS.train_dir)

    data_reader = BatchGenerator(FLAGS.data_dir, FLAGS.batch_size, FLAGS.max_word_length, sentence_limit=FLAGS.max_word_length * FLAGS.num_unroll_steps, num_valid=FLAGS.num_valid)
    print('initialized all dataset readers')
    FLAGS.char_vocab_size = len(data_reader.chars_dict)
    FLAGS.char_embed_size = round(FLAGS.char_vocab_size * 0.66)
    FLAGS.num_classes = data_reader.num_classes
    with tf.Graph().as_default(), tf.Session() as sess:
        # tensorflow seed must be inside graph
        tf.set_random_seed(FLAGS.seed)
        np.random.seed(seed=FLAGS.seed)
        model = Classifier()
        sess.run(tf.global_variables_initializer())
        # initialzie model
        # start training
        for current_epoch in range(FLAGS.max_epochs):
            start = time.time()
            # training step
            for batch_x, batch_y in data_reader.batches():
                current_step = tf.train.global_step(sess, model.global_step)
                feed = {model.train_data: batch_x, model.targets: batch_y, model.learning_rate: FLAGS.learning_rate,
                        model.dropout_keep_prob: FLAGS.dropout_keep_prob}
                _, step, loss, accuracy = sess.run([model.train_op, model.global_step, model.loss, model.accuracy], feed_dict=feed)
                print("{}/{} ({} epochs) step, loss : {:.6f}, accuracy : {:.3f}, time/batch : {:.3f}sec"
                      .format(current_step, data_reader.num_batches * FLAGS.max_epochs, current_epoch, loss, accuracy, time.time() - start))
                start = time.time()
            # model test step
            avg_loss, avg_accuracy = 0.0, 0.0
            start = time.time()
            for valid_x, valid_y in data_reader.valid_batches():
                feed = {model.train_data: valid_x, model.targets: valid_y,
                        model.dropout_keep_prob: 1.0, model.learning_rate: FLAGS.learning_rate}
                loss, accuracy, eval_summary = sess.run([model.loss, model.accuracy], feed_dict=feed)
                avg_accuracy += accuracy * len(valid_x)
                avg_loss += loss * len(valid_x)
            print("({} epochs) evaluation step, loss : {:.6f}, accuracy : {:.3f}, time/batch : {:.3f}sec"
                  .format(current_epoch, avg_loss / len(data_reader.valid_data),
                          avg_accuracy / len(data_reader.valid_data), time.time() - start))


if __name__ == "__main__":
    tf.app.run()
